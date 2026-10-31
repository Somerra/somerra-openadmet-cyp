"""Execute a frozen recipe with shared data, telemetry, and challenge evaluation."""

from __future__ import annotations

import importlib.util
import random
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chemprop import data

from .cyp_research import read, regression_metrics, tasks_for, verify, write
from .download import sha256_file
from .research_tracking import Tracker


def loss_parts(prediction, target, weights):
    mask = torch.isfinite(target)
    errors = (prediction - torch.nan_to_num(target, nan=0.0)).abs() * mask
    counts = mask.sum(0)
    if counts.sum() == 0:
        raise ValueError("Batch has no observed labels")
    sums = errors.sum(0)
    return (sums * weights).sum() / counts.sum(), sums.detach(), counts.detach()


def normalize(values):
    mean, scale = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Empty, nonfinite or constant training head")
    return mean, scale


def dataset(frame, targets):
    ds = data.MoleculeDataset(
        [
            data.MoleculeDatapoint.from_smi(s, y)
            for s, y in zip(frame.SMILES, targets.astype(np.float32), strict=True)
        ]
    )
    ds.cache = True
    return ds


def batch_indices(size, batch_size, rng=None):
    indices = np.arange(size)
    if rng is not None:
        rng.shuffle(indices)
    # Merge a final singleton into its predecessor for BatchNorm, never drop a row.
    batches = [indices[i : i + batch_size] for i in range(0, size, batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] = np.concatenate(batches[-2:])
        batches.pop()
    return batches


def batch(ds, indices, device):
    value = data.collate_batch([ds[int(i)] for i in indices])
    value.bmg.to(device)
    return value.bmg, value.Y.to(device)


def predict(model, ds, device):
    model.eval()
    rows = []
    with torch.inference_mode(), torch.autocast(device.type, enabled=False):
        for indices in batch_indices(len(ds), 256):
            graph, _ = batch(ds, indices, device)
            rows.append(model(graph).float().cpu().numpy())
    result = np.concatenate(rows)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite predictions")
    return result


def load_recipe(path):
    spec = importlib.util.spec_from_file_location("trial_recipe", path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    return recipe


def recipe_assets(recipe):
    """Declare pretrained weights or other local inputs explicitly in ASSETS."""
    return {
        str(Path(path).resolve()): sha256_file(Path(path)) for path in getattr(recipe, "ASSETS", [])
    }


class RecipeData:
    """Recipes may replace graph inputs with tensors, fingerprints, tokens, etc."""

    def __init__(self, recipe, settings):
        self.recipe, self.settings = recipe, settings

    def dataset(self, frame, targets):
        if hasattr(self.recipe, "build_dataset"):
            return self.recipe.build_dataset(frame, targets, self.settings)
        return dataset(frame, targets)

    def batch(self, ds, indices, device):
        return getattr(self.recipe, "batch", batch)(ds, indices, device)

    def predict(self, model, ds, device):
        model.eval()
        rows = []
        with torch.inference_mode(), torch.autocast(device.type, enabled=False):
            for indices in batch_indices(len(ds), self.settings["batch_size"]):
                inputs, _ = self.batch(ds, indices, device)
                rows.append(model(inputs).float().cpu().numpy())
        result = np.concatenate(rows)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite predictions")
        return result


def recipe_tasks(recipe, cyp):
    available = tasks_for(cyp)
    selected = (
        list(recipe.select_tasks(cyp, available)) if hasattr(recipe, "select_tasks") else available
    )
    if (
        not selected
        or selected[0] != available[0]
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(available)
    ):
        raise ValueError("Tasks must start with the focal CYP and use available auxiliary heads")
    return selected


def objective(recipe, prediction, target, weights, context):
    loss = recipe.compute_loss(prediction, target, weights, context)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("Recipe loss must be a finite scalar")
    return loss


def loss_context(frame, tasks, mean, scale, device, settings, epoch, phase):
    return {
        "settings": settings,
        "epoch": epoch,
        "phase": phase,
        "tasks": tasks,
        "mean": torch.tensor(mean, device=device, dtype=torch.float32),
        "scale": torch.tensor(scale, device=device, dtype=torch.float32),
        "primary_low": torch.tensor(
            frame[tasks[0] + "_conf_low"].to_numpy(float), device=device, dtype=torch.float32
        ),
        "primary_high": torch.tensor(
            frame[tasks[0] + "_conf_high"].to_numpy(float), device=device, dtype=torch.float32
        ),
    }


def validate_settings(settings):
    integer_keys = [
        "batch_size",
        "max_epochs",
        "patience",
    ]
    for key in integer_keys:
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Invalid {key}")
    if settings["batch_size"] < 2:
        raise ValueError("Batch size must be >=2")


def train(job):
    root, output = Path(job["root"]), Path(job["output"])
    if sha256_file(root / "manifest.json") != job["manifest_sha256"]:
        raise ValueError("Campaign manifest changed")
    manifest = verify(root)
    if sha256_file(output / "recipe.py") != job["recipe_sha256"]:
        raise ValueError("Recipe snapshot changed")
    for name, digest in job["runtime_sha256"].items():
        if sha256_file(Path(__file__).parent / name) != digest:
            raise ValueError("Runtime changed after this trial was recorded")
    seed = manifest["config"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if not job["smoke"] and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --smoke for CPU integration testing")
    device = torch.device("cpu" if job["smoke"] else "cuda")
    torch.set_float32_matmul_precision("highest")
    recipe = load_recipe(output / "recipe.py")
    settings = dict(recipe.SETTINGS)
    if job["smoke"]:
        settings.update(getattr(recipe, "SMOKE_SETTINGS", {}))
        settings.update(max_epochs=3, patience=2)
    validate_settings(settings)
    tasks = recipe_tasks(recipe, job["cyp"])
    train_frame, val_frame = [pd.read_csv(root / f"{n}.csv") for n in ["train", "validation"]]
    if hasattr(recipe, "select_training_rows"):
        mask = np.asarray(recipe.select_training_rows(train_frame.copy(), tasks, settings))
        if mask.dtype != bool or mask.shape != (len(train_frame),):
            raise ValueError("Training row selector must return a boolean mask of available rows")
        train_frame = train_frame.loc[mask].copy()
    train_frame = train_frame.loc[train_frame[tasks].notna().any(axis=1)].copy()
    if len(train_frame) < 2:
        raise ValueError("Need at least two training rows")
    mean, scale = normalize(train_frame[tasks].to_numpy(float))
    if job["smoke"]:
        # Integration only: intentionally tiny cohorts, never eligible for promotion.
        train_frame = train_frame.loc[train_frame[tasks[0]].notna()].sample(
            n=min(96, train_frame[tasks[0]].notna().sum()), random_state=seed
        )
        val_frame = val_frame.loc[val_frame[tasks[0]].notna()].head(48).copy()
    train_values, val_values = [f[tasks].to_numpy(float) for f in [train_frame, val_frame]]
    train_z, val_z = [(v - mean) / scale for v in [train_values, val_values]]
    adapter = RecipeData(recipe, settings)
    train_ds, val_ds = adapter.dataset(train_frame, train_z), adapter.dataset(val_frame, val_z)
    primary_train = train_frame.loc[train_frame[tasks[0]].notna()].copy()
    primary_ds = adapter.dataset(
        primary_train, (primary_train[tasks].to_numpy(float) - mean) / scale
    )
    weights = np.array(
        [
            settings.get("task_weights", {}).get(
                t, settings["primary_weight" if i == 0 else "auxiliary_weight"]
            )
            for i, t in enumerate(tasks)
        ]
    )
    if not np.isfinite(weights).all() or (weights < 0).any() or weights[0] <= 0:
        raise ValueError("Invalid task weights")
    write(
        output / "scaler.json",
        {
            "tasks": tasks,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "weights": weights.tolist(),
        },
    )
    write(output / "settings.json", settings)
    strategy = read(output / "strategy.json")
    strategy.update(
        {
            "description": recipe.STRATEGY,
            "settings": settings,
            "tasks": tasks,
            "train_rows": len(train_frame),
            "train_label_counts": train_frame[tasks].notna().sum().to_dict(),
            "task_weights": weights.tolist(),
            "assets_sha256": recipe_assets(recipe),
            "packages": {p: version(p) for p in ["torch", "chemprop", "numpy", "pandas", "rdkit"]},
            "loss_reporting": (
                "observed-label-weighted mean of batch objectives; validation is one full batch"
            ),
        }
    )
    train_frame[["Molecule_Name"]].to_csv(output / "training-rows.csv", index=False)
    strategy["training_rows_sha256"] = sha256_file(output / "training-rows.csv")
    write(output / "strategy.json", strategy)
    model = recipe.build_model(len(tasks), settings).to(device)
    optimizer = recipe.build_optimizer(model, settings)
    steps_per_epoch = len(batch_indices(len(train_ds), settings["batch_size"]))
    scheduler = recipe.build_scheduler(optimizer, settings, steps_per_epoch)
    amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    task_weights = torch.tensor(weights, device=device, dtype=torch.float32)
    started = time.monotonic()
    deadline = started + job["reserved_seconds"] - 15
    rng = np.random.default_rng(seed)
    tracker = Tracker(
        output,
        manifest["config"]["wandb"],
        run_id=job["trial_id"],
        group=job["campaign_id"],
        cyp=job["cyp"],
        config={
            **settings,
            "cyp": job["cyp"],
            "fold": manifest["config"]["fold"],
            "seed": seed,
            "smoke": job["smoke"],
            "tasks": tasks,
            "hypothesis": job["hypothesis"],
            "recipe_sha256": job["recipe_sha256"],
            "data_sha256": manifest["files"],
            "strategy": recipe.STRATEGY,
            "selection_metric": "ST-RAE",
            "selection_direction": "minimize",
            "workflow": job.get("workflow", "research"),
        },
        mode=job["wandb_mode"],
        smoke=job["smoke"],
    )
    best, stale, steps, best_epoch = float("inf"), 0, 0, None
    try:
        for epoch in range(1, settings["max_epochs"] + 1):
            model.train()
            sums, counts = np.zeros(len(tasks)), np.zeros(len(tasks))
            total_loss, loss_count = 0.0, 0
            for indices in batch_indices(len(train_ds), settings["batch_size"], rng):
                if time.monotonic() > deadline:
                    raise TimeoutError("Trial time cap reached")
                graph, targets = adapter.batch(train_ds, indices, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device.type, dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    prediction = model(graph)
                    if prediction.shape != targets.shape or not torch.isfinite(prediction).all():
                        raise ValueError("Wrong output shape or nonfinite training predictions")
                    _, errors, observed = loss_parts(prediction.float(), targets, task_weights)
                    context = loss_context(
                        train_frame.iloc[indices],
                        tasks,
                        mean,
                        scale,
                        device,
                        settings,
                        epoch,
                        "train",
                    )
                    loss = objective(recipe, prediction.float(), targets, task_weights, context)
                count = int(observed.sum())
                total_loss += float(loss.detach()) * count
                loss_count += count
                amp.scale(loss).backward()
                amp.step(optimizer)
                amp.update()
                if scheduler is not None:
                    scheduler.step()
                sums += errors.cpu().numpy()
                counts += observed.cpu().numpy()
                steps += 1
            val_pred_z = adapter.predict(model, val_ds, device)
            val_pred = val_pred_z * scale + mean
            val_metrics = regression_metrics(val_frame, val_pred[:, 0], tasks[0])
            train_pred = adapter.predict(model, primary_ds, device) * scale + mean
            train_metrics = regression_metrics(primary_train, train_pred[:, 0], tasks[0])
            observed = np.isfinite(val_z)
            val_errors = np.where(observed, np.abs(val_pred_z - val_z), 0).sum(0)
            val_count = observed.sum(0)
            weighted = sums * weights
            with torch.no_grad():
                val_loss = objective(
                    recipe,
                    torch.as_tensor(val_pred_z, device=device),
                    torch.as_tensor(val_z, device=device, dtype=torch.float32),
                    task_weights,
                    loss_context(
                        val_frame, tasks, mean, scale, device, settings, epoch, "validation"
                    ),
                )
            event = {
                "epoch": epoch,
                "optimizer/step": steps,
                "optimizer/learning_rate": optimizer.param_groups[0]["lr"],
                "train/loss": total_loss / loss_count,
                "val/loss": float(val_loss),
                "diagnostic_mae/train_primary_contribution": float(weighted[0] / counts.sum()),
                "diagnostic_mae/train_auxiliary_contribution": float(
                    weighted[1:].sum() / counts.sum()
                ),
                "diagnostic_mae/val_primary_contribution": float(
                    val_errors[0] * weights[0] / val_count.sum()
                ),
                "diagnostic_mae/val_auxiliary_contribution": float(
                    (val_errors[1:] * weights[1:]).sum() / val_count.sum()
                ),
                "runtime/seconds": time.monotonic() - started,
                **{f"val/{k}": v for k, v in val_metrics.items()},
                **{f"train/{k}": v for k, v in train_metrics.items()},
            }
            for i, task in enumerate(tasks):
                event[f"loss/count/{task}"] = int(counts[i])
                event[f"diagnostic_mae/contribution/{task}"] = float(weighted[i] / counts.sum())
            tracker.log(event)
            write(output / "progress.json", event)
            score = val_metrics["ST-RAE"]
            if score < best:
                best, stale, best_epoch = score, 0, epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "metrics": val_metrics,
                        "tasks": tasks,
                        "settings": settings,
                        "mean": mean.tolist(),
                        "scale": scale.tolist(),
                    },
                    output / "best.pt",
                )
                exported = val_frame[["Molecule_Name", "parent"]].copy()
                exported[tasks[0]] = val_pred[:, 0]
                exported.to_csv(output / "validation-predictions.csv", index=False)
            else:
                stale += 1
            # Keep recovery evidence; interrupted trials are ineligible for promotion.
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "amp": amp.state_dict(),
                    "epoch": epoch,
                    "rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "best": best,
                },
                output / "last.pt",
            )
            if stale >= settings["patience"]:
                break
        saved = torch.load(output / "best.pt", map_location=device, weights_only=True)
        replay = recipe.build_model(len(tasks), settings).to(device)
        replay.load_state_dict(saved["model"])
        replay_pred = adapter.predict(replay, val_ds, device) * scale + mean
        recorded = pd.read_csv(output / "validation-predictions.csv")[tasks[0]].to_numpy()
        error = float(np.max(np.abs(recorded - replay_pred[:, 0])))
        if error > 1e-4:
            raise ValueError(f"Checkpoint replay differs: {error}")
        if recipe_assets(recipe) != strategy["assets_sha256"]:
            raise ValueError("Declared recipe assets changed during training")
        result = {
            "status": "complete",
            "metrics": saved["metrics"],
            "best_epoch": best_epoch,
            "epochs": epoch,
            "optimizer_steps": steps,
            "reload_max_abs": error,
            "smoke": job["smoke"],
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint_sha256": sha256_file(output / "best.pt"),
            "strategy": strategy,
            "strategy_sha256": sha256_file(output / "strategy.json"),
            "recipe_sha256": job["recipe_sha256"],
            "validation_predictions_sha256": sha256_file(output / "validation-predictions.csv"),
        }
        write(output / "result.json", result)
        tracker.finish(
            {
                "best_epoch": best_epoch,
                "status": "complete",
                **{f"selected/{k}": v for k, v in saved["metrics"].items() if v is not None},
            }
        )
    except BaseException as exc:
        tracker.finish({"status": "failed", "error_type": type(exc).__name__}, failed=True)
        raise


def main():
    job = read(sys.argv[1])
    try:
        train(job)
    except Exception as exc:
        write(
            Path(job["output"]) / "result.json",
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "strategy_sha256": sha256_file(Path(job["output"]) / "strategy.json"),
                "recipe_sha256": job["recipe_sha256"],
            },
        )
        raise


if __name__ == "__main__":
    main()
