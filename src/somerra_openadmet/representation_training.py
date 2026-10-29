"""CheMeleon frozen-head and full-tuning screens, with isolated H100 selection.

These are draft experiment runners, not automatic promotion or submission workflows.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .constants import ACTIVITY_ENDPOINTS, EXPECTED_TEST_ROWS, TEST_FILE
from .download import sha256_file
from .metrics import soft_threshold_rae
from .representations import (
    CHEMELEON_SHA256,
    Run,
    atomic_json,
    autocast,
    batch_indices,
    foundation,
    graph_batch,
    graph_dataset,
    identities,
    load_cache,
    prepared_data,
    read_config,
    requested_precision,
)


def masked_mae(prediction, target):
    import torch

    observed = torch.isfinite(target)
    counts = observed.sum(0)
    if not (counts > 0).any():
        return None
    safe = torch.where(observed, target, torch.zeros_like(target))
    errors = (prediction.float() - safe.float()).abs() * observed
    return (errors.sum(0) / counts.clamp_min(1))[counts > 0].mean()


def selected_folds(config, available):
    folds = sorted(available)
    selected = config.get("selected_folds", folds)
    if (
        not isinstance(selected, list)
        or not selected
        or any(type(fold) is not int for fold in selected)
        or len(set(selected)) != len(selected)
        or not set(selected).issubset(folds)
    ):
        raise ValueError(f"selected_folds must be unique fold IDs from {folds}")
    return sorted(selected)


def recipe_for(mode: str, config: dict):
    if mode not in {"frozen", "full"}:
        raise ValueError("adaptation must be frozen or full")
    defaults = {
        "epochs": 300 if mode == "frozen" else 60,
        "batch_size": 256 if mode == "frozen" else 64,
        "learning_rate": 1e-3 if mode == "frozen" else 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "adam" if mode == "frozen" else "adamw",
        "warmup_fraction": 0.0,
        "precision": "float32" if config.get("h100_select") else config.get("precision", "float32"),
    }
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay", "warmup_fraction"):
        defaults[key] = config.get(key, defaults[key])
    if type(defaults["epochs"]) is not int or defaults["epochs"] < 1:
        raise ValueError("epochs must be a positive integer")
    batch_indices([0, 1], defaults["batch_size"])
    if not np.isfinite(defaults["learning_rate"]) or defaults["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(defaults["weight_decay"]) or defaults["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not np.isfinite(defaults["warmup_fraction"]) or not 0 <= defaults["warmup_fraction"] < 1:
        raise ValueError("warmup_fraction must be finite and in [0, 1)")
    return defaults


def scaled_recipes(reference: dict, batch_size: int, precision="bf16"):
    ratio = batch_size / reference["batch_size"]
    multipliers = sorted({1.0, max(1.0, min(math.sqrt(ratio), 4)), max(1.0, min(ratio, 4))})
    return [reference.copy()] + [
        {
            **reference,
            "batch_size": batch_size,
            "learning_rate": reference["learning_rate"] * m,
            "warmup_fraction": 0.05,
            "precision": precision,
        }
        for m in multipliers
    ]


def inner_split(frame, fit_indices, seed):
    fit_indices = np.asarray(fit_indices)
    groups = frame.iloc[fit_indices].cluster.to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("Inner holdout requires at least two molecular groups")
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train, val = next(split.split(fit_indices, groups=groups))
    train, val = fit_indices[train], fit_indices[val]
    for name, indices in (("inner train", train), ("inner validation", val)):
        if frame.iloc[indices][list(ACTIVITY_ENDPOINTS)].notna().sum().min() == 0:
            raise ValueError(f"An endpoint has no labels in {name}; predeclare a different split")
    return train, val


def make_model(
    mode,
    checkpoint=None,
    encoder_parameters=None,
    input_dim=2048,
    head_dropout=0.0,
    target_mean=None,
    target_scale=None,
):
    import torch

    class Regressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = None
            if mode == "full":
                from chemprop import nn

                self.encoder = (
                    nn.BondMessagePassing(**encoder_parameters)
                    if encoder_parameters is not None
                    else foundation(Path(checkpoint))
                )
                self.aggregate = nn.MeanAggregation()
            layers = [
                torch.nn.Linear(input_dim, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
            ]
            if head_dropout:
                layers.append(torch.nn.Dropout(head_dropout))
            layers.append(torch.nn.Linear(512, len(ACTIVITY_ENDPOINTS)))
            self.head = torch.nn.Sequential(*layers)
            self.head_dropout = head_dropout
            self.register_buffer(
                "target_mean",
                None if target_mean is None else torch.tensor(target_mean, dtype=torch.float32),
            )
            self.register_buffer(
                "target_scale",
                None if target_scale is None else torch.tensor(target_scale, dtype=torch.float32),
            )

        def forward(self, value):
            if self.encoder is not None:
                value = self.aggregate(self.encoder(value), value.batch)
            prediction = self.head(value).float()
            if self.target_mean is not None:
                prediction = prediction * self.target_scale + self.target_mean
            return prediction

    return Regressor()


class Inputs:
    def __init__(self, frame, mode, device, features=None):
        import torch

        self.device = device
        self.features = None if features is None else torch.as_tensor(features, device=device)
        self.graphs = graph_dataset(frame.SMILES) if mode == "full" else None

    def batch(self, indices):
        if self.features is not None:
            return self.features[indices]
        return graph_batch(self.graphs, indices, self.device)


def optimizer_for(model, recipe):
    import torch

    optimizer = torch.optim.Adam if recipe["optimizer"] == "adam" else torch.optim.AdamW
    return optimizer(
        model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"]
    )


def precision_probe(model, inputs, indices, targets, requested):
    """Check BF16 forward/backward, then restore every weight and BN buffer."""
    import torch

    if requested == "float32":
        return "float32", {"requested": requested, "actual": requested}
    before = copy.deepcopy(model.state_dict())
    value = inputs.batch(indices)
    target = torch.as_tensor(targets[indices], device=inputs.device)
    result = {"requested": requested, "actual": "float32"}
    try:
        model.eval()
        with torch.no_grad():
            reference = model(value).float()
            with autocast(inputs.device, "bf16"):
                prediction = model(value).float()
            error = float((reference - prediction).abs().max())
            result["prediction_max_abs"] = error if np.isfinite(error) else None
            parity = torch.isfinite(prediction).all() and torch.allclose(
                reference, prediction, atol=0.01, rtol=0.01
            )
        model.train()
        with autocast(inputs.device, "bf16"):
            loss = masked_mae(model(value), target)
        if loss is not None and torch.isfinite(loss):
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.requires_grad]
            finite = all(g is not None and torch.isfinite(g).all() for g in gradients)
            if parity and finite:
                result["actual"] = "bf16"
        if result["actual"] != "bf16":
            result["fallback_reason"] = "BF16 parity or finite-gradient check failed"
    except (NotImplementedError, RuntimeError) as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            raise
        if not isinstance(exc, NotImplementedError) and not any(
            word in str(exc).lower() for word in ("bfloat16", "bf16")
        ):
            raise
        result["fallback_reason"] = f"Unsupported BF16 operation: {exc}"
    finally:
        model.load_state_dict(before)
        model.zero_grad(set_to_none=True)
    return result["actual"], result


def step(model, optimizer, inputs, indices, targets, precision):
    import torch

    target = torch.as_tensor(targets[indices], device=inputs.device)
    if not torch.isfinite(target).any():
        return None, 0.0
    started = time.monotonic()
    value = inputs.batch(indices)
    if inputs.device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    optimizer.zero_grad(set_to_none=True)
    with autocast(inputs.device, precision):
        prediction = model(value)
        if model.target_mean is not None:
            prediction = (prediction - model.target_mean) / model.target_scale
            target = (target - model.target_mean) / model.target_scale
        loss = masked_mae(prediction, target)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite training loss")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    if (
        any(g is None for g in gradients)
        or not torch.stack([torch.isfinite(g).all() for g in gradients]).all()
    ):
        raise ValueError("Missing/non-finite gradient in a trainable parameter")
    optimizer.step()
    return float(loss.detach()), load_seconds


def predictions(model, inputs, indices, batch_size, run=None):
    import torch

    model.eval()
    values = []
    with torch.inference_mode():
        for batch in batch_indices(indices, batch_size):
            if run is not None:
                run.check()
            # Evaluation and persisted predictions always use float32, including BF16 training.
            values.append(model(inputs.batch(batch)).float().cpu().numpy())
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite predictions")
    return result


def validation_mae(prediction, target):
    observed = np.isfinite(target)
    if (observed.sum(0) == 0).any():
        raise ValueError("Validation requires labels for each endpoint")
    return float(np.nanmean(np.nanmean(np.abs(prediction - target), axis=0)))


def projected_training_seconds(history, epochs, future_fits):
    """Exclude first-epoch startup from a conservative remaining-work estimate."""
    durations = [row["seconds"] for row in history[1:]] or [history[-1]["seconds"]]
    return float(np.median(durations)) * (epochs - len(history) + future_fits * epochs) * 1.2


def train_model(
    config, recipe, inputs, targets, indices, run, tag, *, val_indices=None, seed=42, future_fits=0
):
    import torch

    torch.manual_seed(seed)
    means, scales = None, None
    if config.get("standardize_targets", False):
        means = np.nanmean(targets[indices], axis=0).astype(np.float32)
        scales = np.maximum(np.nanstd(targets[indices], axis=0), 1e-6).astype(np.float32)
        if not np.isfinite(means).all() or not np.isfinite(scales).all():
            raise ValueError("Training-only target normalization has missing endpoints")
    model = make_model(
        config["adaptation"],
        config.get("checkpoint"),
        head_dropout=config.get("head_dropout", 0.0),
        target_mean=means,
        target_scale=scales,
    ).to(run.device)
    optimizer = optimizer_for(model, recipe)
    probe_indices = batch_indices(indices, recipe["batch_size"], merge_singleton=True)[0]
    precision, smoke = precision_probe(
        model,
        inputs,
        probe_indices,
        targets,
        requested_precision({**config, "precision": recipe["precision"]}, run.device),
    )
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    updates = 0
    exposures = 0
    total_batches = len(batch_indices(indices, recipe["batch_size"], merge_singleton=True))
    warmup = math.ceil(total_batches * recipe["epochs"] * recipe["warmup_fraction"])
    history = []
    started = time.monotonic()
    artifact = run.output / f"{tag}-training.json"
    for epoch in range(recipe["epochs"]):
        model.train()
        losses, load_seconds = [], 0.0
        epoch_start = time.monotonic()
        for batch in batch_indices(
            rng.permutation(indices), recipe["batch_size"], merge_singleton=True
        ):
            run.check()
            scale = min(1.0, (updates + 1) / warmup) if warmup else 1.0
            for group in optimizer.param_groups:
                group["lr"] = recipe["learning_rate"] * scale
            loss, loaded = step(model, optimizer, inputs, batch, targets, precision)
            if loss is not None:
                updates += 1
                exposures += len(batch)
                losses.append(loss)
                load_seconds += loaded
        if not losses:
            raise ValueError("Training epoch has no observed targets")
        run.sync()
        entry = {
            "epoch": epoch + 1,
            "training_batch_mae": float(np.mean(losses)),
            "seconds": time.monotonic() - epoch_start,
            "elapsed_seconds": time.monotonic() - started,
            "successful_updates": updates,
            "sample_exposures": exposures,
            "data_loader_seconds": load_seconds,
        }
        if val_indices is not None:
            pred = predictions(model, inputs, val_indices, recipe["batch_size"], run)
            entry["inner_validation_mae"] = validation_mae(pred, targets[val_indices])
        history.append(entry)
        record = {
            "recipe": recipe,
            "precision_probe": smoke,
            "history": history,
            "training_indices": np.asarray(indices).tolist(),
            "validation_indices": None if val_indices is None else np.asarray(val_indices).tolist(),
            "seed": seed,
            "complete": epoch + 1 == recipe["epochs"],
            "microbatch_size": recipe["batch_size"],
            "gradient_accumulation": 1,
            "loss_units": "standardized" if means is not None else "pIC50",
            "target_mean": None if means is None else means.tolist(),
            "target_scale": None if scales is None else scales.tolist(),
        }
        atomic_json(artifact, record)
        if epoch == min(2, recipe["epochs"] - 1):
            # The first epoch includes CUDA/kernel initialization. Use warmed-up
            # epochs for projection; the supervisor still enforces the hard deadline.
            estimate = projected_training_seconds(history, recipe["epochs"], future_fits)
            run.record.setdefault("cost_projections", {})[tag] = estimate
            run.save()
            run.check(estimate)
    record["wall_seconds"] = time.monotonic() - started
    record["molecules_per_second"] = exposures / record["wall_seconds"]
    atomic_json(artifact, record)
    return model, optimizer, record


def profile_batches(config, reference, inputs, targets, fit_indices, frame, run, fold):
    import torch

    candidates = (
        [256, 512, 1024, 2048] if config["adaptation"] == "frozen" else [64, 128, 256, 512, 1024]
    )
    sizes = sorted({min(size, len(fit_indices)) for size in candidates})
    # Include the longest molecules in every profiling batch to exercise activation pressure.
    ordered = np.asarray(sorted(fit_indices, key=lambda i: len(frame.iloc[i].SMILES), reverse=True))
    results = []
    for size in sizes:
        run.check()
        model = optimizer = None
        result = {"batch_size": size, "status": "oom"}
        try:
            torch.manual_seed(config.get("seed", 42) + fold)
            model = make_model(config["adaptation"], config.get("checkpoint")).to(run.device)
            optimizer = optimizer_for(model, reference)
            indices = ordered[:size]
            precision, smoke = precision_probe(
                model, inputs, indices, targets, requested_precision(config, run.device)
            )
            model.train()
            for _ in range(2):
                run.check()
                step(model, optimizer, inputs, indices, targets, precision)
            run.sync()
            start = time.monotonic()
            loading = 0.0
            for _ in range(5):
                run.check()
                _, duration = step(model, optimizer, inputs, indices, targets, precision)
                loading += duration
            run.sync()
            seconds = time.monotonic() - start
            result.update(
                status="ok",
                molecules_per_second=5 * size / seconds,
                optimizer_steps_per_second=5 / seconds,
                data_loader_seconds=loading,
                precision_probe=smoke,
                peak_allocated_bytes=torch.cuda.max_memory_allocated()
                if run.device == "cuda"
                else 0,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if run.device == "cuda" else 0,
                memory_stat_scope="cumulative_run_peak",
            )
        except torch.cuda.OutOfMemoryError:
            pass
        finally:
            del model, optimizer
            gc.collect()
            if run.device == "cuda":
                torch.cuda.empty_cache()
        results.append(result)
        atomic_json(run.output / f"fold-{fold}-profile.json", results)
    successful = [r for r in results if r["status"] == "ok"]
    if not successful:
        raise RuntimeError("No profiling batch fits the memory allowance")
    best = max(r["molecules_per_second"] for r in successful)
    return min(r["batch_size"] for r in successful if r["molecules_per_second"] >= 0.95 * best)


def save_checkpoint(path, model, optimizer, metadata):
    import torch

    parameters = None
    if model.encoder is not None:
        # Convert Chemprop's str-enums/Lightning metadata to plain JSON primitives.
        # This keeps new checkpoints loadable with weights_only=True without allowlists.
        parameters = json.loads(
            json.dumps({k: v for k, v in model.encoder.hparams.items() if k != "cls"})
        )
    saved = {
        **metadata,
        "schema_version": 1,
        "encoder_parameters": parameters,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "tasks": list(ACTIVITY_ENDPOINTS),
        "input_dim": model.head[0].in_features,
        "head_dropout": model.head_dropout,
        "target_mean": None if model.target_mean is None else model.target_mean.cpu().tolist(),
        "target_scale": None if model.target_scale is None else model.target_scale.cpu().tolist(),
    }
    temporary = path.with_suffix(".pt.partial")
    torch.save(saved, temporary)
    os.replace(temporary, path)


def restore_checkpoint(path, device="cpu"):
    import torch

    saved = torch.load(path, weights_only=True, map_location="cpu")
    if saved.get("schema_version") != 1 or saved["tasks"] != list(ACTIVITY_ENDPOINTS):
        raise ValueError("Unsupported representation checkpoint")
    if saved["checkpoint_sha256"] != CHEMELEON_SHA256:
        raise ValueError("Unexpected foundation provenance")
    model = make_model(
        saved["adaptation"],
        encoder_parameters=saved["encoder_parameters"],
        input_dim=saved["input_dim"],
        head_dropout=saved.get("head_dropout", 0.0),
        target_mean=saved.get("target_mean"),
        target_scale=saved.get("target_scale"),
    )
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval(), saved


def oof_rows(frame, indices, prediction):
    held = frame.iloc[indices]
    rows = []
    for column, endpoint in enumerate(ACTIVITY_ENDPOINTS):
        observed = held[endpoint].notna()
        part = held.loc[observed, ["Molecule_Name", "parent", "cluster", "fold"]].copy()
        part["endpoint"] = endpoint
        part["y_true"] = held.loc[observed, endpoint]
        part["conf_low"] = held.loc[observed, endpoint + "_conf_low"]
        part["conf_high"] = held.loc[observed, endpoint + "_conf_high"]
        part["y_pred"] = prediction[observed.to_numpy(), column]
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def profile_representations(config_path):
    """Throughput-only profiling in a supervised, label-free worker."""
    from .representation_budget import maybe_supervise

    config = read_config(
        config_path,
        {"prepared_dir", "cache_dir", "checkpoint"},
        {"prepared_dir", "cache_dir", "checkpoint", "output_dir", "max_gpu_hours"},
    )
    supervised = maybe_supervise("profile", config_path)
    if supervised is not None:
        return supervised
    frame, manifest = prepared_data(Path(config["prepared_dir"]))
    features, _ = load_cache(Path(config["cache_dir"]), frame)
    recipes = {}
    with Run(config) as run:
        targets = np.zeros((len(frame), len(ACTIVITY_ENDPOINTS)), dtype=np.float32)
        for mode in ("frozen", "full"):
            model_config = {**config, "adaptation": mode, "precision": "bf16"}
            reference = recipe_for(mode, {})
            inputs = Inputs(frame, mode, run.device, features if mode == "frozen" else None)
            batch = profile_batches(
                model_config,
                reference,
                inputs,
                targets,
                np.arange(len(frame)),
                frame,
                run,
                0 if mode == "frozen" else 1,
            )
            scale = max(1.0, min(math.sqrt(batch / reference["batch_size"]), 4.0))
            recipes[mode] = {
                **reference,
                "batch_size": batch,
                "learning_rate": reference["learning_rate"] * scale,
                "warmup_fraction": 0.05,
                "precision": "bf16",
            }
            del inputs
            gc.collect()
        run.record.update(
            complete=True,
            recipes=recipes,
            real_labels_used=False,
            training_sha256=manifest["training_sha256"],
        )
    return run.record


def screen_representations(config_path):
    from .representation_budget import maybe_supervise

    supervised = maybe_supervise("screen", config_path)
    if supervised is not None:
        return supervised
    config = read_config(
        config_path,
        {
            "prepared_dir",
            "checkpoint",
            "cache_dir",
            "adaptation",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "warmup_fraction",
            "selected_folds",
            "h100_select",
            "head_dropout",
            "standardize_targets",
        },
        {"prepared_dir", "adaptation", "output_dir", "max_gpu_hours"},
    )
    mode = config["adaptation"]
    recipe = recipe_for(mode, config)
    if not 0 <= config.get("head_dropout", 0.0) < 1:
        raise ValueError("head_dropout must be in [0, 1)")
    if type(config.get("standardize_targets", False)) is not bool:
        raise ValueError("standardize_targets must be boolean")
    if type(config.get("h100_select", False)) is not bool:
        raise ValueError("h100_select must be boolean")
    frame, prepared = prepared_data(Path(config["prepared_dir"]))
    if prepared["n_folds"] != 5:
        raise ValueError("Representation comparison requires the common five-fold manifest")
    targets = frame[list(ACTIVITY_ENDPOINTS)].to_numpy(dtype=np.float32)
    folds = sorted(int(f) for f in frame.fold.unique())
    requested = selected_folds(config, folds)
    features, cache = None, None
    if mode == "frozen":
        if not config.get("cache_dir"):
            raise ValueError("Frozen training requires cache_dir")
        features, cache = load_cache(Path(config["cache_dir"]), frame)
    else:
        if not config.get("checkpoint"):
            raise ValueError("Full tuning requires checkpoint")
        # Verify provenance before reserving GPU time.
        del_model = foundation(Path(config["checkpoint"]))
        del del_model
    with Run(config) as run:
        run.record.update(
            prepared_manifest=prepared,
            checkpoint_sha256=CHEMELEON_SHA256,
            cache_manifest=cache,
            completed_folds=[],
            selected_folds=requested,
            requested_folds_complete=False,
            checkpoint_hashes={},
            evaluation_precision="float32",
            promoted=False,
        )
        run.save()
        inputs = Inputs(frame, mode, run.device, features)
        rows = []
        for position, fold in enumerate(requested):
            fit = np.flatnonzero(frame.fold.to_numpy() != fold)
            holdout = np.flatnonzero(frame.fold.to_numpy() == fold)
            if (np.isfinite(targets[fit]).sum(0) == 0).any():
                raise ValueError("An endpoint has no outer-training labels")
            seed = config.get("seed", 42) + fold
            chosen = recipe.copy()
            if config.get("h100_select", False):
                inner_fit, inner_val = inner_split(frame, fit, seed)
                batch = profile_batches(
                    config, recipe, inputs, targets, inner_fit, frame, run, fold
                )
                candidates = scaled_recipes(recipe, batch, config.get("precision", "bf16"))
                trials = []
                for trial, candidate in enumerate(candidates):
                    model, optimizer, record = train_model(
                        config,
                        candidate,
                        inputs,
                        targets,
                        inner_fit,
                        run,
                        f"fold-{fold}-trial-{trial}",
                        val_indices=inner_val,
                        seed=seed,
                        future_fits=len(candidates) - trial,
                    )
                    trials.append(
                        {
                            "recipe": candidate,
                            "mae": record["history"][-1]["inner_validation_mae"],
                            "wall_seconds": record["wall_seconds"],
                        }
                    )
                    del model, optimizer
                    gc.collect()
                chosen = min(
                    trials,
                    key=lambda t: (t["mae"], t["wall_seconds"], t["recipe"]["learning_rate"]),
                )["recipe"]
                atomic_json(
                    run.output / f"fold-{fold}-selection.json",
                    {
                        "inner_train_ids": frame.iloc[inner_fit].Molecule_Name.tolist(),
                        "inner_validation_ids": frame.iloc[inner_val].Molecule_Name.tolist(),
                        "trials": trials,
                        "chosen": chosen,
                    },
                )
            model, optimizer, training = train_model(
                config,
                chosen,
                inputs,
                targets,
                fit,
                run,
                f"fold-{fold}",
                seed=seed,
                future_fits=0 if config.get("h100_select") else len(requested) - position - 1,
            )
            # Outer labels are first used for scoring after all training decisions are fixed.
            prediction = predictions(model, inputs, holdout, chosen["batch_size"], run)
            checkpoint = run.output / f"fold-{fold}.pt"
            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                {
                    "adaptation": mode,
                    "checkpoint_sha256": CHEMELEON_SHA256,
                    "fold": fold,
                    "seed": seed,
                    "recipe": chosen,
                    "training_sha256": prepared["training_sha256"],
                    "cache_manifest": cache,
                    "training": training,
                },
            )
            # Verify the full held-out prediction table on CPU before accepting the fold.
            replay_model, _ = restore_checkpoint(checkpoint)
            replay_inputs = Inputs(
                frame.iloc[holdout], mode, "cpu", None if features is None else features[holdout]
            )
            replay = predictions(
                replay_model, replay_inputs, np.arange(len(holdout)), chosen["batch_size"], run
            )
            difference = float(np.max(np.abs(prediction - replay)))
            run.record.setdefault("cpu_replay_max_abs", {})[str(fold)] = difference
            run.save()
            if difference > 1e-4:
                raise ValueError(f"CPU replay exceeds 1e-4 pIC50: {difference}")
            rows.append(oof_rows(frame, holdout, prediction))
            pd.concat(rows, ignore_index=True).to_csv(
                run.output / "oof_predictions.csv", index=False
            )
            run.record["completed_folds"].append(fold)
            run.record["checkpoint_hashes"][str(fold)] = sha256_file(checkpoint)
            run.save()
            del model, optimizer, replay_model, replay_inputs
            gc.collect()
        run.record["requested_folds_complete"] = True
        run.record["oof_sha256"] = sha256_file(run.output / "oof_predictions.csv")
        if requested == folds:
            oof = pd.concat(rows, ignore_index=True)
            scores = {}
            for endpoint in ACTIVITY_ENDPOINTS:
                subset = oof.loc[oof.endpoint == endpoint]
                scores[endpoint] = {
                    "ST-RAE": soft_threshold_rae(
                        subset.y_true, subset.y_pred, subset.conf_low, subset.conf_high
                    ),
                    "MAE": float(np.mean(np.abs(subset.y_true - subset.y_pred))),
                    "observed_labels": len(subset),
                }
            metrics = {
                "endpoints": scores,
                "MA-ST-RAE": float(np.mean([v["ST-RAE"] for v in scores.values()])),
                "purpose": "screening_only",
            }
            atomic_json(run.output / "metrics.json", metrics)
            run.record.update(complete=True, metrics=metrics)
        run.save()
    return run.record


def predict_representations(config_path):
    from .representation_budget import maybe_supervise

    supervised = maybe_supervise("predict", config_path)
    if supervised is not None:
        return supervised
    config = read_config(
        config_path,
        {"run_dir", "blind_csv", "cache_dir", "batch_size"},
        {"run_dir", "blind_csv", "output_dir", "max_gpu_hours"},
    )
    source = Path(config["run_dir"])
    evidence = json.loads((source / "run_manifest.json").read_text())
    if not evidence.get("complete") or evidence.get("completed_folds") != list(range(5)):
        raise ValueError("Blind export requires a completed five-fold run")
    if sha256_file(source / "oof_predictions.csv") != evidence["oof_sha256"]:
        raise ValueError("OOF predictions changed")
    blind_path = Path(config["blind_csv"])
    if sha256_file(blind_path) != evidence["prepared_manifest"]["input_sha256"][TEST_FILE]:
        raise ValueError("Blind input changed")
    frame = pd.read_csv(blind_path)
    identities(frame)
    if len(frame) != EXPECTED_TEST_ROWS:
        raise ValueError("Expected 750 blind molecules")
    mode = evidence["config"]["adaptation"]
    features = None
    if mode == "frozen":
        cache_dir = Path(config.get("cache_dir", evidence["config"]["cache_dir"]))
        features, cache = load_cache(cache_dir, frame)
        if cache != evidence["cache_manifest"]:
            raise ValueError("Prediction cache differs from training representation provenance")
    with Run(config) as run:
        inputs = Inputs(frame, mode, run.device, features)
        values = []
        for fold in evidence["completed_folds"]:
            path = source / f"fold-{fold}.pt"
            if sha256_file(path) != evidence["checkpoint_hashes"][str(fold)]:
                raise ValueError("Fold checkpoint changed")
            model, saved = restore_checkpoint(path, run.device)
            if (
                saved["fold"] != fold
                or saved["training_sha256"] != evidence["prepared_manifest"]["training_sha256"]
            ):
                raise ValueError("Checkpoint split provenance mismatch")
            values.append(
                predictions(
                    model, inputs, np.arange(len(frame)), config.get("batch_size", 256), run
                )
            )
            del model
        output = frame[["Molecule_Name", "SMILES"]].copy()
        output[list(ACTIVITY_ENDPOINTS)] = np.mean(values, axis=0)
        path = run.output / "activity_submission.csv"
        output.to_csv(path, index=False)
        from .validation import validate_submission

        validation = validate_submission(path, frame, "activity")
        if not validation.valid:
            raise ValueError(f"Invalid prediction file: {validation.errors}")
        run.record.update(
            complete=True,
            validation=validation.to_dict(),
            submission_sha256=sha256_file(path),
            source_manifest_sha256=sha256_file(source / "run_manifest.json"),
            status="candidate_only_not_promoted_or_uploaded",
        )
    return run.record
