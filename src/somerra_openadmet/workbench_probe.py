"""Matched frozen encoders on pinned Workbench auxiliary-task folds; no calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .constants import ACTIVITY_ENDPOINTS
from .data_sources.workbench import MEMBER_TASKS
from .download import sha256_file
from .frozen_benchmark import load_features, predict_head
from .reference_member import score_predictions
from .representations import Run, atomic_json, batch_indices

PROTOCOL = "workbench-mt-aux8-frozen-v1"
TASKS = MEMBER_TASKS["mt_aux8"]
VIEWS = {"chemeleon": "mean", "mol_jepa": "cls", "atom_jepa": "scalar"}
# Pinned historical assignments distinguish these stereochemical scaffolds.
# Preserve exact Workbench parity and report the parent overlap; never add new exceptions.
PRESERVED_OVERLAPS = {"LOUPRKONTZGTKE": [0, 1], "MJIHNNLFOKEZEW": [0, 4]}
RECIPE = {
    "hidden_dim": 512,
    "dropout": 0.2,
    "batch_size": 256,
    "max_epochs": 400,
    "patience": 40,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "Adam",
    "precision": "float32",
    "auxiliary_weight": 0.3,
    "selection": "minimum weighted standardized validation MAE across all eight tasks",
    "loss_denominator": "number of observed labels, not sum of weights",
    "target_scaling": "training-fold population mean and SD, minimum SD 1e-6",
    "feature_scaling": "training-fold population mean and SD, minimum SD 1e-6",
    "seed": "42 + fold; one fit per fold, no repeat seeds",
    "calibration": "none",
}
PIN = Path(__file__).resolve().parents[2] / "docs/results/workbench-protocol-artifacts.json"


def implementation_hashes():
    return {
        name: sha256_file(Path(__file__).parent / name)
        for name in (
            "workbench_probe.py",
            "frozen_benchmark.py",
            "representations.py",
            "representation_budget.py",
            "campaign_budget.py",
            "reference_member.py",
            "metrics.py",
            "mol_jepa.py",
            "atom_jepa.py",
            "constants.py",
            "download.py",
            "data_sources/workbench.py",
        )
    }


def load_bundle(directory):
    """Verify original bytes and row alignment before using the saved fold column."""
    directory = Path(directory)
    pins = json.loads(PIN.read_text())["sha256"]
    for relative, expected in pins.items():
        if sha256_file(directory.parent / relative) != expected:
            raise ValueError(f"Pinned Workbench artifact changed: {relative}")
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256_file(directory / "official.csv") != manifest["file_sha256"]["official.csv"]:
        raise ValueError("Official labels changed")
    frame = pd.read_csv(directory / "mt_aux8.csv")
    official = pd.read_csv(directory / "official.csv").set_index("molecule_name")
    if frame.molecule_name.duplicated().any() or official.index.duplicated().any():
        raise ValueError("Duplicate molecule IDs")
    if set(frame.molecule_name) != set(official.index):
        raise ValueError("Workbench/official identities differ")
    official = official.loc[frame.molecule_name].reset_index()
    if not frame.smiles.equals(official.smiles):
        raise ValueError("Workbench/official structures differ")
    if not np.allclose(frame[TASKS[:4]], official[TASKS[:4]], equal_nan=True):
        raise ValueError("Workbench/official scored labels differ")
    validate_folds(frame, PRESERVED_OVERLAPS)
    truth = official.rename(columns={"molecule_name": "Molecule_Name"}).copy()
    for ep in ACTIVITY_ENDPOINTS:
        truth[ep] = truth[ep.lower()]
        truth[ep + "_conf_low"] = truth[ep.lower() + "_ci_lower"]
        truth[ep + "_conf_high"] = truth[ep.lower() + "_ci_upper"]
    return (
        frame,
        truth,
        {
            "pinned_artifacts": pins,
            "official_sha256": manifest["file_sha256"]["official.csv"],
            "tasks": TASKS,
            "rows": len(frame),
            "observed_labels": frame[TASKS].notna().sum().to_dict(),
            "historical_cross_fold_connectivities": frame.loc[
                frame.connectivity.isin(PRESERVED_OVERLAPS),
                ["molecule_name", "connectivity", "scaffold", "fold"],
            ].to_dict("records"),
        },
    )


def validate_folds(frame, allowed_overlaps=None):
    if frame.fold.isna().any() or set(frame.fold) != set(range(5)):
        raise ValueError("Expected all five saved folds")
    if frame.groupby("scaffold", dropna=False).fold.nunique().max() != 1:
        raise ValueError("A scaffold crosses folds")
    connectivity_folds = frame.groupby("connectivity").fold.agg(lambda x: sorted(set(x)))
    overlaps = connectivity_folds[connectivity_folds.map(len) > 1].to_dict()
    if overlaps != (allowed_overlaps or {}):
        raise ValueError("A connectivity crosses folds outside the exact historical exceptions")
    return [(np.flatnonzero(frame.fold != k), np.flatnonzero(frame.fold == k)) for k in range(5)]


def training_statistics(features, targets, auxiliary_weight=0.3):
    if not np.isfinite(features).all() or np.isinf(targets).any():
        raise ValueError("Invalid features or labels")
    counts = np.isfinite(targets).sum(0)
    if targets.shape[1] != 8 or (counts == 0).any():
        raise ValueError("Every training head needs labels")
    primary = 1.0 / counts[:4]
    weights = np.r_[primary / primary.mean(), np.full(4, auxiliary_weight)]
    return {
        "feature_mean": features.mean(0).astype(np.float32),
        "feature_scale": np.maximum(features.std(0), 1e-6).astype(np.float32),
        "target_mean": np.nanmean(targets, axis=0).astype(np.float32),
        "target_scale": np.maximum(np.nanstd(targets, axis=0), 1e-6).astype(np.float32),
        "task_weights": weights.astype(np.float32),
        "training_label_counts": counts,
    }


class Head(nn.Module):
    def __init__(self, dimension, recipe, statistics=None):
        super().__init__()
        defaults = {
            "feature_mean": np.zeros(dimension),
            "feature_scale": np.ones(dimension),
            "target_mean": np.zeros(8),
            "target_scale": np.ones(8),
            "task_weights": np.ones(8),
        }
        for key, value in defaults.items():
            self.register_buffer(
                key,
                torch.as_tensor(
                    value if statistics is None else statistics[key], dtype=torch.float32
                ),
            )
        self.head = nn.Sequential(
            nn.Linear(dimension, recipe["hidden_dim"]),
            nn.BatchNorm1d(recipe["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(recipe["dropout"]),
            nn.Linear(recipe["hidden_dim"], 8),
        )

    def forward(self, x):
        return (
            self.head((x - self.feature_mean) / self.feature_scale) * self.target_scale
            + self.target_mean
        )


def weighted_loss(model, prediction, target):
    mask = torch.isfinite(target)
    if not mask.any():
        raise ValueError("No observed labels in batch")
    safe = torch.where(mask, target, prediction.detach())
    errors = (prediction - safe).abs() / model.target_scale
    return (errors * mask * model.task_weights).sum() / mask.sum()


def validation_loss(model, x, y):
    model.eval()
    with torch.inference_mode():
        return float(weighted_loss(model, model(x), y))


def fit(features, targets, train, valid, seed, run, output, recipe=None):
    recipe = RECIPE if recipe is None else recipe
    output = Path(output)
    output.mkdir()
    statistics = training_statistics(features[train], targets[train], recipe["auxiliary_weight"])
    torch.manual_seed(seed)
    model = Head(features.shape[1], recipe, statistics).to(run.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"]
    )
    x = torch.as_tensor(features, device=run.device)
    y = torch.as_tensor(targets, device=run.device)
    rng = np.random.default_rng(seed)
    best, best_epoch, state = float("inf"), 0, None
    history = []
    started = time.monotonic()
    for epoch in range(1, recipe["max_epochs"] + 1):
        model.train()
        order = rng.permutation(train)
        total, observed = 0.0, 0
        for indices in batch_indices(order, recipe["batch_size"], merge_singleton=True):
            run.check()
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(model, model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            count = int(torch.isfinite(y[indices]).sum())
            total += float(loss.detach()) * count
            observed += count
        run.check()
        value = validation_loss(model, x[valid], y[valid])
        if not np.isfinite(value):
            raise ValueError("Nonfinite validation loss")
        row = {
            "epoch": epoch,
            "training_loss": total / observed,
            "validation_loss": value,
            "molecule_order_sha256": hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(row)
        if value < best:
            best, best_epoch = value, epoch
            state = {
                key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()
            }
            torch.save(
                {
                    "protocol": PROTOCOL,
                    "dimension": features.shape[1],
                    "recipe": recipe,
                    "tasks": TASKS,
                    "state_dict": state,
                    "epoch": best_epoch,
                },
                output / "best.pt",
            )
        atomic_json(output / "curve.json", history)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"fit": output.name, "best_epoch": best_epoch, **row}), flush=True)
        if epoch - best_epoch >= recipe["patience"]:
            break
    model.load_state_dict(state)
    gpu_prediction = predict_head(model.eval(), features[valid])
    restored, _ = restore(output / "best.pt")
    replay = predict_head(restored, features[valid])
    error = float(np.max(np.abs(gpu_prediction - replay)))
    if error > 1e-4:
        raise ValueError(f"Checkpoint CPU replay differs: {error}")
    return gpu_prediction, {
        "selected_epoch": best_epoch,
        "epochs_run": len(history),
        "validation_loss": best,
        "cpu_replay_max_abs": error,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_label_counts": statistics["training_label_counts"].tolist(),
        "task_weights": statistics["task_weights"].tolist(),
        "checkpoint_sha256": sha256_file(output / "best.pt"),
        "curve_sha256": sha256_file(output / "curve.json"),
    }


def restore(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["protocol"] != PROTOCOL or saved["tasks"] != TASKS:
        raise ValueError("Incompatible Workbench probe checkpoint")
    model = Head(saved["dimension"], saved["recipe"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval(), saved


def scored_predictions(values):
    return pd.DataFrame({ep: values[:, TASKS.index(ep.lower())] for ep in ACTIVITY_ENDPOINTS})


def prepare(config):
    if config.get("protocol") != PROTOCOL or set(config["encoders"]) != set(VIEWS):
        raise ValueError("Expected the fixed three-encoder Workbench comparison")
    if config.get("device") != "cuda":
        raise ValueError("The comparison requires GPU fitting")
    frame, truth, provenance = load_bundle(config["data_dir"])
    features, cache_records = {}, {}
    query = frame.rename(columns={"molecule_name": "Molecule_Name", "smiles": "SMILES"})
    for encoder, view in VIEWS.items():
        item = config["encoders"][encoder]
        values, metadata = load_features(
            {**item, "encoder": encoder, "representation": view}, query
        )
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Expected one finite feature vector per molecule")
        features[encoder] = values
        cache_records[encoder] = {
            "representation": view,
            "dimension": values.shape[1],
            "metadata": metadata,
            "manifest_sha256": sha256_file(Path(item["cache_dir"]) / "manifest.json"),
        }
    design = {
        "protocol": PROTOCOL,
        "recipe": RECIPE,
        "data": provenance,
        "caches": cache_records,
        "implementation_sha256": implementation_hashes(),
        "encoder_trainable": False,
        "calibration": "none",
        "fits": 15,
        "submission_authorized": False,
        "folds": frame[["molecule_name", "fold"]].to_dict("records"),
        "evaluation": "checkpoint-selected OOF development scores; no independent internal test",
    }
    return frame, truth, features, design


def screen(config_path):
    from .representation_budget import maybe_supervise

    config = json.loads(Path(config_path).read_text())
    supervised = maybe_supervise("workbench-probe", config_path)
    if supervised is not None:
        return supervised
    frame, truth, features, design = prepare(config)
    folds = validate_folds(frame, PRESERVED_OVERLAPS)
    targets = frame[TASKS].to_numpy(dtype=np.float32)
    with Run(config) as run:
        atomic_json(run.output / "design.json", design)
        run.record.update(protocol=PROTOCOL, calibration="none", submission_authorized=False)
        run.save()
        predictions = {
            encoder: np.full(targets.shape, np.nan, dtype=np.float32) for encoder in VIEWS
        }
        results = []
        for fold, (train, valid) in enumerate(folds):
            for encoder in VIEWS:
                output = run.output / f"{encoder}-fold-{fold}"
                values, record = fit(
                    features[encoder], targets, train, valid, 42 + fold, run, output
                )
                predictions[encoder][valid] = values
                rows = frame.iloc[valid][["molecule_name", "fold"]].reset_index(drop=True)
                for j, task in enumerate(TASKS):
                    rows[task] = values[:, j]
                rows.to_csv(output / "oof.csv", index=False)
                record.update(
                    encoder=encoder,
                    fold=fold,
                    seed=42 + fold,
                    metrics=score_predictions(
                        truth.iloc[valid].reset_index(drop=True), scored_predictions(values)
                    ),
                    oof_sha256=sha256_file(output / "oof.csv"),
                )
                atomic_json(output / "complete.json", record)
                results.append(record)
                atomic_json(
                    run.output / "progress.json",
                    {"completed_fits": len(results), "total_fits": 15, "last_fit": record},
                )
        summary = {}
        for encoder, values in predictions.items():
            if not np.isfinite(values).all():
                raise ValueError("Incomplete five-fold OOF predictions")
            scored = scored_predictions(values)
            summary[encoder] = score_predictions(truth, scored)
            scored.insert(0, "fold", frame.fold)
            scored.insert(0, "Molecule_Name", frame.molecule_name)
            scored.to_csv(run.output / f"{encoder}-oof.csv", index=False)
        # Detect source/data/cache mutations during fitting before accepting the run.
        _, _, _, final_design = prepare(config)
        if final_design != design:
            raise ValueError("Frozen study inputs or implementation changed during fitting")
        result = {
            "protocol": PROTOCOL,
            "complete": True,
            "fits": results,
            "metrics": summary,
            "calibration": "none",
            "submission_authorized": False,
            "evaluation": design["evaluation"],
        }
        atomic_json(run.output / "results.json", result)
        run.record.update(
            complete=True,
            completed_fits=15,
            metrics=summary,
            results_sha256=sha256_file(run.output / "results.json"),
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "status"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.command == "plan":
        _, _, _, design = prepare(config)
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "fits": 15,
                    "rows": design["data"]["rows"],
                    "recipe": RECIPE,
                    "observed_labels": design["data"]["observed_labels"],
                    "max_gpu_hours": config["max_gpu_hours"],
                },
                indent=2,
            )
        )
    elif args.command == "run":
        result = screen(args.config)
        print(json.dumps(result["metrics"], indent=2))
    else:
        root = Path(config["output_dir"])
        record = (
            root / "results.json" if (root / "results.json").exists() else root / "progress.json"
        )
        print(record.read_text() if record.exists() else "No completed fits yet")


if __name__ == "__main__":
    main()
