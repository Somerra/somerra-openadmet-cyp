"""One common MLP learner for frozen molecular representations.

No encoder is instantiated during fitting. Outer labels are used only after the
fixed training horizon; standardization statistics come exclusively from fit rows.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .constants import ACTIVITY_ENDPOINTS
from .download import sha256_file
from .metrics import soft_threshold_rae
from .representation_training import masked_mae, oof_rows
from .representations import (
    Run,
    atomic_json,
    batch_indices,
    load_cache,
    prepared_data,
    read_config,
)

PROTOCOL = "matched_frozen_v1"
# Historical recipe: retain it for artifact replay, not as a validated probe of
# representation quality. See docs/atom-jepa-results-20261029.md for the outstanding
# regularized, inner-selected head experiment; it needs a new protocol version.
RECIPE = {
    "epochs": 300,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "Adam",
    "precision": "float32",
    "head": "D->512->BatchNorm->ReLU->4",
    "checkpoint_selection": "final_epoch",
    "loss": "endpoint-balanced masked MAE in raw pIC50",
}


class FrozenMLP(nn.Module):
    def __init__(self, dimension, mean=None, scale=None):
        super().__init__()
        self.register_buffer(
            "mean", torch.zeros(dimension) if mean is None else torch.as_tensor(mean).float()
        )
        self.register_buffer(
            "scale", torch.ones(dimension) if scale is None else torch.as_tensor(scale).float()
        )
        self.head = nn.Sequential(
            nn.Linear(dimension, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, len(ACTIVITY_ENDPOINTS)),
        )

    def forward(self, features):
        return self.head((features - self.mean) / self.scale)


def make_head(features, fit_indices, scaling, seed):
    from sklearn.preprocessing import StandardScaler

    mean = scale = None
    if scaling == "standard":
        scaler = StandardScaler().fit(features[fit_indices])
        mean, scale = scaler.mean_, scaler.scale_
    elif scaling != "none":
        raise ValueError("scaling must be none or standard")
    torch.manual_seed(seed)
    return FrozenMLP(features.shape[1], mean, scale)


def load_features(config, frame):
    encoder = config["encoder"]
    if encoder == "chemeleon":
        if config["representation"] != "mean":
            raise ValueError("CheMeleon requires the mean representation")
        return load_cache(Path(config["cache_dir"]), frame)
    if encoder == "mol_jepa":
        from .mol_jepa import MODEL_REVISION, load_view

        if config["representation"] != "cls":
            raise ValueError("Matched Mol-JEPA benchmark requires CLS, not a token head")
        pin = json.loads((Path(config["checkpoint"]) / "provenance.json").read_text())
        if pin.get("permitted_use") != "user-confirmed noncommercial research":
            raise ValueError("Missing Mol-JEPA research-use evidence")
        return load_view(
            Path(config["cache_dir"]),
            frame,
            "cls",
            {
                "model_revision": MODEL_REVISION,
                "checkpoint_sha256": pin["sha256"]["model.safetensors"],
            },
        )
    if encoder == "atom_jepa":
        from .atom_jepa import load_view

        return load_view(config["cache_dir"], frame, config["representation"])
    raise ValueError(f"Unsupported frozen encoder: {encoder}")


def restore_head(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        saved.get("schema_version") != 3
        or saved.get("protocol") != PROTOCOL
        or saved.get("tasks") != list(ACTIVITY_ENDPOINTS)
    ):
        raise ValueError("Incompatible frozen-head checkpoint")
    model = FrozenMLP(saved["dimension"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval(), saved


def predict_head(model, features, batch_size=256):
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        result = [
            model(torch.as_tensor(features[i : i + batch_size], device=device)).cpu().numpy()
            for i in range(0, len(features), batch_size)
        ]
    values = np.concatenate(result)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite predictions")
    return values


def scores(rows):
    endpoints = {}
    for endpoint in ACTIVITY_ENDPOINTS:
        part = rows.loc[rows.endpoint == endpoint]
        if part.empty:
            raise ValueError(f"No observed held-out labels: {endpoint}")
        endpoints[endpoint] = {
            "ST-RAE": soft_threshold_rae(part.y_true, part.y_pred, part.conf_low, part.conf_high),
            "MAE": float(np.abs(part.y_true - part.y_pred).mean()),
            "n": len(part),
        }
    return {
        "endpoints": endpoints,
        "MA-ST-RAE": float(np.mean([v["ST-RAE"] for v in endpoints.values()])),
        "macro_MAE": float(np.mean([v["MAE"] for v in endpoints.values()])),
    }


def fit_head(features, targets, fit, seed, scaling, run, epochs=300):
    """Epoch override is for unit tests only; public pilot recipe is fixed."""
    if features.ndim not in (2, 3) or not np.isfinite(features).all():
        raise ValueError("Expected finite [molecule, feature] or [molecule, conformer, feature]")
    multi = features.ndim == 3
    if multi and features.shape[1] < 1:
        raise ValueError("No conformers")
    # Keep the scaler fixed to the original conformer across augmentation arms.
    reference = features[:, 0] if multi else features
    model = make_head(reference, fit, scaling, seed).to(run.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.as_tensor(features, device=run.device)
    y = torch.as_tensor(targets, device=run.device)
    rng = np.random.default_rng(seed)
    conformer_rng = np.random.default_rng(seed + 100_000)
    conformer_ordering = hashlib.sha256()
    ordering = hashlib.sha256()
    history = []
    updates = 0
    started = time.monotonic()
    for epoch in range(epochs):
        model.train()
        losses = []
        for indices in batch_indices(rng.permutation(fit), 256, merge_singleton=True):
            run.check()
            ordering.update(indices.astype("<i8").tobytes())
            optimizer.zero_grad(set_to_none=True)
            if multi:
                choices = conformer_rng.integers(features.shape[1], size=len(indices))
                conformer_ordering.update(choices.astype("<i8").tobytes())
                inputs = x[indices, choices]
            else:
                inputs = x[indices]
            loss = masked_mae(model(inputs), y[indices])
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError("Non-finite MLP loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Missing or non-finite MLP gradient")
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
        if not losses:
            raise ValueError("No observed training labels")
        run.sync()
        history.append(
            {
                "epoch": epoch + 1,
                "training_mae": float(np.mean(losses)),
                "elapsed_seconds": time.monotonic() - started,
                "updates": updates,
            }
        )
        if epoch == 0:
            run.check(history[0]["elapsed_seconds"] * (epochs - 1) * 1.2)
        if epoch % 50 == 0 or epoch == epochs - 1:
            atomic_json(run.output / "training.json", history)
    return model, {
        "successful_updates": updates,
        "batch_order_sha256": ordering.hexdigest(),
        "train_conformers": features.shape[1] if multi else 1,
        "conformer_order_sha256": conformer_ordering.hexdigest() if multi else None,
        "training_seconds": time.monotonic() - started,
        "head_parameters": sum(p.numel() for p in model.parameters()),
        "history": history,
    }


def screen(config_path):
    config = read_config(
        Path(config_path),
        {
            "protocol",
            "encoder",
            "representation",
            "prepared_dir",
            "cache_dir",
            "checkpoint",
            "scaling",
            "selected_folds",
            "adaptation",
            "ledger",
        },
        {
            "protocol",
            "encoder",
            "representation",
            "prepared_dir",
            "cache_dir",
            "output_dir",
            "max_gpu_hours",
            "scaling",
            "selected_folds",
            "seed",
        },
    )
    if config["protocol"] != PROTOCOL or config.get("adaptation", "frozen") != "frozen":
        raise ValueError("This protocol only fits frozen representations")
    if config.get("precision", "float32") != "float32":
        raise ValueError("Matched MLP training requires float32")
    if config["selected_folds"] != [0]:
        raise ValueError("This pilot is predeclared for outer fold 0 only")
    from .representation_budget import maybe_supervise

    supervised = maybe_supervise("frozen-pilot", config_path)
    if supervised is not None:
        return supervised
    frame, prepared = prepared_data(Path(config["prepared_dir"]))
    if prepared["n_folds"] != 5:
        raise ValueError("Expected shared five-fold manifest")
    features, cache = load_features(config, frame)
    targets = frame[list(ACTIVITY_ENDPOINTS)].to_numpy(dtype=np.float32)
    fit = np.flatnonzero(frame.fold.to_numpy() != 0)
    held = np.flatnonzero(frame.fold.to_numpy() == 0)
    if (np.isfinite(targets[fit]).sum(0) == 0).any():
        raise ValueError("An endpoint has no training labels")
    with Run(config) as run:
        run.record.update(
            protocol=PROTOCOL,
            recipe=RECIPE,
            prepared_manifest=prepared,
            cache_manifest=cache,
            completed_folds=[],
            selected_folds=[0],
            requested_folds_complete=False,
            promoted=False,
            purpose="single_fold_representation_pilot",
            encoder_trainable=False,
        )
        run.save()
        model, training = fit_head(features, targets, fit, config["seed"], config["scaling"], run)
        prediction = predict_head(model, features[held])
        saved = {
            "schema_version": 3,
            "protocol": PROTOCOL,
            "tasks": list(ACTIVITY_ENDPOINTS),
            "dimension": features.shape[1],
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": config,
            "recipe": RECIPE,
            "fold": 0,
            "seed": config["seed"],
            "training_sha256": prepared["training_sha256"],
            "cache_manifest_sha256": sha256_file(Path(config["cache_dir"]) / "manifest.json"),
            "fit_ids": frame.iloc[fit].Molecule_Name.tolist(),
            "held_ids": frame.iloc[held].Molecule_Name.tolist(),
        }
        checkpoint = run.output / "fold-0.pt"
        torch.save(saved, checkpoint)
        restored, _ = restore_head(checkpoint)
        replay = predict_head(restored, features[held])
        delta = float(np.abs(prediction - replay).max())
        if delta > 1e-4:
            raise ValueError(f"CPU replay exceeds 1e-4 pIC50: {delta}")
        rows = oof_rows(frame, held, prediction)
        rows.to_csv(run.output / "oof_predictions.csv", index=False)
        metrics = scores(rows)
        atomic_json(run.output / "pilot_metrics.json", metrics)
        atomic_json(
            run.output / "split.json", {"fit_ids": saved["fit_ids"], "held_ids": saved["held_ids"]}
        )
        run.record.update(
            training=training,
            metrics=metrics,
            completed_folds=[0],
            requested_folds_complete=True,
            complete=False,
            pilot_complete=True,
            cpu_replay_max_abs=delta,
            input_dimension=features.shape[1],
            checkpoint_sha256=sha256_file(checkpoint),
            oof_sha256=sha256_file(run.output / "oof_predictions.csv"),
        )
        return run.record


def paired_seed_bootstrap(predictions, candidate, reference, samples=1000, seed=42):
    """CI for mean per-seed metric difference, not a seed-ensemble score.

    Molecular clusters are resampled jointly across all targets and paired seeds.
    """
    keys = ["Molecule_Name", "endpoint", "cluster", "fold", "y_true", "conf_low", "conf_high"]
    arrays = {}
    reference_keys = None
    for name in (candidate, reference):
        parts = []
        for model_seed in (42, 43, 44):
            part = predictions[(predictions.arm == name) & (predictions.seed == model_seed)]
            part = part.sort_values(["Molecule_Name", "endpoint"]).reset_index(drop=True)
            if part.duplicated(["Molecule_Name", "endpoint"]).any():
                raise ValueError("Duplicate predictions")
            if reference_keys is None:
                reference_keys = part[keys]
            elif not part[keys].equals(reference_keys):
                raise ValueError("Comparison rows, labels, or folds do not match")
            parts.append(
                np.maximum(part.y_pred - part.conf_high, 0).to_numpy()
                + np.maximum(part.conf_low - part.y_pred, 0).to_numpy()
            )
        arrays[name] = np.mean(parts, axis=0)
    data = reference_keys
    groups, group_index = np.unique(data.cluster, return_inverse=True)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))[group_index]
        differences = []
        for endpoint in ACTIVITY_ENDPOINTS:
            keep = data.endpoint.to_numpy() == endpoint
            part, weight = data.loc[keep], weights[keep]
            if weight.sum() == 0:
                break
            average = np.average(part.y_true, weights=weight)
            denominator = np.sum(
                weight
                * (np.maximum(average - part.conf_high, 0) + np.maximum(part.conf_low - average, 0))
            )
            if denominator <= 0:
                break
            differences.append(
                float(
                    np.sum(weight * (arrays[candidate][keep] - arrays[reference][keep]))
                    / denominator
                )
            )
        if len(differences) == 4:
            deltas.append(float(np.mean(differences)))
    if len(deltas) < samples * 0.95:
        raise ValueError("Too many undefined bootstrap replicates")
    return {
        "candidate": candidate,
        "reference": reference,
        "samples": len(deltas),
        "delta_95ci": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "interpretation": "exploratory fold-0 cluster bootstrap of mean per-seed score difference",
    }
