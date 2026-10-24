"""Chemprop encoder extraction and masked multi-assay regression screening.

Imports are deliberately optional: the CPU CLI does not require PyTorch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .campaign import write_json
from .download import sha256_file


def validate_official_auxiliary(frame):
    """Reject deferred sources even when a historical flat config supplies paths."""
    from .constants import EMAX_TRAIN_FILE, SINGLE_CONCENTRATION_FILE, TDI_TRAIN_FILE

    allowed = {EMAX_TRAIN_FILE, SINGLE_CONCENTRATION_FILE, TDI_TRAIN_FILE}
    if "source" not in frame or not set(frame.source).issubset(allowed):
        raise ValueError(
            "Only official auxiliary assays are supported; external sources are deferred"
        )


def masked_huber(prediction, target):
    import torch

    observed = torch.isfinite(target)
    safe = torch.where(observed, target, torch.zeros_like(target))
    losses = torch.nn.functional.huber_loss(prediction, safe, reduction="none")
    count = observed.sum(dim=0)
    active = count > 0
    if not active.any():
        raise ValueError("Batch has no observed targets")
    return ((losses * observed).sum(dim=0) / count.clamp_min(1))[active].mean()


def source_balanced_huber(prediction, target, sources):
    """Balance official direct/auxiliary families, then observed regression heads."""
    import torch

    observed = torch.isfinite(target)
    safe = torch.where(observed, target, torch.zeros_like(target))
    losses = torch.nn.functional.huber_loss(prediction, safe, reduction="none")
    counts = observed.sum(dim=0)
    head_loss = (losses * observed).sum(dim=0) / counts.clamp_min(1)
    families = []
    for source in torch.unique(sources):
        selected = (sources == source) & (counts > 0)
        if selected.any():
            families.append(head_loss[selected].mean())
    if not families:
        raise ValueError("Batch has no observed targets")
    return torch.stack(families).mean()


def interval_loss(prediction, target, lower, upper):
    """Distance outside observed confidence bounds plus a small center penalty."""
    import torch

    observed = torch.isfinite(target)
    lower = torch.where(observed, lower, torch.zeros_like(lower))
    upper = torch.where(observed, upper, torch.zeros_like(upper))
    target = torch.where(observed, target, torch.zeros_like(target))
    if not (torch.isfinite(lower).all() and torch.isfinite(upper).all()):
        raise ValueError("Observed interval targets require finite bounds")
    if ((lower > upper) & observed).any():
        raise ValueError("Reversed interval bounds")
    distance = torch.relu(lower - prediction) + torch.relu(prediction - upper)
    center = torch.nn.functional.huber_loss(prediction, target, reduction="none")
    count = observed.sum(dim=0)
    if not (count > 0).any():
        raise ValueError("Batch has no observed targets")
    return (((distance + 0.1 * center) * observed).sum(0) / count.clamp_min(1))[count > 0].mean()


def encoder(checkpoint: Path | None):
    import torch
    from chemprop import nn

    if checkpoint is None:
        return nn.BondMessagePassing()
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model = nn.BondMessagePassing(**saved["hyper_parameters"])
    model.load_state_dict(saved["state_dict"])
    return model


def loader(smiles, targets=None, *, shuffle=False, batch_size=64):
    from chemprop import data

    points = [
        data.MoleculeDatapoint.from_smi(s, y=None if targets is None else targets[i])
        for i, s in enumerate(smiles)
    ]
    dataset = data.MoleculeDataset(points)
    return data.build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=None if shuffle else False,
    )


def encode_smiles(smiles, checkpoint: Path):
    import torch
    from chemprop import nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.25)
    model = encoder(checkpoint).to(device).eval()
    aggregate = nn.MeanAggregation()
    rows = []
    with torch.inference_mode():
        for batch in loader(list(smiles)):
            batch.bmg.to(device)
            rows.append(aggregate(model(batch.bmg), batch.bmg.batch).cpu().numpy())
    return np.concatenate(rows)


def embeddings(prepared_dir: Path, checkpoint: Path, output_dir: Path) -> dict:
    import torch
    from chemprop import nn

    manifest = json.loads((prepared_dir / "manifest.json").read_text())
    path = prepared_dir / "training.csv"
    if sha256_file(path) != manifest["training_sha256"]:
        raise ValueError("Prepared corpus checksum mismatch")
    frame = pd.read_csv(path)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.25)
    start = time.monotonic()
    model = encoder(checkpoint).to(device).eval()
    aggregate = nn.MeanAggregation()
    rows = []
    with torch.inference_mode():
        for batch in loader(frame.SMILES):
            batch.bmg.to(device)
            rows.append(aggregate(model(batch.bmg), batch.bmg.batch).cpu().numpy())
    matrix = np.concatenate(rows)
    if len(matrix) != len(frame) or not np.isfinite(matrix).all():
        raise ValueError("Invalid embedding matrix")
    np.save(output_dir / "embeddings.npy", matrix)
    result = {
        "training_sha256": manifest["training_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "embeddings_sha256": sha256_file(output_dir / "embeddings.npy"),
        "shape": list(matrix.shape),
        "device": device,
        "gpu_hours": (time.monotonic() - start) / 3600 if device == "cuda" else 0,
    }
    write_json(output_dir / "manifest.json", result)
    return result


def screen(
    config_path: Path | None = None,
    *,
    model_config: Path | None = None,
    data_config: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Fixed-epoch outer CV for screening; meta-model selection remains nested."""
    import torch

    from .budget import GPUBudget, budget_ledger_path
    from .configuration import resolve_config, source_hashes
    from .constants import ACTIVITY_ENDPOINTS
    from .data_sources.official import neural_training_frame
    from .metrics import soft_threshold_rae

    config, provenance = resolve_config(
        config_path, model_config=model_config, data_config=data_config, output_dir=output_dir
    )
    allowed = {
        "prepared_dir",
        "output_dir",
        "checkpoint",
        "auxiliary_dir",
        "auxiliary_dirs",
        "epochs",
        "seed",
        "learning_rate",
        "max_gpu_hours",
        "loss",
        "model_family",
        "auxiliary_tasks",
        "task_weights",
        "batch_size",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown config keys: {set(config) - allowed}")
    prepared = Path(config["prepared_dir"])
    manifest = json.loads((prepared / "manifest.json").read_text())
    if sha256_file(prepared / "training.csv") != manifest["training_sha256"]:
        raise ValueError("Prepared data checksum mismatch")
    train = pd.read_csv(prepared / "training.csv")
    checkpoint = Path(config["checkpoint"]) if config.get("checkpoint") else None
    aux = None
    aux_manifest = []
    directories = config.get("auxiliary_dirs", [])
    if config.get("auxiliary_dir"):
        directories = [config["auxiliary_dir"], *directories]
    frames = []
    for directory in directories:
        aux_dir = Path(directory)
        evidence = json.loads((aux_dir / "manifest.json").read_text())
        if sha256_file(aux_dir / "assays.csv") != evidence["assays_sha256"]:
            raise ValueError("Auxiliary checksum mismatch")
        aux_manifest.append(evidence)
        frame = pd.read_csv(aux_dir / "assays.csv")
        validate_official_auxiliary(frame)
        frames.append(frame)
    if frames:
        aux = pd.concat(frames, ignore_index=True)
    if config.get("auxiliary_tasks") is not None:
        from .data_sources.official import select_auxiliary_tasks

        aux = select_auxiliary_tasks(aux, config["auxiliary_tasks"])
    family = config.get("model_family", "legacy")
    if family not in {"legacy", "scigantic_mt_aux8"}:
        raise ValueError("Unknown neural model_family")
    if family == "scigantic_mt_aux8" and checkpoint is not None:
        raise ValueError("The eight-head baseline uses random initialization")
    loss_name = config.get("loss", "huber")
    if loss_name not in {"huber", "interval", "mae", "interval_mae"}:
        raise ValueError("Unknown neural loss")
    if loss_name == "interval" and aux is not None:
        raise ValueError("Interval ablation currently supports official targets only")
    if train.groupby("cluster").fold.nunique().max() != 1:
        raise ValueError("A molecular cluster crosses folds")
    folds = sorted(int(fold) for fold in train.fold.unique())
    folds_to_run = folds
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("Neural screening requires a CUDA GPU")
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(0.25)
    device = "cuda"
    start = time.monotonic()
    budget = float(config.get("max_gpu_hours", 2)) * 3600
    if not 0 < budget <= 72 * 3600:
        raise ValueError("GPU budget must be positive and at most 72 hours")
    epochs = int(config.get("epochs", 30))
    if epochs < 1:
        raise ValueError("epochs must be positive")
    rows = []
    allocation = GPUBudget(budget_ledger_path(), output, budget / 3600)
    allocation.reserve()
    run_record = {
        "config": config,
        "molecule_representation": "standardized_parent_v1",
        "training_aggregation": "median_per_parent",
        "gpu_ledger": str(allocation.path),
        "prepared_manifest": manifest,
        "auxiliary_manifest": aux_manifest,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "gpu_hours": None,
        "source_sha256": source_hashes(),
        "configuration_provenance": provenance,
        "selected_folds": folds_to_run,
        "completed_folds": [],
        "requested_folds_complete": False,
        "complete": False,
    }
    write_json(output / "run_manifest.json", run_record)
    try:
        for fold in folds_to_run:
            torch.manual_seed(config.get("seed", 42) + int(fold))
            fit = train.loc[train.fold != fold]
            val = train.loc[train.fold == fold]
            joined = neural_training_frame(fit, aux, val.parent.tolist())
            tasks = list(joined.columns)
            smiles = joined.index.tolist()
            values = joined.to_numpy(dtype=np.float32)
            means = np.nanmean(values, axis=0)
            scales = np.nanstd(values, axis=0)
            if not np.isfinite(means).all():
                raise ValueError("An assay head has no training labels")
            scales = np.maximum(scales, 1e-6)
            source_ids = [0 if t in ACTIVITY_ENDPOINTS else 1 for t in tasks]
            sources = torch.tensor(source_ids, device=device)
            targets = (values - means) / scales
            if loss_name == "interval":
                lower = (
                    fit.groupby("parent")[[t + "_conf_low" for t in tasks]]
                    .median()
                    .reindex(joined.index)
                    .to_numpy(dtype=np.float32)
                )
                upper = (
                    fit.groupby("parent")[[t + "_conf_high" for t in tasks]]
                    .median()
                    .reindex(joined.index)
                    .to_numpy(dtype=np.float32)
                )
                targets = np.concatenate(
                    [targets, (lower - means) / scales, (upper - means) / scales], axis=1
                )
            if loss_name == "interval_mae":
                lower = (
                    fit.groupby("parent")[[t + "_conf_low" for t in ACTIVITY_ENDPOINTS]]
                    .median()
                    .reindex(joined.index)
                    .to_numpy(dtype=np.float32)
                )
                upper = (
                    fit.groupby("parent")[[t + "_conf_high" for t in ACTIVITY_ENDPOINTS]]
                    .median()
                    .reindex(joined.index)
                    .to_numpy(dtype=np.float32)
                )
                targets = np.concatenate(
                    [targets, (lower - means[:4]) / scales[:4], (upper - means[:4]) / scales[:4]],
                    axis=1,
                )
            training = loader(
                smiles, targets, shuffle=True, batch_size=int(config.get("batch_size", 64))
            )
            validation = loader(val.parent.tolist())
            from .neural_models import build_model, masked_mae

            mp, agg, head = build_model(family, len(tasks), checkpoint)
            mp, agg, head = mp.to(device), agg.to(device), head.to(device)
            task_weights = torch.tensor(
                [config.get("task_weights", {}).get(t, 1.0) for t in tasks], device=device
            )
            optimizer = torch.optim.AdamW(
                [*mp.parameters(), *head.parameters()],
                lr=float(config.get("learning_rate", 1e-4)),
                weight_decay=1e-4,
            )
            for epoch in range(epochs):
                mp.train()
                head.train()
                total = 0.0
                for batch in training:
                    if time.monotonic() - start >= budget:
                        raise TimeoutError("Neural screening exhausted its GPU allocation")
                    batch.bmg.to(device)
                    prediction = head(agg(mp(batch.bmg), batch.bmg.batch))
                    if loss_name == "interval":
                        target, lower, upper = batch.Y.to(device).chunk(3, dim=1)
                        loss = interval_loss(prediction, target, lower, upper)
                    elif loss_name in {"mae", "interval_mae"}:
                        packed = batch.Y.to(device)
                        target = packed[:, : len(tasks)]
                        bounds = (
                            (packed[:, len(tasks) : len(tasks) + 4], packed[:, len(tasks) + 4 :])
                            if loss_name == "interval_mae"
                            else None
                        )
                        loss = masked_mae(prediction, target, task_weights, bounds=bounds)
                    else:
                        loss = source_balanced_huber(prediction, batch.Y.to(device), sources)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite neural training loss")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [*mp.parameters(), *head.parameters()], 5.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    total += loss.item()
                print(
                    json.dumps(
                        {"fold": int(fold), "epoch": epoch + 1, "loss": total / len(training)}
                    ),
                    flush=True,
                )
            mp.eval()
            head.eval()
            predictions = []
            with torch.inference_mode():
                for batch in validation:
                    batch.bmg.to(device)
                    predictions.append(head(agg(mp(batch.bmg), batch.bmg.batch)).cpu().numpy())
            predictions = np.concatenate(predictions) * scales + means
            for column, endpoint in enumerate(ACTIVITY_ENDPOINTS):
                observed = val[endpoint].notna()
                frame = val.loc[observed, ["Molecule_Name", "parent", "cluster", "fold"]].copy()
                frame["endpoint"] = endpoint
                frame["y_true"] = val.loc[observed, endpoint]
                frame["conf_low"] = val.loc[observed, endpoint + "_conf_low"]
                frame["conf_high"] = val.loc[observed, endpoint + "_conf_high"]
                frame["y_pred"] = predictions[observed, column]
                rows.append(frame)
            torch.save(
                {
                    "model_family": family,
                    "encoder_parameters": mp.hparams if family == "legacy" else {},
                    "encoder_state": mp.state_dict(),
                    "head_state": head.state_dict(),
                    "tasks": tasks,
                    "means": means.tolist(),
                    "scales": scales.tolist(),
                },
                output / f"fold-{fold}.pt",
            )
            pd.concat(rows).to_csv(output / "oof_predictions.csv", index=False)
            run_record.setdefault("checkpoint_sha256_by_fold", {})[str(fold)] = sha256_file(
                output / f"fold-{fold}.pt"
            )
            fold_oof = pd.concat(rows).query("fold == @fold")
            run_record.setdefault("fold_scores", {})[str(fold)] = {
                endpoint: soft_threshold_rae(f.y_true, f.y_pred, f.conf_low, f.conf_high)
                for endpoint, f in fold_oof.groupby("endpoint")
            }
            run_record["completed_folds"].append(int(fold))
            run_record["requested_folds_complete"] = run_record["completed_folds"] == folds_to_run
            write_json(output / "run_manifest.json", run_record)
        result = {"selected_folds": folds_to_run, "complete": False}
        if folds_to_run == folds:
            oof = pd.concat(rows)
            scores = {
                endpoint: soft_threshold_rae(f.y_true, f.y_pred, f.conf_low, f.conf_high)
                for endpoint in ACTIVITY_ENDPOINTS
                for f in [oof.loc[oof.endpoint == endpoint]]
            }
            result = {
                "endpoints": scores,
                "MA-ST-RAE": float(np.mean(list(scores.values()))),
                "purpose": "screening; not nested model selection",
                "selected_folds": folds_to_run,
                "complete": True,
            }
            write_json(output / "metrics.json", result)
        return result
    finally:
        used_hours = allocation.finish()
        run_record["gpu_hours"] = used_hours
        run_record["complete"] = (
            (output / "metrics.json").exists()
            and folds_to_run == folds
            and run_record["requested_folds_complete"]
        )
        write_json(output / "run_manifest.json", run_record)
