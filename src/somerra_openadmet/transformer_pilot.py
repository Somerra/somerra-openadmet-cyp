"""Bounded paired R25-member/SMILES pilot: python -m ...transformer_pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit

from .budget import GPUBudget
from .campaign import write_json
from .constants import ACTIVITY_ENDPOINTS
from .download import sha256_file
from .metrics import soft_threshold_rae
from .smiles_transformer import (
    TOKENIZER,
    ReferenceMember,
    build_model,
    count_parameters,
    load_checkpoint,
    masked_mae,
    matched_architecture,
    tokenize,
)

TASKS = list(ACTIVITY_ENDPOINTS) + [
    f"{ep.split('_')[0]}_log2fc_4.9504951e-05M" for ep in ACTIVITY_ENDPOINTS
]
WEIGHTS = [1.0961034297943115, 1.2044342756271362, 1.0366363525390625, 0.6628257036209106] + [
    0.29999998211860657
] * 4


def prepare(config):
    prepared, auxiliary = Path(config["prepared_dir"]), Path(config["auxiliary_dir"])
    hashes = {}
    for directory, name, key in [
        (prepared, "training.csv", "training_sha256"),
        (auxiliary, "assays.csv", "assays_sha256"),
    ]:
        actual = sha256_file(directory / name)
        if actual != json.loads((directory / "manifest.json").read_text())[key]:
            raise ValueError("Input checksum mismatch")
        hashes[str(directory / name)] = actual
    frame = pd.read_csv(prepared / "training.csv")
    aux = pd.read_csv(auxiliary / "assays.csv")
    wide = aux.loc[aux.task.isin(TASKS[4:])].pivot_table(
        index="parent", columns="task", values="value", aggfunc="median"
    )
    frame = frame.join(wide[TASKS[4:]], on="parent")
    if frame.groupby("cluster").fold.nunique().max() != 1:
        raise ValueError("Cluster crosses folds")
    if frame.groupby("parent").fold.nunique().max() != 1:
        raise ValueError("Parent crosses folds")
    molecules = [Chem.MolFromSmiles(s) for s in frame.SMILES]
    if any(m is None for m in molecules):
        raise ValueError("Invalid molecule")
    frame["canonical_smiles"] = [Chem.MolToSmiles(m, isomericSmiles=True) for m in molecules]
    outer = np.flatnonzero(frame.fold.to_numpy() == 0)
    remaining = np.flatnonzero(frame.fold.to_numpy() != 0)
    fit, val = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            remaining, groups=frame.iloc[remaining].cluster
        )
    )
    fit, val = remaining[fit], remaining[val]
    for key in ["cluster", "parent"]:
        groups = [set(frame.iloc[ix][key]) for ix in [fit, val, outer]]
        if any(groups[i] & groups[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
            raise ValueError("Split overlap")
    values = frame[TASKS].to_numpy(dtype=np.float32)
    means, scales = np.nanmean(values[fit], axis=0), np.nanstd(values[fit], axis=0)
    if not np.isfinite(means).all():
        raise ValueError("Missing training head")
    scales = np.maximum(scales, 1e-6)
    return frame, (values - means) / scales, means, scales, (fit, val, outer), hashes


class Inputs:
    def __init__(self, smiles):
        from chemprop import data

        self.smiles = list(smiles)
        tokenize(self.smiles)  # Validate vocabulary; no learned or held-out vocabulary.
        self.graphs = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in smiles])
        self.graphs.cache = True

    def batch(self, indices, family, device):
        if family == "smiles_transformer":
            return tokenize([self.smiles[i] for i in indices]).to(device)
        from chemprop.data import collate_batch

        graph = collate_batch([self.graphs[int(i)] for i in indices]).bmg
        graph.to(device)
        return graph


def batches(indices, size=64):
    # Avoid a singleton BatchNorm training batch without losing observations.
    return np.array_split(indices, max(1, math.ceil(len(indices) / size)))


def predict(model, inputs, indices, family, device="cpu"):
    model.eval()
    with torch.inference_mode():
        return np.concatenate(
            [
                model(inputs.batch(ix, family, device)).float().cpu().numpy()
                for ix in batches(indices)
            ]
        )


def scores(frame, indices, prediction):
    result = {}
    for j, ep in enumerate(ACTIVITY_ENDPOINTS):
        rows = frame.iloc[indices]
        valid = rows[ep].notna() & rows[ep + "_conf_low"].notna() & rows[ep + "_conf_high"].notna()
        result[ep] = float(
            soft_threshold_rae(
                rows.loc[valid, ep],
                prediction[valid, j],
                rows.loc[valid, ep + "_conf_low"],
                rows.loc[valid, ep + "_conf_high"],
            )
        )
    return {"endpoints": result, "MA-ST-RAE": float(np.mean(list(result.values())))}


def train_one(
    architecture,
    inputs,
    targets,
    means,
    scales,
    splits,
    frame,
    output,
    deadline,
    training=None,
):
    from chemprop.schedulers import build_NoamLike_LRSched

    settings = {
        "seed": 42,
        "batch_size": 64,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "peak_lr": 1e-3,
        "schedule": "noam",
        "max_epochs": 400,
        "patience": 40,
        "min_epochs": 0,
        "precision": "float16",
    }
    if training and set(training) - set(settings):
        raise ValueError("Unknown training settings")
    settings.update(training or {})
    if settings["batch_size"] < 2 or settings["max_epochs"] < 3:
        raise ValueError("Invalid training dimensions")
    if settings["optimizer"] not in {"adam", "adamw"}:
        raise ValueError("Unknown optimizer")
    if settings["schedule"] not in {"noam", "cosine"}:
        raise ValueError("Unknown schedule")
    if settings["precision"] not in {"float16", "bfloat16"}:
        raise ValueError("Unknown precision")
    torch.manual_seed(settings["seed"])
    family = architecture["family"]
    fit, val, _ = splits
    model = build_model(architecture).cuda()
    optimizer_class = torch.optim.AdamW if settings["optimizer"] == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=1e-4, weight_decay=settings["weight_decay"])
    per_epoch = len(batches(fit, settings["batch_size"]))
    if settings["schedule"] == "noam":
        scheduler = build_NoamLike_LRSched(
            optimizer,
            2 * per_epoch,
            (settings["max_epochs"] - 2) * per_epoch,
            1e-4,
            settings["peak_lr"],
            1e-4,
        )
    else:
        warmup = 5 * per_epoch
        total = settings["max_epochs"] * per_epoch

        def multiplier(step):
            if step < warmup:
                return 0.1 + 0.9 * step / warmup
            progress = min(1, (step - warmup) / max(1, total - warmup))
            return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2

        for group in optimizer.param_groups:
            group["lr"] = settings["peak_lr"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    dtype = getattr(torch, settings["precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    weights = torch.tensor(WEIGHTS, device="cuda")
    truth = torch.tensor(targets, device="cuda")
    rng = np.random.default_rng(settings["seed"])
    start, steps, examples, best, stale = time.monotonic(), 0, 0, float("inf"), 0
    curves, best_epoch = [], None
    overflow_steps = 0
    torch.cuda.reset_peak_memory_stats()
    reason = "max_epochs"
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        losses = []
        for ix in batches(rng.permutation(fit), settings["batch_size"]):
            if time.monotonic() >= deadline - 15:
                reason = "budget_limited"
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                batch = (
                    inputs.training_batch(ix, family, "cuda", epoch)
                    if hasattr(inputs, "training_batch")
                    else inputs.batch(ix, family, "cuda")
                )
                pred = model(batch)
                loss = masked_mae(pred, truth[ix], weights)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite = all(
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            )
            if not finite and not scaler.is_enabled():
                raise ValueError("Nonfinite unscaled gradients")
            scaler.step(optimizer)  # GradScaler skips updates on overflow.
            scaler.update()
            if finite:
                scheduler.step()
                steps += 1
            else:
                overflow_steps += 1
                if scaler.get_scale() < 1:
                    raise ValueError("Persistent nonfinite gradients at unit loss scale")
            examples += len(ix)
            losses.append(float(loss.detach()))
        if reason == "budget_limited":
            break
        pred = predict(model, inputs, val, family, "cuda")
        value = float(
            masked_mae(torch.tensor(pred), torch.tensor(targets[val]), torch.tensor(WEIGHTS))
        )
        torch.cuda.synchronize()
        entry = {
            "epoch": epoch,
            "steps": steps,
            "overflow_steps": overflow_steps,
            "seconds": time.monotonic() - start,
            "train_loss": float(np.mean(losses)),
            "validation_loss": value,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **scores(frame, val, pred * scales + means),
        }
        curves.append(entry)
        write_json(output / f"{family}-curves.json", curves)
        if value < best:
            best, best_epoch, stale = value, epoch, 0
            torch.save(
                {
                    "architecture": architecture,
                    "training": settings,
                    "tokenizer": TOKENIZER,
                    "tasks": TASKS,
                    "means": means.tolist(),
                    "scales": scales.tolist(),
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "steps": steps,
                },
                output / f"{family}.pt",
            )
        else:
            stale += 1
        print(json.dumps({"family": family, **entry}), flush=True)
        if stale >= settings["patience"] and epoch >= settings["min_epochs"]:
            reason = "early_stopping"
            break
    elapsed = time.monotonic() - start
    result = {
        "status": reason,
        "training": settings,
        "overflow_steps": overflow_steps,
        "best_epoch": best_epoch,
        "steps": steps,
        "seconds": elapsed,
        "examples_per_second": examples / elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "parameters": count_parameters(model),
        "completed_epochs": len(curves),
    }
    del model, optimizer, truth, weights, scaler, scheduler
    torch.cuda.empty_cache()
    if best_epoch is None:
        raise RuntimeError("Budget ended before the first validation checkpoint")
    return result


def run(config_path):
    config = json.loads(config_path.read_text())
    if set(config) - {"prepared_dir", "auxiliary_dir", "output_dir", "ledger", "prior_attempts"}:
        raise ValueError("Unexpected config keys")
    ledger_path = Path(os.environ.get("SOMERRA_GPU_LEDGER", config["ledger"]))
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    prior_paths = [str(Path(p).resolve()) for p in config.get("prior_attempts", [])]
    if len(set(prior_paths)) != len(prior_paths):
        raise ValueError("Duplicate prior attempts")
    prior_hours = 0.0
    for prior in prior_paths:
        evidence = json.loads((Path(prior) / "run_manifest.json").read_text())
        charged = ledger[prior]["used_hours"]
        if charged != evidence["gpu_hours"] or evidence.get("complete"):
            raise ValueError("Invalid recovery accounting")
        prior_hours += charged
    remaining_hours = 1.0 - prior_hours
    if remaining_hours <= 300 / 3600:
        raise ValueError("Insufficient remaining pilot allocation")
    frame, targets, means, scales, splits, hashes = prepare(config)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    reference_count = count_parameters(ReferenceMember())
    architectures = [{"family": "mt_aux8", "n_tasks": 8}, matched_architecture(reference_count)]
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    split_frame = frame[["Molecule_Name", "parent", "cluster", "fold"]].copy()
    for name, ix in zip(["fit", "inner_validation", "outer_test"], splits, strict=True):
        split_frame.loc[ix, "partition"] = name
    split_frame.to_csv(output / "splits.csv", index=False)
    record = {
        "config": config,
        "architectures": architectures,
        "input_hashes": hashes,
        "source_hashes": {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
        "split_sha256": sha256_file(output / "splits.csv"),
        "tasks": TASKS,
        "task_weights": WEIGHTS,
        "seed": 42,
        "outer_fold": 0,
        "reference_commit": "e70a2920d611684c9aaff07400d951e2e79dd5f6",
        "purpose": "adapted local one-fold pilot, not R25 reproduction or promotion",
        "open_source_code": False,
        "model_report_link": "",
        "complete": False,
        "versions": {"torch": torch.__version__},
        "results": {},
        "prior_gpu_hours": prior_hours,
        "reserved_hours": remaining_hours,
    }
    write_json(output / "run_manifest.json", record)
    inputs = Inputs(frame.canonical_smiles.tolist())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_per_process_memory_fraction(0.08)
    allocation = GPUBudget(ledger_path, output, remaining_hours)
    allocation.reserve()
    global_deadline = time.monotonic() + remaining_hours * 3600
    try:
        smoke_start = time.monotonic()
        for architecture in architectures:
            model = build_model(architecture).cuda()
            ix = splits[0][:8]
            pred = model(inputs.batch(ix, architecture["family"], "cuda"))
            loss = masked_mae(
                pred, torch.tensor(targets[ix], device="cuda"), torch.tensor(WEIGHTS, device="cuda")
            )
            loss.backward()
            if not torch.isfinite(loss) or not all(
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            ):
                raise ValueError("GPU smoke failed")
            del model, pred, loss
            torch.cuda.empty_cache()
        record["smoke_seconds"] = time.monotonic() - smoke_start
        if record["smoke_seconds"] > 300:
            raise TimeoutError("Smoke allocation exceeded")
        for architecture in architectures:
            deadline = min(
                global_deadline - 20, time.monotonic() + (remaining_hours * 3600 - 300) / 2
            )
            record["results"][architecture["family"]] = train_one(
                architecture, inputs, targets, means, scales, splits, frame, output, deadline
            )
            write_json(output / "run_manifest.json", record)
    except BaseException as exc:
        record["error"] = repr(exc)
        raise
    finally:
        record["gpu_hours"] = allocation.finish()
        write_json(output / "run_manifest.json", record)
    # CPU-only final evaluation: outer labels never select checkpoints or architecture.
    predictions = []
    for architecture in architectures:
        family = architecture["family"]
        model, saved = load_checkpoint(output / f"{family}.pt")
        outer = splits[2]
        pred = predict(model, inputs, outer, family) * scales + means
        record["results"][family]["outer"] = scores(frame, outer, pred)
        for j, ep in enumerate(ACTIVITY_ENDPOINTS):
            rows = frame.iloc[outer][["Molecule_Name", "parent", "cluster", "fold"]].copy()
            rows["endpoint"], rows["family"] = ep, family
            rows["y_true"], rows["y_pred"] = frame.iloc[outer][ep].to_numpy(), pred[:, j]
            rows["conf_low"] = frame.iloc[outer][ep + "_conf_low"].to_numpy()
            rows["conf_high"] = frame.iloc[outer][ep + "_conf_high"].to_numpy()
            predictions.append(rows.loc[rows.y_true.notna()])
    pd.concat(predictions).to_csv(output / "outer_predictions.csv", index=False)
    record["complete"] = True
    record["outer_macro_delta"] = (
        record["results"]["smiles_transformer"]["outer"]["MA-ST-RAE"]
        - record["results"]["mt_aux8"]["outer"]["MA-ST-RAE"]
    )
    write_json(output / "run_manifest.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))
