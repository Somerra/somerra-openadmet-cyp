"""Private, source-data reproduction of the historical four-member briford recipe."""

import argparse
import fcntl
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .constants import ACTIVITY_ENDPOINTS
from .data_sources.workbench import MEMBER_TASKS, prepare
from .download import sha256_file
from .reference_member import score_predictions
from .validation import validate_submission

MEMBERS = list(MEMBER_TASKS)
MOMENTS = {
    "cyp1a2": (4.412, 1.553),
    "cyp2c9": (4.830, 1.101),
    "cyp2d6": (3.107, 1.599),
    "cyp3a4": (4.880, 1.272),
}
OOF_FACTORS = {"cyp1a2": 1.32, "cyp2c9": 1.23, "cyp2d6": 1.66, "cyp3a4": 1.07}
RECIPE = {
    "n_folds": 5,
    "max_epochs": 400,
    "patience": 40,
    "batch_size": 64,
    "hidden_dim": 700,
    "depth": 5,
    "dropout": 0.1,
    "ffn_hidden_dim": 300,
    "ffn_num_layers": 2,
    "warmup_epochs": 2,
    "init_lr": 1e-4,
    "max_lr": 1e-3,
    "final_lr": 1e-4,
    "seed": 42,
    "precision": "16-mixed",
    "loss": "mae",
    "aggregation": "NormAggregation(100)",
    "batch_norm": True,
}


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def code_hashes():
    root = Path(__file__).parent
    return {
        name: sha256_file(root / name)
        for name in [
            "briford.py",
            "data_sources/workbench.py",
            "metrics.py",
            "validation.py",
            "reference_member.py",
        ]
    }


def verify(root):
    root = Path(root)
    manifest = read(root / "data/manifest.json")
    if manifest["status"] != "prepared":
        raise ValueError("Data preparation incomplete")
    for name, digest in manifest["file_sha256"].items():
        if sha256_file(root / "data" / name) != digest:
            raise ValueError(f"Changed source data: {name}")
    return manifest


def check_contract(root):
    manifest = verify(root)
    contract = read(Path(root) / "contract.json")
    if (
        contract["code_sha256"] != code_hashes()
        or contract["recipe"] != RECIPE
        or contract["data_manifest_sha256"] != sha256_file(Path(root) / "data/manifest.json")
    ):
        raise ValueError("Frozen reproduction contract changed")
    return manifest


def safe_batch_size(n, batch=64):
    return batch + 1 if n % batch == 1 else batch


def make_model(scaler, weights):
    import torch
    from chemprop import models, nn

    mp = nn.BondMessagePassing(d_h=700, depth=5, dropout=0.1)
    ffn = nn.RegressionFFN(
        input_dim=700,
        hidden_dim=300,
        n_layers=2,
        dropout=0.1,
        n_tasks=len(weights),
        criterion=nn.metrics.MAE(task_weights=torch.tensor(weights)),
        output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
    )
    return models.MPNN(
        mp,
        nn.NormAggregation(),
        ffn,
        batch_norm=True,
        metrics=None,
        warmup_epochs=2,
        init_lr=1e-4,
        max_lr=1e-3,
        final_lr=1e-4,
    )


def predict_frame(model, smiles):
    import torch
    from chemprop import data

    ds = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in smiles])
    ds.cache = True
    loader = data.build_dataloader(
        ds, batch_size=safe_batch_size(len(ds)), shuffle=False, num_workers=0
    )
    model.eval()
    result = []
    with torch.inference_mode():
        for batch in loader:
            batch.bmg.to(model.device)
            result.append(model(batch.bmg).detach().cpu().numpy())
    values = np.concatenate(result)
    if len(values) != len(smiles) or not np.isfinite(values).all():
        raise ValueError("Invalid prediction shape or values")
    return values


def worker(root, member, fold, seconds):
    import lightning as pl
    import torch
    from chemprop import data
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    root = Path(root)
    manifest = check_contract(root)
    if not torch.cuda.is_available():
        raise RuntimeError("This reproduction requires CUDA; no CPU fallback")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("medium")
    spec = manifest["members"][member]
    tasks = spec["tasks"]
    frame = pd.read_csv(root / "data" / f"{member}.csv", keep_default_na=True)
    training = frame.loc[frame.fold != fold].copy()
    validation = frame.loc[frame.fold == fold].copy()
    out = root / "fits" / member / str(fold)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        raise FileExistsError("Fold already completed")
    train_ds = data.MoleculeDataset(
        [
            data.MoleculeDatapoint.from_smi(s, y)
            for s, y in zip(training.smiles, training[tasks].to_numpy(dtype=float), strict=True)
        ]
    )
    val_ds = data.MoleculeDataset(
        [
            data.MoleculeDatapoint.from_smi(s, y)
            for s, y in zip(validation.smiles, validation[tasks].to_numpy(dtype=float), strict=True)
        ]
    )
    scaler = train_ds.normalize_targets()
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError("Empty training head")
    val_ds.normalize_targets(scaler)
    train_ds.cache = True
    val_ds.cache = True
    loaders = {
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }
    train_loader = data.build_dataloader(train_ds, batch_size=64, shuffle=True, **loaders)
    valid_loader = data.build_dataloader(
        val_ds, batch_size=safe_batch_size(len(val_ds)), shuffle=False, **loaders
    )
    pl.seed_everything(42 + fold)
    model = make_model(scaler, spec["task_weights"])
    checkpoint = ModelCheckpoint(
        dirpath=out,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        enable_version_counter=False,
    )
    early = EarlyStopping(monitor="val_loss", patience=40, mode="min")
    started = time.monotonic()

    class Curve(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            loss = float(trainer.callback_metrics["val_loss"].cpu())
            if not np.isfinite(loss):
                raise ValueError("Nonfinite validation loss")
            row = {
                "epoch": trainer.current_epoch + 1,
                "val_loss": loss,
                "elapsed_seconds": time.monotonic() - started,
            }
            with (out / "curve.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            write(out / "progress.json", row)
            if row["epoch"] % 10 == 0:
                print(json.dumps({"member": member, "fold": fold, **row}), flush=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=400,
        precision="16-mixed",
        logger=False,
        enable_progress_bar=False,
        callbacks=[early, checkpoint, Curve()],
        max_time={"seconds": max(1, int(seconds - 60))},
    )
    resume = out / "last.ckpt"
    trainer.fit(
        model, train_loader, valid_loader, ckpt_path=str(resume) if resume.exists() else None
    )
    if trainer.current_epoch < 400 and not early.stopped_epoch:
        raise RuntimeError("Allocation ended before the fold completed its training recipe")
    state = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.to("cuda")
    val_predictions = predict_frame(model, validation.smiles)
    blind = pd.read_csv(root / "data/raw/blind.csv")
    blind_predictions = predict_frame(model, blind.smiles)
    reloaded = make_model(scaler, spec["task_weights"])
    reloaded.load_state_dict(state["state_dict"])
    reloaded.to("cuda")
    parity = predict_frame(reloaded, blind.smiles.iloc[:64])
    # Compare the same batching to avoid batch-size-dependent numerical drift.
    original = predict_frame(model, blind.smiles.iloc[:64])
    if not np.allclose(parity, original, rtol=1e-6, atol=1e-6):
        raise ValueError("Checkpoint reload parity failed")
    pd.DataFrame(val_predictions, columns=tasks).assign(
        molecule_name=validation.molecule_name.to_numpy()
    ).to_csv(out / "oof.csv", index=False)
    pd.DataFrame(blind_predictions, columns=tasks).assign(
        molecule_name=blind.molecule_name.to_numpy()
    ).to_csv(out / "blind.csv", index=False)
    write(out / "scaler.json", {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()})
    write(
        out / "complete.json",
        {
            "member": member,
            "fold": fold,
            "contract_sha256": sha256_file(root / "contract.json"),
            "best_epoch": state["epoch"] + 1,
            "epochs": trainer.current_epoch,
            "best_val_loss": float(checkpoint.best_model_score.cpu()),
            "train_rows": len(training),
            "validation_rows": len(validation),
            "file_sha256": {
                p.name: sha256_file(p)
                for p in [
                    out / "best.ckpt",
                    out / "scaler.json",
                    out / "oof.csv",
                    out / "blind.csv",
                ]
            },
        },
    )


def verify_complete(root, member, fold):
    path = Path(root) / "fits" / member / str(fold)
    record = read(path / "complete.json")
    if record["member"] != member or record["fold"] != fold:
        raise ValueError("Completed fold identity mismatch")
    if record["contract_sha256"] != sha256_file(Path(root) / "contract.json"):
        raise ValueError("Completed fold belongs to another contract")
    for name, digest in record["file_sha256"].items():
        if sha256_file(path / name) != digest:
            raise ValueError(f"Changed fold output: {member}/{fold}/{name}")
    return path


def run(root):
    root = Path(root).resolve()
    verify(root)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (root / "contract.json").exists():
            write(
                root / "contract.json",
                {
                    "recipe": RECIPE,
                    "code_sha256": code_hashes(),
                    "data_manifest_sha256": sha256_file(root / "data/manifest.json"),
                    "allocation_hours": 4.0,
                    "allocation": "new_user_authorized_briford_reproduction",
                    "submission_authorized": True,
                    "open_source_code": False,
                    "model_report_link": "",
                    "versions": {
                        p: importlib.metadata.version(p)
                        for p in ["chemprop", "torch", "lightning", "rdkit", "numpy", "pandas"]
                    },
                },
            )
        check_contract(root)
        series = (
            read(root / "series.json")
            if (root / "series.json").exists()
            else {
                "used_seconds": 0.0,
                "limit_seconds": 14400.0,
                "attempts": [],
                "status": "prepared",
            }
        )
        # Charge an interrupted supervisor conservatively up to its saved reservation.
        if series.get("active"):
            series["used_seconds"] += series["active"]["reserved_seconds"]
            series["attempts"].append(
                {**series.pop("active"), "status": "interrupted_reservation_charged"}
            )
            write(root / "series.json", series)
        for member in MEMBERS:
            for fold in range(5):
                path = root / "fits" / member / str(fold)
                if (path / "complete.json").exists():
                    verify_complete(root, member, fold)
                    continue
                remaining = series["limit_seconds"] - series["used_seconds"]
                if remaining < 90:
                    series["status"] = "budget_exhausted"
                    write(root / "series.json", series)
                    return series
                check_contract(root)
                path.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                series.update(
                    status="running",
                    active={
                        "member": member,
                        "fold": fold,
                        "reserved_seconds": remaining,
                        "started_unix": time.time(),
                    },
                )
                write(root / "series.json", series)
                with (path / "fit.log").open("a") as log:
                    command = [
                        sys.executable,
                        "-m",
                        "somerra_openadmet.briford",
                        "worker",
                        "--output",
                        str(root),
                        "--member",
                        member,
                        "--fold",
                        str(fold),
                        "--seconds",
                        str(remaining),
                    ]
                    try:
                        result = subprocess.run(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=remaining,
                            check=False,
                        )
                        code = result.returncode
                    except subprocess.TimeoutExpired:
                        code = 124
                elapsed = time.monotonic() - started
                series["used_seconds"] += elapsed
                series["attempts"].append(
                    {**series.pop("active"), "used_seconds": elapsed, "returncode": code}
                )
                series["status"] = "running" if code == 0 else "failed"
                write(root / "series.json", series)
                if code:
                    raise RuntimeError(f"Fold {member}/{fold} failed; see {path / 'fit.log'}")
                verify_complete(root, member, fold)
        series["status"] = "training_complete"
        write(root / "series.json", series)
        result = export(root)
        series["status"] = "predictions_prepared_not_submitted"
        series["submission_sha256"] = result["submission_sha256"]
        write(root / "series.json", series)
        return series


def calibrated(values, isoform, correlation):
    """Historical --oof --strae placement; sample SD, then floor at 1.0."""
    values = np.asarray(values, dtype=float)
    rho = float(correlation * OOF_FACTORS[isoform])
    if not np.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("Historical correlation extrapolation is outside its valid range")
    mean, sd = MOMENTS[isoform]
    target_sd = rho * sd
    if isoform == "cyp2d6":
        mean, target_sd = 3.57, 0.90
    observed_sd = values.std(ddof=1)
    if not observed_sd > 0:
        raise ValueError("Cannot calibrate constant predictions")
    transformed = mean + (values - values.mean()) * target_sd / observed_sd
    return np.maximum(transformed, 1.0), {
        "oof_pearson": correlation,
        "adjusted_pearson": rho,
        "target_mean": mean,
        "target_sd": target_sd,
        "raw_mean": float(values.mean()),
        "raw_sd": float(observed_sd),
        "floor_count": int((transformed < 1).sum()),
    }


def export(root):
    root = Path(root)
    manifest = check_contract(root)
    predictions, captures = {}, {}
    for member in MEMBERS:
        frames, blinds = [], []
        tasks = manifest["members"][member]["tasks"]
        expected = pd.read_csv(root / "data" / f"{member}.csv").set_index("molecule_name")
        for k in range(5):
            path = verify_complete(root, member, k)
            f = pd.read_csv(path / "oof.csv").set_index("molecule_name")
            if set(f.index) != set(expected.index[expected.fold == k]):
                raise ValueError("OOF identities do not match the frozen fold")
            frames.append(f)
            b = pd.read_csv(path / "blind.csv").set_index("molecule_name")
            blinds.append(b)
        combined = pd.concat(frames)
        if combined.index.duplicated().any() or set(combined.index) != set(expected.index):
            raise ValueError("Incomplete OOF coverage")
        if any(set(b.index) != set(blinds[0].index) for b in blinds):
            raise ValueError("Blind member identity mismatch")
        captures[member] = combined[tasks]
        predictions[member] = sum(b.reindex(blinds[0].index)[tasks] for b in blinds) / 5
    official = pd.read_csv(root / "data/official.csv").set_index("molecule_name")
    blind = pd.read_csv(root / "data/blind.csv")
    raw_sub = blind[["SMILES", "Molecule_Name"]].copy()
    final_sub = raw_sub.copy()
    oof = pd.DataFrame(index=official.index, columns=list(ACTIVITY_ENDPOINTS), dtype=float)
    details = {}
    for ep in ACTIVITY_ENDPOINTS:
        task = ep.lower()
        names = MEMBERS if task.startswith("cyp2d6") else MEMBERS[:2]
        ids = official.index[official[task].notna()]
        for name in names:
            if not set(ids) <= set(captures[name].index):
                raise ValueError("Missing scored OOF labels")
        vector = sum(captures[n].loc[ids, task] for n in names) / len(names)
        rho = float(pearsonr(official.loc[ids, task], vector).statistic)
        raw = sum(
            predictions[n].reindex(blind.Molecule_Name)[task].to_numpy() for n in names
        ) / len(names)
        raw_sub[ep] = raw
        final_sub[ep], details[ep] = calibrated(raw, task.split("_")[0], rho)
        oof.loc[ids, ep] = vector
        # Fill unlabelled positions only for shared scorer's finite shape contract.
        remaining = official.index.difference(ids)
        oof.loc[remaining, ep] = sum(captures[n].reindex(remaining)[task] for n in MEMBERS[:2]) / 2
    truth = official.reset_index().rename(columns={"molecule_name": "Molecule_Name"})
    for ep in ACTIVITY_ENDPOINTS:
        truth[ep] = truth[ep.lower()]
        truth[ep + "_conf_low"] = truth[ep.lower() + "_ci_lower"]
        truth[ep + "_conf_high"] = truth[ep.lower() + "_ci_upper"]
    scores = score_predictions(truth, oof.reset_index(drop=True))
    destination = root / "predictions"
    destination.mkdir(exist_ok=True)
    raw_sub.to_csv(destination / "raw.csv", index=False)
    final_sub.to_csv(destination / "activity_submission.csv", index=False)
    oof.to_csv(destination / "raw-oof.csv")
    validation = validate_submission(destination / "activity_submission.csv", blind, "activity")
    if not validation.valid or validation.warnings:
        raise ValueError(validation.to_dict())
    for ep in ACTIVITY_ENDPOINTS:
        unbound = final_sub[ep] > 1
        if not raw_sub.loc[unbound, ep].rank().equals(final_sub.loc[unbound, ep].rank()):
            raise ValueError("Calibration reordered predictions above the floor")
    result = {
        "status": "completed",
        "submission_sha256": sha256_file(destination / "activity_submission.csv"),
        "test_sha256": manifest["official_test_sha256"],
        "contract_sha256": sha256_file(root / "contract.json"),
        "open_source_code": False,
        "model_report_link": "",
        "submission_authorized": True,
        "validation": validation.to_dict(),
        "calibration": details,
        "raw_oof_metrics": scores,
        "limitation": (
            "OOF selects checkpoints; calibration includes historical "
            "leaderboard-derived constants."
        ),
        "training": "20 freshly trained networks; no third-party predictions or checkpoints",
    }
    write(destination / "run_manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "status", "predict", "worker"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--official-raw", type=Path)
    parser.add_argument("--member", choices=MEMBERS)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output / "data", args.source, args.official_raw)
    elif args.command == "run":
        result = run(args.output)
    elif args.command == "predict":
        result = export(args.output)
    elif args.command == "worker":
        return worker(args.output, args.member, args.fold, args.seconds)
    else:
        result = (
            read(args.output / "series.json")
            if (args.output / "series.json").exists()
            else {"status": "not_started"}
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
