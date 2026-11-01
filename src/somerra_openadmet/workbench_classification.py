"""Two-head TDI adaptation of the private Workbench regression recipe.

Checkpoint-selected OOF and OOF-tuned thresholds are development results.
This module prepares private predictions; it never uploads them.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .classification_metrics import best_threshold, diagnostics
from .constants import IDENTIFIER_COLUMNS, TDI_ENDPOINTS, TDI_TRAIN_FILE, TEST_FILE
from .data import load_tdi, load_test
from .data_sources.common import parent_smiles
from .download import sha256_file, verify_activity_data
from .validation import validate_submission


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def scaffold_folds(smiles, seed=42, n_folds=5):
    """Workbench first-seen scaffold IDs and shuffled greedy row balancing.

    Adapted from MIT-licensed Workbench revision 8c72c7559471febaf0962fd551b4eaab8db3a136.
    For this new TDI bundle, callers supply standardized parents for grouping only.
    """
    keys, group_ids, lookup = [], [], {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError("Invalid molecular structure")
        key = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        keys.append(key)
        if key not in lookup:
            lookup[key] = len(lookup)
        group_ids.append(lookup[key])
    groups, counts = np.unique(group_ids, return_counts=True)
    if len(groups) < n_folds:
        raise ValueError("Insufficient scaffold groups")
    loads = np.zeros(n_folds, dtype=int)
    mapping = {}
    for i in np.random.default_rng(seed).permutation(len(groups)):
        fold = int(np.argmin(loads))
        mapping[groups[i]] = fold
        loads[fold] += counts[i]
    return np.array([mapping[g] for g in group_ids]), keys


def task_weights(labels):
    labels = np.asarray(labels, dtype=float)
    counts = np.isfinite(labels).sum(axis=0)
    for column in labels.T:
        observed = column[np.isfinite(column)]
        if set(np.unique(observed)) != {0.0, 1.0}:
            raise ValueError("Each training head requires both binary classes")
    inverse = 1.0 / counts
    return (inverse / inverse.mean()).tolist()


def prepare(config_path):
    config = read(config_path)
    root, source = Path(config["output_dir"]).resolve(), Path(config["data_dir"]).resolve()
    if root.exists():
        raise FileExistsError(root)
    if config["open_source_code"] or config["model_report_link"] or config["upload_authorized"]:
        raise ValueError("This runner only prepares private prediction files")
    verification = verify_activity_data(source)
    if not verification.valid:
        raise ValueError(verification.errors)
    frame, blind = load_tdi(source), load_test(source)
    frame = frame[list(IDENTIFIER_COLUMNS) + list(TDI_ENDPOINTS)].copy()
    if frame.Molecule_Name.duplicated().any() or frame[list(IDENTIFIER_COLUMNS)].isna().any().any():
        raise ValueError("Invalid training identities")
    for ep in TDI_ENDPOINTS:
        frame[ep] = frame[ep].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    frame["parent"] = frame.SMILES.map(parent_smiles)
    blind_parents = set(blind.SMILES.map(parent_smiles))
    excluded = frame.loc[frame.parent.isin(blind_parents)].copy()
    frame = frame.loc[
        ~frame.parent.isin(blind_parents) & frame[list(TDI_ENDPOINTS)].notna().any(axis=1)
    ].reset_index(drop=True)
    conflicts = {ep: int((frame.groupby("parent")[ep].nunique() > 1).sum()) for ep in TDI_ENDPOINTS}
    if any(conflicts.values()):
        raise ValueError(f"Conflicting labels for molecular parents: {conflicts}")
    frame["fold"], frame["scaffold"] = scaffold_folds(
        frame.parent, config["seed"], config["n_folds"]
    )
    fold_counts = {}
    for fold in range(config["n_folds"]):
        train, val = frame.loc[frame.fold != fold], frame.loc[frame.fold == fold]
        if set(train.parent) & set(val.parent) or set(train.scaffold) & set(val.scaffold):
            raise ValueError("Molecular parent/scaffold leakage")
        weights = task_weights(train[list(TDI_ENDPOINTS)])
        task_weights(val[list(TDI_ENDPOINTS)])  # Reject a one-class validation head.
        fold_counts[str(fold)] = {
            "training_rows": len(train),
            "validation_rows": len(val),
            "task_weights": weights,
            "validation_labels": {ep: int(val[ep].count()) for ep in TDI_ENDPOINTS},
            "validation_positives": {ep: int(val[ep].sum()) for ep in TDI_ENDPOINTS},
        }
    (root / "data").mkdir(parents=True)
    frame.to_csv(root / "data/training.csv", index=False)
    blind.to_csv(root / "data/blind.csv", index=False)
    excluded.to_csv(root / "data/excluded_blind_parents.csv", index=False)
    write(root / "config.json", config)
    package = Path(__file__).parent
    snapshot = root / "source/somerra_openadmet"
    snapshot.mkdir(parents=True)
    for file in package.rglob("*.py"):
        target = snapshot / file.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    license_path = Path(__file__).resolve().parents[2] / "licenses/Workbench-MIT.txt"
    if license_path.exists():
        shutil.copyfile(license_path, root / "source/Workbench-MIT.txt")
    manifest = {
        "status": "prepared",
        "rows": len(frame),
        "parents": frame.parent.nunique(),
        "excluded_blind_overlap": len(excluded),
        "conflicting_parents": conflicts,
        "endpoints": list(TDI_ENDPOINTS),
        "folds": fold_counts,
        "input_sha256": {name: sha256_file(source / name) for name in [TDI_TRAIN_FILE, TEST_FILE]},
        "file_sha256": {
            str(p.relative_to(root)): sha256_file(p)
            for p in [
                *sorted((root / "data").glob("*.csv")),
                root / "config.json",
                *sorted(snapshot.rglob("*.py")),
            ]
        },
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["chemprop", "torch", "lightning", "rdkit", "numpy", "pandas"]
        },
        "split": "New TDI bundle; Workbench greedy scaffold method on standardized parents",
        "model_inputs": "original official SMILES only",
        "supervision": "two official binary TDI heads; no auxiliary or external labels",
        "validation": "same fold selects checkpoint and supplies development OOF",
        "thresholds": "one per endpoint, exact MCC maximization on pooled development OOF",
        "independent_internal_test": False,
        "budget": "separate bounded classification run; prior regression ledgers unchanged",
        "open_source_code": False,
        "model_report_link": "",
        "upload_authorized": False,
    }
    write(root / "manifest.json", manifest)
    return manifest


def verify(root):
    manifest = read(root / "manifest.json")
    for name, digest in manifest["file_sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Frozen artifact changed: {name}")
    if (
        sha256_file(Path(__file__))
        != manifest["file_sha256"]["source/somerra_openadmet/workbench_classification.py"]
    ):
        raise ValueError("Runner differs from frozen source")
    return manifest


def make_model(config, weights):
    import torch
    from chemprop import models, nn

    encoder = nn.BondMessagePassing(
        d_h=config["hidden_dim"], depth=config["depth"], dropout=config["dropout"]
    )
    head = nn.BinaryClassificationFFN(
        input_dim=config["hidden_dim"],
        hidden_dim=config["ffn_hidden_dim"],
        n_layers=config["ffn_num_layers"],
        dropout=config["dropout"],
        n_tasks=2,
        criterion=nn.metrics.BCELoss(task_weights=torch.tensor(weights)),
    )
    return models.MPNN(
        encoder,
        nn.NormAggregation(norm=100),
        head,
        batch_norm=True,
        metrics=[],
        warmup_epochs=config["warmup_epochs"],
        init_lr=config["init_lr"],
        max_lr=config["max_lr"],
        final_lr=config["final_lr"],
    )


def predict(model, smiles, batch_size=64):
    import torch
    from chemprop import data

    dataset = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in smiles])
    dataset.cache = True
    loader = data.build_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )
    model.eval()
    before = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    result = []
    try:
        torch.use_deterministic_algorithms(True)
        with torch.inference_mode():
            for batch in loader:
                batch.bmg.to(model.device)
                result.append(model(batch.bmg).cpu().numpy())
    finally:
        torch.use_deterministic_algorithms(before, warn_only=warn)
    values = np.concatenate(result)
    if values.shape != (len(smiles), 2) or not np.isfinite(values).all():
        raise ValueError("Invalid prediction shape or nonfinite values")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Outputs are not probabilities")
    return values


def worker(root, fold, seconds):
    import lightning as pl
    import torch
    from chemprop import data
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    manifest = verify(root)
    config = read(root / "config.json")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("medium")
    torch.cuda.set_per_process_memory_fraction(0.5)
    pl.seed_everything(config["seed"] + fold, workers=True)
    frame, blind = pd.read_csv(root / "data/training.csv"), pd.read_csv(root / "data/blind.csv")
    train, val = frame.loc[frame.fold != fold], frame.loc[frame.fold == fold]
    weights = task_weights(train[list(TDI_ENDPOINTS)])
    if weights != manifest["folds"][str(fold)]["task_weights"]:
        raise ValueError("Training weight mismatch")
    out = root / "fits" / str(fold)
    out.mkdir(parents=True, exist_ok=False)

    def dataset(rows):
        ds = data.MoleculeDataset(
            [
                data.MoleculeDatapoint.from_smi(s, y)
                for s, y in zip(rows.SMILES, rows[list(TDI_ENDPOINTS)].to_numpy(float), strict=True)
            ]
        )
        ds.cache = True
        return ds

    loaders = {
        "num_workers": config["num_workers"],
        "pin_memory": True,
        "persistent_workers": config["num_workers"] > 0,
    }
    train_loader = data.build_dataloader(
        dataset(train), batch_size=config["batch_size"], shuffle=True, **loaders
    )
    val_loader = data.build_dataloader(
        dataset(val), batch_size=config["batch_size"], shuffle=False, drop_last=False, **loaders
    )
    model = make_model(config, weights)
    checkpoint = ModelCheckpoint(
        dirpath=out,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        enable_version_counter=False,
    )
    early = EarlyStopping(monitor="val_loss", patience=config["patience"], mode="min")
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

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config["max_epochs"],
        precision=config["precision"],
        logger=False,
        enable_progress_bar=False,
        callbacks=[early, checkpoint, Curve()],
        max_time={"seconds": max(1, int(seconds - 60))},
    )
    trainer.fit(model, train_loader, val_loader)
    if trainer.current_epoch < config["max_epochs"] and not early.stopped_epoch:
        raise TimeoutError("Time cap reached before early stopping or epoch completion")
    state = torch.load(out / "best.ckpt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.to("cuda")
    oof, test = predict(model, val.SMILES), predict(model, blind.SMILES)
    reloaded = make_model(config, weights)
    reloaded.load_state_dict(state["state_dict"])
    reloaded.to("cuda")
    reload_error = float(np.max(np.abs(predict(reloaded, blind.SMILES) - test)))
    if reload_error > 1e-6:
        raise ValueError(f"Checkpoint replay failed: {reload_error}")
    for name, rows, probabilities in [("oof", val, oof), ("blind", blind, test)]:
        result = rows[list(IDENTIFIER_COLUMNS)].copy()
        result[list(TDI_ENDPOINTS)] = probabilities
        result.to_csv(out / f"{name}.csv", index=False)
    write(
        out / "complete.json",
        {
            "fold": fold,
            "epochs": trainer.current_epoch,
            "best_epoch": int(state["epoch"]) + 1,
            "best_val_bce": float(checkpoint.best_model_score.cpu()),
            "task_weights": weights,
            "reload_max_abs_error": reload_error,
            "wall_seconds": time.monotonic() - started,
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "sha256": {
                name: sha256_file(out / name) for name in ["best.ckpt", "oof.csv", "blind.csv"]
            },
        },
    )


def export(root):
    verify(root)
    config = read(root / "config.json")
    frame, blind = pd.read_csv(root / "data/training.csv"), pd.read_csv(root / "data/blind.csv")
    oof = np.full((len(frame), 2), np.nan)
    test_predictions, completions = [], []
    for fold in range(config["n_folds"]):
        out = root / "fits" / str(fold)
        record = read(out / "complete.json")
        if record["manifest_sha256"] != sha256_file(root / "manifest.json"):
            raise ValueError("Fold trained against a different manifest")
        for name, digest in record["sha256"].items():
            if sha256_file(out / name) != digest:
                raise ValueError(f"Changed fold artifact {fold}/{name}")
        holdout, test = pd.read_csv(out / "oof.csv"), pd.read_csv(out / "blind.csv")
        wanted = frame.loc[frame.fold == fold, "Molecule_Name"]
        if list(holdout.Molecule_Name) != list(wanted):
            raise ValueError("OOF identity mismatch")
        if not test[list(IDENTIFIER_COLUMNS)].equals(blind[list(IDENTIFIER_COLUMNS)]):
            raise ValueError("Blind identity mismatch")
        oof[frame.fold == fold] = holdout[list(TDI_ENDPOINTS)].to_numpy()
        test_predictions.append(test[list(TDI_ENDPOINTS)].to_numpy())
        completions.append(record)
    probability = np.mean(test_predictions, axis=0)
    if not np.isfinite(oof).all() or not np.isfinite(probability).all():
        raise ValueError("Incomplete predictions")
    report, thresholds = {}, {}
    submission = blind[list(IDENTIFIER_COLUMNS)].copy()
    oof_table = frame.copy()
    for j, ep in enumerate(TDI_ENDPOINTS):
        observed = frame[ep].notna().to_numpy()
        y, p = frame.loc[observed, ep].to_numpy(int), oof[observed, j]
        threshold, _ = best_threshold(y, p)
        thresholds[ep] = threshold
        report[ep] = {
            "at_0_5": diagnostics(y, p, p >= 0.5),
            "oof_tuned": diagnostics(y, p, p >= threshold),
            "threshold": threshold,
            "blind_positive_count": int((probability[:, j] >= threshold).sum()),
        }
        submission[ep] = probability[:, j] >= threshold
        oof_table[ep + "_probability"] = oof[:, j]
    if any(submission[ep].nunique() != 2 for ep in TDI_ENDPOINTS):
        raise ValueError("Portal rejects constant classification predictions")
    output = root / "predictions"
    output.mkdir(exist_ok=True)
    submission.to_csv(output / "tdi_submission.csv", index=False)
    probabilities = blind[list(IDENTIFIER_COLUMNS)].copy()
    probabilities[list(TDI_ENDPOINTS)] = probability
    probabilities.to_csv(output / "blind_probabilities.csv", index=False)
    oof_table.to_csv(output / "oof_probabilities.csv", index=False)
    validation = validate_submission(output / "tdi_submission.csv", blind, "tdi")
    if not validation.valid or validation.warnings:
        raise ValueError(validation.to_dict())
    summary = {
        "status": "candidate_prepared",
        "submitted": False,
        "validation": validation.to_dict(),
        "endpoints": report,
        "thresholds": thresholds,
        "macro_mcc_0_5": float(np.mean([v["at_0_5"]["mcc"] for v in report.values()])),
        "macro_mcc_oof_tuned": float(np.mean([v["oof_tuned"]["mcc"] for v in report.values()])),
        "evaluation_caveat": "Checkpoint-selected OOF; thresholds tuned on the same labels. "
        "Development estimates, not independent validation.",
        "folds": completions,
        "submission_sha256": sha256_file(output / "tdi_submission.csv"),
        "open_source_code": False,
        "model_report_link": "",
        "upload_authorized": False,
    }
    write(output / "summary.json", summary)
    return summary


def run(root):
    verify(root)
    config = read(root / "config.json")
    with (root / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = root / "ledger.json"
        ledger = (
            read(ledger_path)
            if ledger_path.exists()
            else {
                "limit_seconds": config["max_wall_hours"] * 3600,
                "attempts": [],
                "accounting": "serial worker wall seconds; separate classification allocation",
            }
        )
        if any(a["status"] == "running" for a in ledger["attempts"]):
            raise RuntimeError(
                "Unsettled attempt; reconcile its process and charge before resuming"
            )
        ledger["status"] = "running"
        write(ledger_path, ledger)
        for fold in range(config["n_folds"]):
            if (root / "fits" / str(fold) / "complete.json").exists():
                continue
            used = sum(a["used_seconds"] for a in ledger["attempts"])
            remaining = ledger["limit_seconds"] - used
            if remaining < 120:
                raise TimeoutError("Classification time allocation exhausted")
            attempt = {
                "fold": fold,
                "status": "running",
                "started_unix": time.time(),
                "reserved_seconds": remaining,
            }
            ledger["attempts"].append(attempt)
            write(ledger_path, ledger)
            started = time.monotonic()
            env = dict(
                os.environ,
                PYTHONPATH=str(root / "source"),
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
                OMP_NUM_THREADS="4",
            )
            try:
                with (root / f"fold-{fold}.log").open("a") as log:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "somerra_openadmet.workbench_classification",
                            "worker",
                            "--output",
                            str(root),
                            "--fold",
                            str(fold),
                            "--seconds",
                            str(remaining),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=remaining,
                        check=False,
                    )
                if result.returncode:
                    raise RuntimeError(f"Fold {fold} failed; inspect fold-{fold}.log")
                attempt["status"] = "complete"
            except BaseException as exc:
                attempt["status"] = "failed"
                attempt["error"] = repr(exc)
                ledger["status"] = "failed"
                raise
            finally:
                attempt["used_seconds"] = time.monotonic() - started
                write(ledger_path, ledger)
            print(json.dumps({"fold": fold, "status": "complete"}), flush=True)
        summary = export(root)
        ledger["status"] = "candidate_prepared"
        write(ledger_path, ledger)
        print(
            json.dumps({k: summary[k] for k in ["status", "macro_mcc_0_5", "macro_mcc_oof_tuned"]}),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "worker", "export"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seconds", type=float, default=7200)
    args = parser.parse_args()
    if args.stage == "prepare":
        print(json.dumps(prepare(args.config), indent=2))
    else:
        root = args.output.resolve()
        if args.stage == "run":
            run(root)
        elif args.stage == "worker":
            worker(root, args.fold, args.seconds)
        else:
            print(json.dumps(export(root), indent=2))


if __name__ == "__main__":
    main()
