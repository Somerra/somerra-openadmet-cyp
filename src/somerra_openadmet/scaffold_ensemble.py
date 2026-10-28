"""Frozen scaffold folds, rotating held-out evaluation and equal-weight blind ensemble."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ACTIVITY_ENDPOINTS
from .data_sources.common import write_json
from .download import sha256_file
from .molecular_similarity import scaffold

PROTOCOL = "scaffold-five-fold-v1"


def assign_folds(pool, official):
    """Balance whole scaffolds using sizes/label availability, never target values."""
    parents = sorted(set(pool.parent) | set(official.parent))
    assignments = pd.DataFrame({"parent": parents})
    assignments["scaffold"] = [scaffold(p) for p in parents]
    mapping = assignments.set_index("parent").scaffold
    sizes = pool.assign(scaffold=pool.parent.map(mapping)).groupby("scaffold").size()
    counts = official[list(ACTIVITY_ENDPOINTS)].notna().astype(int)
    counts["official_rows"] = 1
    counts["scaffold"] = official.parent.map(mapping)
    features = counts.groupby("scaffold").sum().join(sizes.rename("pool_rows"), how="outer")
    features = features.fillna(0).sort_index()
    if len(features) < 5:
        raise ValueError("Need at least five distinct scaffolds")
    values = features.to_numpy(float)
    targets = np.maximum(values.sum(axis=0) / 5, 1)
    rng = np.random.default_rng(42)
    # Largest relative groups first; seeded ties, deterministic across row order.
    tie = rng.random(len(values))
    order = sorted(range(len(values)), key=lambda i: (-max(values[i] / targets), tie[i]))
    loads = np.zeros((5, values.shape[1]))
    folds = {}
    for i in order:
        costs = (((loads + values[i]) / targets) ** 2 - (loads / targets) ** 2).sum(axis=1)
        choices = np.flatnonzero(np.isclose(costs, costs.min(), rtol=0, atol=1e-12))
        fold = int(rng.choice(choices))
        folds[features.index[i]] = fold
        loads[fold] += values[i]
    assignments["fold"] = assignments.scaffold.map(folds)
    return assignments


def partitions(folds, heldout):
    return np.where(
        folds == heldout,
        "internal_test",
        np.where(folds == (heldout + 1) % 5, "validation", "train"),
    )


def load_fold_frames(bundle, manifest, recipe="full", final=False):
    """Enforce scaffold separation across official AND auxiliary rows at load time."""
    from .reference_member_data import select_tasks

    bundle = Path(bundle)
    assignments = pd.read_csv(bundle / "folds.csv", keep_default_na=False)
    official = pd.read_csv(bundle / "official.csv")
    pool = pd.read_csv(bundle / "pool.csv")
    blind = set(pd.read_csv(bundle / "test.csv").parent)
    k = manifest["fold"]
    if not isinstance(k, int) or k not in range(5):
        raise ValueError("Invalid scaffold fold")
    if (
        assignments.parent.duplicated().any()
        or set(assignments.fold) != set(range(5))
        or assignments.groupby("scaffold").fold.nunique().max() != 1
    ):
        raise ValueError("Scaffold crosses folds or invalid assignments")
    if set(assignments.parent) != set(pool.parent) | set(official.parent):
        raise ValueError("Incomplete scaffold parent coverage")
    if set(assignments.parent) & blind:
        raise ValueError("Official blind parents in scaffold training pool")
    # Recompute chemistry: a modified assignment cannot invent scaffold identities.
    if assignments.scaffold.tolist() != [scaffold(p) for p in assignments.parent]:
        raise ValueError("Scaffold identity mismatch")
    mapping = assignments.set_index("parent").fold
    single = manifest.get("protocol") == "scaffold-single-v1"
    if final and not single:
        raise ValueError("Scaffold folds must retain validation and held-out scaffolds")
    expected = (
        np.where(official.parent.map(mapping) == k, "validation", "train")
        if single
        else partitions(official.parent.map(mapping), k)
    )
    if not np.array_equal(official.partition.to_numpy(), expected):
        raise ValueError("Official partition differs from scaffold rotation")
    if manifest.get("exclusion_policy") != "whole_scaffold":
        raise ValueError("Scaffold exclusion policy mismatch")
    pool_folds = pool.parent.map(mapping)
    if not final:
        pool = pool.loc[~pool_folds.isin([k] if single else [k, (k + 1) % 5])].copy()
    tasks = select_tasks(manifest, recipe)
    pool = pool.loc[pool[tasks].notna().any(axis=1)].reset_index(drop=True)
    validation = official.loc[official.partition == "validation"].reset_index(drop=True)
    return pool, validation, tasks, manifest


def prepare(base, output, model, min_labels=100):
    from .model_recipe import resolve_model
    from .reference_member_data import load_frames

    base, output = Path(base).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    pool, _, _, original = load_frames(base, final=True)
    official = pd.read_csv(base / "official.csv")
    # Preserve the verified complete pool, including source record weights and bounds.
    full = pd.read_csv(base / "pool.csv")
    blind = pd.read_csv(base / "test.csv")
    full = full.loc[~full.parent.isin(set(blind.parent))].reset_index(drop=True)
    if set(official.parent) & set(blind.parent):
        raise ValueError("Official train/blind parent overlap")
    assignment = assign_folds(full, official)
    mapping = assignment.set_index("parent").fold
    official_folds = official.parent.map(mapping)
    coverage = official[list(ACTIVITY_ENDPOINTS)].notna().groupby(official_folds).sum()
    if len(coverage) != 5 or (coverage < min_labels).any().any():
        raise ValueError(f"Insufficient endpoint coverage; no scaffold groups split: {coverage}")
    output.mkdir(parents=True)
    assignment.to_csv(output / "folds.csv", index=False)
    write_json(output / "model.json", resolve_model(json.loads(Path(model).read_text())))
    summaries = []
    for k in range(5):
        dest = output / f"fold-{k}" / "data"
        dest.mkdir(parents=True)
        fold_official = official.copy()
        fold_official["partition"] = partitions(official_folds, k)
        fold_official.to_csv(dest / "official.csv", index=False)
        full.to_csv(dest / "pool.csv", index=False)
        for name, source in [("test.csv", base / "test.csv"), ("folds.csv", output / "folds.csv")]:
            (dest / name).write_bytes(source.read_bytes())
        manifest = {
            **original,
            "protocol": PROTOCOL,
            "fold": k,
            "exclusion_policy": "whole_scaffold",
            "seed": 42,
            "source_bundle": str(base),
            "source_bundle_manifest_sha256": sha256_file(base / "manifest.json"),
            "folds_sha256": sha256_file(output / "folds.csv"),
            "validation_role": "Next scaffold fold selects checkpoint; held-out fold only scores",
            "partition_rows": fold_official.partition.value_counts().to_dict(),
            "validation_label_counts": coverage.loc[(k + 1) % 5].to_dict(),
            "internal_test_label_counts": coverage.loc[k].to_dict(),
            "file_sha256": {
                n: sha256_file(dest / n)
                for n in ["official.csv", "pool.csv", "test.csv", "folds.csv"]
            },
        }
        # Do not carry obsolete assignment provenance as if it described this split.
        for key in ["assignments_sha256", "split_manifest_sha256"]:
            manifest.pop(key, None)
        write_json(dest / "manifest.json", manifest)
        train, valid, _, _ = load_frames(dest)
        train_scaffolds = set(train.parent.map(assignment.set_index("parent").scaffold))
        held = fold_official.loc[fold_official.partition == "internal_test"]
        held_scaffolds = set(held.parent.map(assignment.set_index("parent").scaffold))
        val_scaffolds = set(valid.parent.map(assignment.set_index("parent").scaffold))
        if train_scaffolds & (held_scaffolds | val_scaffolds) or held_scaffolds & val_scaffolds:
            raise ValueError("Scaffold leakage")
        summaries.append(
            {
                "fold": k,
                "training_records": len(train),
                "training_parents": train.parent.nunique(),
                "official_partition_rows": manifest["partition_rows"],
                "heldout_label_counts": coverage.loc[k].to_dict(),
                "scaffold_overlap": 0,
            }
        )
    plan = {
        "status": "prepared",
        "protocol": PROTOCOL,
        "seed": 42,
        "epochs": 100,
        "scheduler_horizon": 200,
        "folds": summaries,
        "folds_sha256": sha256_file(output / "folds.csv"),
        "model_sha256": sha256_file(output / "model.json"),
        "data_manifest_sha256": [
            sha256_file(output / f"fold-{k}/data/manifest.json") for k in range(5)
        ],
        "source_bundle_manifest_sha256": sha256_file(base / "manifest.json"),
        "source_training_records": len(pool),
        "pool_records": len(full),
        "official_records": len(official),
        "scaffolds": assignment.scaffold.nunique(),
        "blind_parent_overlap": 0,
        "ensemble_weights": [0.2] * 5,
        "oof_role": "One held-out prediction per official record; NOT an ensemble score",
        "open_source_code": False,
        "model_report_link": "",
    }
    write_json(output / "plan.json", plan)
    return plan


def aligned_mean(frames):
    """Reject order/identity drift instead of silently averaging different molecules."""
    if len(frames) != 5:
        raise ValueError("A complete five-fold ensemble is required")
    first = frames[0]
    for frame in frames:
        if frame.Molecule_Name.duplicated().any():
            raise ValueError("Duplicate prediction IDs")
        if not frame[["Molecule_Name", "SMILES"]].equals(first[["Molecule_Name", "SMILES"]]):
            raise ValueError("Ensemble molecule identity/order mismatch")
        if not np.isfinite(frame[list(ACTIVITY_ENDPOINTS)].to_numpy()).all():
            raise ValueError("Nonfinite ensemble member predictions")
    result = first[["Molecule_Name", "SMILES"]].copy()
    result[list(ACTIVITY_ENDPOINTS)] = np.mean(
        [f[list(ACTIVITY_ENDPOINTS)].to_numpy() for f in frames], axis=0
    )
    return result


def aggregate(output, raw):
    from .data import load_test
    from .reference_member import score_predictions
    from .validation import validate_submission

    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    frames, oof, members = [], [], []
    settings = json.loads((output / "model.json").read_text())
    sources = None
    for k in range(5):
        fold = output / f"fold-{k}"
        record = json.loads((fold / "fit/run_manifest.json").read_text())
        manifest = record["data_manifest"]
        if (
            not record.get("complete")
            or record["completed_epochs"] != 100
            or record["model_config"] != settings
            or record["seed"] != 42
            or manifest["fold"] != k
            or manifest["protocol"] != PROTOCOL
            or manifest["folds_sha256"] != plan["folds_sha256"]
            or record["data_manifest_sha256"] != plan["data_manifest_sha256"][k]
            or record["best.ckpt_sha256"] != sha256_file(fold / "fit/best.ckpt")
        ):
            raise ValueError("Incomplete or mismatched ensemble member")
        if sources is not None and sources != record["source_sha256"]:
            raise ValueError("Member implementations differ")
        sources = record["source_sha256"]
        export = json.loads((fold / "predictions/manifest.json").read_text())
        path = fold / "predictions/activity_submission.csv"
        if export["checkpoint_sha256"] != record["best.ckpt_sha256"] or export[
            "submission_sha256"
        ] != sha256_file(path):
            raise ValueError("Changed ensemble predictions or checkpoint provenance")
        frames.append(pd.read_csv(path))
        held = pd.read_csv(fold / "fit/internal-test.csv")
        official = pd.read_csv(fold / "data/official.csv")
        expected = official.loc[official.partition == "internal_test", ["Molecule_Name", "parent"]]
        if not held[["Molecule_Name", "parent"]].equals(expected.reset_index(drop=True)):
            raise ValueError("OOF identities differ from held-out scaffold fold")
        oof.append(held.assign(fold=k))
        members.append(
            {
                "fold": k,
                "run": str(fold / "fit"),
                "checkpoint_sha256": record["best.ckpt_sha256"],
                "selected_epoch": record["best_epoch"],
                "validation_score": record["best_score"],
                "heldout_score": record["internal_test_metrics"]["MA-ST-RAE"],
                "prediction_sha256": export["submission_sha256"],
                "weight": 0.2,
            }
        )
    combined = pd.concat(oof, ignore_index=True)
    truth = pd.read_csv(output / "fold-0/data/official.csv")
    if combined.Molecule_Name.duplicated().any() or set(combined.Molecule_Name) != set(
        truth.Molecule_Name
    ):
        raise ValueError("OOF coverage must be exactly once per official record")
    combined = combined.set_index("Molecule_Name").loc[truth.Molecule_Name].reset_index()
    score = score_predictions(truth, combined[list(ACTIVITY_ENDPOINTS)])
    dest = output / "ensemble"
    dest.mkdir(exist_ok=False)
    result = aligned_mean(frames)
    result.to_csv(dest / "activity_submission.csv", index=False)
    validation = validate_submission(dest / "activity_submission.csv", load_test(raw), "activity")
    if not validation.valid or validation.warnings:
        raise ValueError(validation.to_dict())
    combined.to_csv(dest / "oof.csv", index=False)
    report = {
        "status": "prepared_not_submitted",
        "members": members,
        "oof_metrics": score,
        "oof_role": plan["oof_role"],
        "ensemble_internal_score": None,
        "limitation": "OOF score is single held-out models. Averaging all five on training "
        "molecules would leak. Official ensemble score requires submission.",
        "validation": validation.to_dict(),
        "submission_sha256": sha256_file(dest / "activity_submission.csv"),
        "oof_sha256": sha256_file(dest / "oof.csv"),
        "open_source_code": False,
        "model_report_link": "",
    }
    write_json(dest / "manifest.json", report)
    return report


def run(output, raw, ledger_path, hours):
    from .campaign_budget import CampaignLedger
    from .configuration import source_hashes
    from .reference_member import export

    output, raw, ledger_path = map(lambda p: Path(p).resolve(), (output, raw, ledger_path))
    if not np.isfinite(hours) or not 0 < hours <= 2:
        raise ValueError("Provide a finite series allocation of at most two GPU-hours")
    plan = json.loads((output / "plan.json").read_text())
    if sha256_file(output / "model.json") != plan["model_sha256"]:
        raise ValueError("Changed frozen model")
    if sha256_file(output / "folds.csv") != plan["folds_sha256"]:
        raise ValueError("Changed frozen folds")
    state_path = output / "series.json"
    with state_path.open("x") as handle:
        json.dump({"status": "starting"}, handle)
    ledger = CampaignLedger(ledger_path)
    sources = source_hashes()
    state = {
        "status": "running",
        "gpu_limit_hours": hours,
        "runs": [],
        "source_sha256": sources,
        "ledger": str(ledger_path),
        "used_hours": 0.0,
    }
    try:
        for k in range(5):
            fold = output / f"fold-{k}"
            if sources != source_hashes():
                raise ValueError("Source changed during scaffold series")
            if sha256_file(fold / "data/manifest.json") != plan["data_manifest_sha256"][k]:
                raise ValueError("Changed frozen fold data")
            remaining = hours - state["used_hours"]
            allowance = min(0.13, remaining)
            if allowance < 0.05:
                raise RuntimeError("Series allocation exhausted; completed folds retained")
            dest = fold / "fit"
            ledger.reserve_many([(dest, allowance)])
            command = [
                sys.executable,
                "-m",
                "somerra_openadmet.submission_selection",
                "_train",
                "--bundle",
                str(fold / "data"),
                "--output",
                str(dest),
                "--strategy",
                str(output / "model.json"),
                "--ledger",
                str(ledger_path),
                "--hours",
                str(allowance),
            ]
            process, started = None, time.monotonic()
            state["current_fold"] = k
            write_json(state_path, state)
            try:
                with (fold / "fit.log").open("w") as log:
                    process = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    )
                    while process.poll() is None:
                        elapsed = time.monotonic() - started
                        curve = dest / "curve.jsonl"
                        stale = time.time() - curve.stat().st_mtime if curve.exists() else elapsed
                        if elapsed >= allowance * 3600 or stale > 600:
                            os.killpg(process.pid, signal.SIGKILL)
                            break
                        time.sleep(5)
                    code = process.wait()
            finally:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                used = (time.monotonic() - started) / 3600 if process else 0.0
                ledger.settle(dest, used, "exited" if process else "not_started")
            state["used_hours"] += used
            state["runs"].append({"fold": k, "run": str(dest), "hours": used, "returncode": code})
            write_json(state_path, state)
            if code != 0:
                raise RuntimeError(f"Fold {k} failed; checkpoints and logs retained")
            if sources != source_hashes():
                raise ValueError("Source changed during fold")
            export(dest, fold / "data", raw, fold / "predictions", selected_validation=True)
        report = aggregate(output, raw)
        state.update(
            status="prepared_not_submitted",
            oof_metrics=report["oof_metrics"],
            ensemble=str(output / "ensemble/activity_submission.csv"),
        )
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(state_path, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ["bundle", "model", "output"]:
        prep.add_argument("--" + name, type=Path, required=True)
    launch = commands.add_parser("run")
    for name in ["output", "raw", "ledger"]:
        launch.add_argument("--" + name, type=Path, required=True)
    launch.add_argument("--hours", type=float, required=True)
    status = commands.add_parser("status")
    status.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.bundle, args.output, args.model)
    elif args.command == "run":
        result = run(args.output, args.raw, args.ledger, args.hours)
    else:
        result = json.loads((args.output / "plan.json").read_text())
        if (args.output / "series.json").exists():
            result["series"] = json.loads((args.output / "series.json").read_text())
        result["progress"] = []
        for k in range(5):
            path = args.output / f"fold-{k}/fit/run_manifest.json"
            if path.exists():
                record = json.loads(path.read_text())
                result["progress"].append(
                    {
                        key: record.get(key)
                        for key in ["status", "completed_epochs", "best_epoch", "best_score"]
                    }
                    | {"fold": k}
                )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
