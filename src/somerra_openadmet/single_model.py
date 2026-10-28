"""R25-inspired single predictor: scaffold selection, then a fixed-epoch full-data fit."""

import argparse
import json
import math
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
from .foundation import encoder_identity
from .model_recipe import resolve_model
from .reference_member_data import load_frames

PROTOCOL = "scaffold-single-v1"


def prepare(source_fold, output, encoder, model_config=None):
    """Reuse audited folds; fold zero is fixed validation, never selected by scores."""
    source, output = Path(source_fold).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    _, _, tasks, original = load_frames(source)
    if original.get("protocol") != "scaffold-five-fold-v1":
        raise ValueError("Single-model preparation requires a verified scaffold-fold bundle")
    if len(tasks) != 26:
        raise ValueError("The R25-inspired single recipe requires the complete 26-task pool")
    identity = encoder_identity(encoder)
    model = resolve_model(model_config or {})
    if model["hidden_dim"] != 2048:
        raise ValueError("CheMeleon encoder requires hidden_dim 2048")
    assignment = pd.read_csv(source / "folds.csv", keep_default_na=False)
    official = pd.read_csv(source / "official.csv")
    official["partition"] = np.where(
        official.parent.map(assignment.set_index("parent").fold) == 0, "validation", "train"
    )
    coverage = official.loc[official.partition == "validation", list(ACTIVITY_ENDPOINTS)].count()
    if (coverage < 100).any():
        raise ValueError("Insufficient scaffold-validation coverage")
    output.mkdir(parents=True)
    bundle = output / "data"
    bundle.mkdir()
    official.to_csv(bundle / "official.csv", index=False)
    for name in ["pool.csv", "test.csv", "folds.csv"]:
        (bundle / name).write_bytes((source / name).read_bytes())
    manifest = {
        **original,
        "protocol": PROTOCOL,
        "fold": 0,
        "validation_role": "Fixed scaffold fold 0 selects epoch; no internal test",
        "source_fold_manifest_sha256": sha256_file(source / "manifest.json"),
        "partition_rows": official.partition.value_counts().to_dict(),
        "validation_label_counts": coverage.to_dict(),
        "file_sha256": {
            n: sha256_file(bundle / n)
            for n in ["official.csv", "pool.csv", "test.csv", "folds.csv"]
        },
    }
    manifest.pop("internal_test_label_counts", None)
    write_json(bundle / "manifest.json", manifest)
    train, valid, _, _ = load_frames(bundle)
    full, _, _, _ = load_frames(bundle, final=True)
    write_json(output / "model.json", model)
    plan = {
        "protocol": PROTOCOL,
        "status": "prepared",
        "seed": 42,
        "selection_epochs": 100,
        "scheduler_horizon": 200,
        "initialization": identity,
        "encoder_checkpoint": str(Path(encoder).resolve()),
        "model_sha256": sha256_file(output / "model.json"),
        "data_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "selection_training_records": len(train),
        "validation_records": len(valid),
        "final_training_records": len(full),
        "official_partition_rows": manifest["partition_rows"],
        "calibration": "none",
        "ensemble": False,
        "validation_limit": "Previously explored molecules; development score, not untouched test",
        "open_source_code": False,
        "model_report_link": "",
    }
    write_json(output / "plan.json", plan)
    return plan


def selected_epochs(record, plan, model):
    """Gate final fitting on a complete, provenance-matched selection run."""
    if (
        not record.get("complete")
        or record.get("final")
        or record.get("completed_epochs") != plan["selection_epochs"]
        or record.get("epoch_limit") != plan["selection_epochs"]
        or record.get("initialization") != plan["initialization"]
        or record.get("model_config") != model
        or record.get("data_manifest_sha256") != plan["data_manifest_sha256"]
        or record.get("seed") != plan["seed"]
        or record.get("scheduler_horizon") != plan["scheduler_horizon"]
    ):
        raise ValueError("Selection run incomplete or differs from frozen plan")
    epoch = record.get("best_epoch")
    if type(epoch) is not int or not 1 <= epoch <= plan["selection_epochs"]:
        raise ValueError("Invalid validation-selected epoch count")
    return epoch


def supervise(output, stage, plan, epochs, hours, ledger_path):
    from .campaign_budget import CampaignLedger

    dest = output / stage
    ledger = CampaignLedger(ledger_path)
    ledger.reserve_many([(dest, hours)])
    command = [
        sys.executable,
        "-m",
        "somerra_openadmet.single_model",
        "_train",
        "--output",
        str(output),
        "--stage",
        stage,
        "--epochs",
        str(epochs),
        "--hours",
        str(hours),
        "--ledger",
        str(ledger_path),
    ]
    process, started = None, time.monotonic()
    try:
        with (output / (stage + ".log")).open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            while process.poll() is None:
                elapsed = time.monotonic() - started
                curve = dest / "curve.jsonl"
                stale = time.time() - curve.stat().st_mtime if curve.exists() else elapsed
                if elapsed >= hours * 3600 or stale > 600:
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
    if code != 0:
        raise RuntimeError(f"{stage} failed; artifacts and ledger charge retained")
    record = json.loads((dest / "run_manifest.json").read_text())
    return record, used


def verify_inputs(output, plan):
    if (
        sha256_file(output / "model.json") != plan["model_sha256"]
        or sha256_file(output / "data/manifest.json") != plan["data_manifest_sha256"]
        or encoder_identity(plan["encoder_checkpoint"]) != plan["initialization"]
    ):
        raise ValueError("Changed frozen single-model inputs")
    load_frames(output / "data")


def run(output, raw, ledger_path, hours):
    from .campaign_budget import CampaignLedger
    from .configuration import source_hashes
    from .reference_member import export

    output, raw, ledger_path = map(lambda p: Path(p).resolve(), (output, raw, ledger_path))
    if not math.isfinite(hours) or not 0 < hours <= 2:
        raise ValueError("Provide at most two hours for this two-fit experiment")
    ledger = CampaignLedger(ledger_path)
    if ledger.snapshot()["committed_hours"] + hours > 12:
        raise ValueError("Insufficient remaining shared allocation")
    plan = json.loads((output / "plan.json").read_text())
    verify_inputs(output, plan)
    model = json.loads((output / "model.json").read_text())
    state_path = output / "run.json"
    with state_path.open("x") as handle:
        json.dump({"status": "starting"}, handle)
    sources = source_hashes()
    state = {
        "status": "selecting",
        "gpu_limit_hours": hours,
        "runs": [],
        "ledger": str(ledger_path),
        "source_sha256": sources,
    }
    try:
        write_json(state_path, state)
        record, used = supervise(output, "selection", plan, 100, hours / 2, ledger_path)
        state["runs"].append({"stage": "selection", "hours": used})
        if sources != source_hashes() or record["source_sha256"] != sources:
            raise ValueError("Source changed during selection")
        epoch = selected_epochs(record, plan, model)
        verify_inputs(output, plan)
        decision = {
            "epochs": epoch,
            "validation_score": record["best_score"],
            "selection_manifest_sha256": sha256_file(output / "selection/run_manifest.json"),
            "model_sha256": plan["model_sha256"],
            "initialization": plan["initialization"],
            "decision": "Refit same recipe from foundation initialization on all eligible data",
            "calibration": "none",
        }
        write_json(output / "selection-decision.json", decision)
        state.update(status="refitting", decision=decision)
        write_json(state_path, state)
        remaining = hours - used
        if remaining < 0.05:
            raise RuntimeError("Insufficient series allocation for final fit")
        final, used = supervise(output, "final", plan, epoch, remaining, ledger_path)
        state["runs"].append({"stage": "final", "hours": used})
        if (
            sources != source_hashes()
            or final["source_sha256"] != sources
            or not final.get("complete")
            or not final["final"]
            or final["completed_epochs"] != epoch
            or final["epoch_limit"] != epoch
            or final["model_config"] != model
            or final["initialization"] != plan["initialization"]
            or final["data_manifest_sha256"] != plan["data_manifest_sha256"]
            or final["training_records"] != plan["final_training_records"]
        ):
            raise ValueError("Final fit differs from frozen decision")
        verify_inputs(output, plan)
        export(output / "final", output / "data", raw, output / "predictions")
        state.update(
            status="prepared_not_submitted",
            selected_epoch=epoch,
            validation_score=record["best_score"],
            final_internal_score=None,
            predictions=str(output / "predictions/activity_submission.csv"),
        )
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        state["used_hours_completed_stages"] = sum(r["hours"] for r in state["runs"])
        state["accounting_note"] = "Ledger is authoritative and includes failed subprocesses"
        write_json(state_path, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ["source-fold", "output", "encoder"]:
        prep.add_argument("--" + name, type=Path, required=True)
    launch = commands.add_parser("run")
    for name in ["output", "raw", "ledger"]:
        launch.add_argument("--" + name, type=Path, required=True)
    launch.add_argument("--hours", type=float, required=True)
    status = commands.add_parser("status")
    status.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("_train")
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--stage", choices=["selection", "final"], required=True)
    worker.add_argument("--epochs", type=int, required=True)
    worker.add_argument("--hours", type=float, required=True)
    worker.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.source_fold, args.output, args.encoder)
    elif args.command == "run":
        result = run(args.output, args.raw, args.ledger, args.hours)
    elif args.command == "_train":
        from .reference_member import train

        plan = json.loads((args.output / "plan.json").read_text())
        verify_inputs(args.output, plan)
        model = json.loads((args.output / "model.json").read_text())
        if args.stage == "selection" and args.epochs != plan["selection_epochs"]:
            raise ValueError("Selection must use the fixed epoch limit")
        if args.stage == "final":
            selected = json.loads((args.output / "selection/run_manifest.json").read_text())
            if args.epochs != selected_epochs(selected, plan, model):
                raise ValueError("Final epochs differ from validation decision")
        result = train(
            args.output / "data",
            args.output / args.stage,
            epochs=args.epochs,
            hours=args.hours,
            campaign_ledger=args.ledger,
            model_config=model,
            encoder_checkpoint=plan["encoder_checkpoint"],
            final=args.stage == "final",
        )
    else:
        result = json.loads((args.output / "plan.json").read_text())
        if (args.output / "run.json").exists():
            result["run"] = json.loads((args.output / "run.json").read_text())
        for stage in ["selection", "final"]:
            path = args.output / stage / "run_manifest.json"
            if path.exists():
                record = json.loads(path.read_text())
                result[stage] = {
                    k: record.get(k)
                    for k in ["status", "completed_epochs", "best_epoch", "best_score"]
                }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
