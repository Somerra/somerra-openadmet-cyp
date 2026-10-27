"""Supervise one depth-six submission-selection fit and export its best checkpoint."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .campaign_budget import CampaignLedger
from .configuration import source_hashes
from .data_sources.common import write_json
from .model_recipe import resolve_model
from .reference_member_data import load_frames, submission_selection_bundle


def run(base_bundle, strategy, raw, output, ledger_path, hours):
    if not 0 < hours <= 0.4:
        raise ValueError("Submission fitting allocation must be at most 0.4 hours")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / "data"
    submission_selection_bundle(base_bundle, bundle)
    settings = resolve_model(json.loads(Path(strategy).read_text()))
    if settings["depth"] != 6:
        raise ValueError("This preparation requires the chosen depth-six recipe")
    recipe = output / "model.json"
    write_json(recipe, settings)
    pool, validation, _, manifest = load_frames(bundle)
    sources = source_hashes()
    state = {
        "status": "running",
        "bundle": str(bundle),
        "model_config": settings,
        "training_records": len(pool),
        "training_parents": pool.parent.nunique(),
        "validation_records": len(validation),
        "validation_parents": validation.parent.nunique(),
        "data_manifest": manifest,
        "source_sha256": sources,
        "runs": [],
        "open_source_code": False,
        "model_report_link": "",
    }
    write_json(output / "preparation.json", state)
    ledger = CampaignLedger(ledger_path)
    remaining, previous, fitted = hours, None, None
    try:
        for attempt in range(2):
            if sources != source_hashes():
                raise ValueError("Source changed during submission preparation")
            dest = output / ("fit" if attempt == 0 else "recovery")
            ledger.reserve_many([(dest, remaining)])
            command = [
                sys.executable,
                "-m",
                "somerra_openadmet.submission_selection",
                "_train",
                "--bundle",
                str(bundle),
                "--output",
                str(dest),
                "--strategy",
                str(recipe),
                "--ledger",
                str(ledger_path),
                "--hours",
                str(remaining),
            ]
            if previous:
                command += ["--resume-from", str(previous)]
            process, started = None, time.monotonic()
            try:
                with (output / (dest.name + ".log")).open("w") as log:
                    process = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    )
                    while process.poll() is None:
                        elapsed = time.monotonic() - started
                        curve = dest / "curve.jsonl"
                        stale = time.time() - curve.stat().st_mtime if curve.exists() else elapsed
                        if elapsed >= remaining * 3600 or stale > 600:
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
            state["runs"].append({"path": str(dest), "hours": used, "returncode": code})
            write_json(output / "preparation.json", state)
            remaining -= used
            if code == 0:
                fitted = dest
                break
            if not (dest / "recovery.json").exists() or remaining <= 0.05:
                raise RuntimeError("Submission fit failed; checkpoints retained")
            previous = dest
        if fitted is None:
            raise RuntimeError("Submission fit recovery failed")
        if sources != source_hashes():
            raise ValueError("Source changed during submission preparation")
        record = json.loads((fitted / "run_manifest.json").read_text())
        if record["model_config"] != settings or record["completed_epochs"] != 100:
            raise ValueError("Completed fit differs from selected recipe/epoch policy")
        from .reference_member import export

        export(fitted, bundle, raw, output / "predictions", selected_validation=True)
        state.update(
            status="prepared_not_submitted",
            selected_run=str(fitted),
            selected_epoch=record["best_epoch"],
            validation_score=record["best_score"],
        )
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(output / "preparation.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "_train"])
    for name in ["bundle", "output", "strategy", "ledger"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--hours", type=float, default=0.4)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.command == "_train":
        from .reference_member import train

        train(
            args.bundle,
            args.output,
            epochs=100,
            hours=args.hours,
            campaign_ledger=args.ledger,
            model_config=json.loads(args.strategy.read_text()),
            resume_from=args.resume_from,
        )
    else:
        if args.raw is None:
            parser.error("prepare requires --raw")
        run(args.bundle, args.strategy, args.raw, args.output, args.ledger, args.hours)


if __name__ == "__main__":
    main()
