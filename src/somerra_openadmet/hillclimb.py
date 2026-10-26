"""Bounded regression campaign on a frozen molecular split; never uploads."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .data_sources.common import write_json
from .download import sha256_file

PRIOR_HOURS = 6.74626703225102
TOTAL_HOURS = 12.0
FINAL_RESERVE = 1.0
VARIANTS = {
    "reference": ("mae", 1.0),
    "interval": ("interval", 1.0),
    "aux-half": ("mae", 0.5),
    "interval-aux-half": ("interval", 0.5),
}


def qualifies(reference, candidate):
    if not candidate.get("complete"):
        return False
    base, new = reference["internal_test_metrics"], candidate["internal_test_metrics"]
    return new["MA-ST-RAE"] <= 0.97 * base["MA-ST-RAE"] and all(
        new["endpoints"][ep] - score <= 0.02 for ep, score in base["endpoints"].items()
    )


def run_campaign(bundle, output, raw):
    root = Path(__file__).resolve().parents[2]
    output.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "running",
        "prior_gpu_hours": PRIOR_HOURS,
        "gpu_limit_hours": TOTAL_HOURS,
        "used_hours": PRIOR_HOURS,
        "runs": [],
        "seed": 42,
        "epoch_limit": 100,
        "selection": "internal-test point MA-ST-RAE; >=3% improvement; endpoint regression <=0.02",
        "planned_variants": list(VARIANTS),
        "final_reserve_hours": FINAL_RESERVE,
        "source_hashes": {str(p): sha256_file(p) for p in sorted((root / "src").rglob("*.py"))},
        "bundle_sha256": sha256_file(bundle / "manifest.json"),
    }
    state["source_hashes"][str(root / "autoresearch.py")] = sha256_file(root / "autoresearch.py")
    state_path = output / "study.json"

    def publish():
        state["heartbeat_unix"] = time.time()
        write_json(state_path, state)

    def check_frozen():
        if sha256_file(bundle / "manifest.json") != state["bundle_sha256"] or any(
            sha256_file(Path(p)) != digest for p, digest in state["source_hashes"].items()
        ):
            raise ValueError("Frozen implementation or bundle changed")

    def run_many(specifications):
        """At most two independent jobs; charge summed process lifetimes conservatively."""
        check_frozen()
        if len(specifications) > 2:
            raise ValueError("At most two concurrent GPU jobs")
        if state["used_hours"] + sum(s["hours"] for s in specifications) > TOTAL_HOURS:
            raise ValueError("Campaign allocation exhausted")
        jobs = []
        results = {}
        try:
            for spec in specifications:
                name = spec["name"]
                dest = output / name
                variant = spec["variant"]
                loss, weight = VARIANTS[variant]
                command = [
                    sys.executable,
                    str(root / "autoresearch.py"),
                    "hillclimb",
                    "train",
                    "--bundle",
                    str(bundle),
                    "--output",
                    str(dest),
                    "--hours",
                    str(spec["hours"]),
                    "--epochs",
                    str(spec.get("epochs", 100)),
                    "--loss-kind",
                    loss,
                    "--auxiliary-weight",
                    str(weight),
                ]
                if spec.get("final"):
                    command.append("--final")
                if spec.get("smoke"):
                    command.append("--smoke")
                log = (output / (name + ".log")).open("w")
                started = time.monotonic()
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONPATH": str(root / "src")},
                    start_new_session=True,
                )
                entry = {
                    **spec,
                    "pid": process.pid,
                    "status": "running",
                    "path": str(dest),
                    "command": command,
                }
                entry["command"] = command
                state["runs"].append(entry)
                jobs.append((process, log, started, entry, dest))
                publish()
            while jobs:
                for job in jobs[:]:
                    process, log, started, entry, dest = job
                    elapsed = time.monotonic() - started
                    curve = dest / "curve.jsonl"
                    stale = time.time() - curve.stat().st_mtime if curve.exists() else elapsed
                    if process.poll() is None and (elapsed >= entry["hours"] * 3600 or stale > 600):
                        entry["stop_reason"] = (
                            "budget" if elapsed >= entry["hours"] * 3600 else "stalled"
                        )
                        os.killpg(process.pid, signal.SIGKILL)
                    if process.poll() is not None:
                        code = process.wait()
                        log.close()
                        used = (time.monotonic() - started) / 3600
                        state["used_hours"] += used
                        entry.update(
                            status="finished" if code == 0 else "failed",
                            used_hours=used,
                            returncode=code,
                        )
                        jobs.remove(job)
                        if code:
                            remaining = entry["hours"] - used
                            if (
                                not entry.get("retried")
                                and remaining > 0.1
                                and (dest / "recovery.json").exists()
                            ):
                                retry_name = entry["name"] + "-retry"
                                retry_dest = output / retry_name
                                command = entry["command"][:]
                                command[command.index("--output") + 1] = str(retry_dest)
                                command[command.index("--hours") + 1] = str(remaining)
                                command += ["--resume-from", str(dest)]
                                retry_log = (output / (retry_name + ".log")).open("w")
                                retry_start = time.monotonic()
                                retry_process = subprocess.Popen(
                                    command,
                                    stdout=retry_log,
                                    stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONPATH": str(root / "src")},
                                    start_new_session=True,
                                )
                                retry_entry = {
                                    **entry,
                                    "name": retry_name,
                                    "result_key": entry["name"],
                                    "retried": True,
                                    "hours": remaining,
                                    "pid": retry_process.pid,
                                    "status": "running",
                                    "path": str(retry_dest),
                                    "command": command,
                                }
                                for key in ["used_hours", "returncode", "stop_reason"]:
                                    retry_entry.pop(key, None)
                                state["runs"].append(retry_entry)
                                jobs.append(
                                    (retry_process, retry_log, retry_start, retry_entry, retry_dest)
                                )
                                continue
                            raise RuntimeError(
                                f"{entry['name']} failed; recovery checkpoint retained"
                            )
                        result = json.loads((dest / "run_manifest.json").read_text())
                        if not entry.get("smoke") and not result.get("complete"):
                            raise RuntimeError(f"{entry['name']} was censored; not eligible")
                        result["run_path"] = str(dest)
                        results[entry.get("result_key", entry["name"])] = result
                state["active_elapsed_hours"] = sum((time.monotonic() - j[2]) / 3600 for j in jobs)
                publish()
                if jobs:
                    time.sleep(15)
            return results
        finally:
            for process, log, started, entry, _dest in jobs:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                log.close()
                used = (time.monotonic() - started) / 3600
                state["used_hours"] += used
                entry.update(status="interrupted", used_hours=used)
            publish()

    publish()
    try:
        run_many(
            [dict(name="smoke", variant="interval-aux-half", epochs=1, hours=0.10, smoke=True)]
        )
        reference = run_many([dict(name="reference", variant="reference", hours=1.0)])["reference"]
        state["reference_score"] = reference["internal_test_metrics"]
        publish()
        contenders = run_many(
            [dict(name=v, variant=v, hours=1.0) for v in ("interval", "aux-half")]
        )
        accepted = {
            name: result for name, result in contenders.items() if qualifies(reference, result)
        }
        if len(accepted) == 2 and state["used_hours"] + 1 + FINAL_RESERVE <= TOTAL_HOURS:
            combined = run_many(
                [dict(name="interval-aux-half", variant="interval-aux-half", hours=1.0)]
            )
            for name, result in combined.items():
                if qualifies(reference, result):
                    accepted[name] = result
        else:
            state["combination_skipped"] = "Both single changes must meet the predeclared gate"
        selected = "reference"
        winner = reference
        if accepted:
            selected, winner = min(
                accepted.items(), key=lambda x: x[1]["internal_test_metrics"]["MA-ST-RAE"]
            )
        state.update(
            selected_variant=selected,
            selected_run=winner["run_path"],
            selected_metrics=winner["internal_test_metrics"],
            selected_bootstrap=winner["internal_test_bootstrap"],
        )
        publish()
        remaining = TOTAL_HOURS - state["used_hours"]
        if remaining < FINAL_RESERVE:
            raise RuntimeError("Insufficient reserved final-fit budget")
        final = run_many(
            [
                dict(
                    name="final",
                    variant=selected,
                    epochs=winner["best_epoch"],
                    hours=FINAL_RESERVE,
                    final=True,
                )
            ]
        )["final"]
        from .reference_member import export

        export(Path(final["run_path"]), bundle, raw, output / "predictions")
        state["status"] = "complete_predictions_prepared_not_submitted"
    except BaseException as exc:
        state.update(status="stopped", error=repr(exc))
        raise
    finally:
        publish()
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ["run", "train"]:
        p = subs.add_parser(name)
        p.add_argument("--bundle", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "run":
            p.add_argument("--raw", type=Path, required=True)
        else:
            p.add_argument("--hours", type=float, required=True)
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--loss-kind", choices=["mae", "interval"], default="mae")
            p.add_argument("--auxiliary-weight", type=float, default=1.0)
            p.add_argument("--resume-from", type=Path)
            p.add_argument("--final", action="store_true")
            p.add_argument("--smoke", action="store_true")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "run":
        return run_campaign(**args)
    from .reference_member import train

    return train(**args)
