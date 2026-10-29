"""Bounded concurrent orchestration of unchanged briford fold workers."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .briford import MEMBERS, check_contract, export, read, verify_complete, write
from .download import sha256_file


def worker_command(root, member, fold, seconds):
    return [
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
        str(seconds),
    ]


def execute(command, log_path, seconds):
    """Bound wall time and clean up the worker's dataloader processes too."""
    started = time.monotonic()
    with log_path.open("a") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            try:
                code = process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                code = 124
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return {"returncode": code, "used_seconds": time.monotonic() - started}


def wave(root, jobs, series, command_factory=worker_command):
    # Sum of worker wall times is conservative on one shared GPU. Reserving disjoint
    # portions prevents concurrent jobs from each spending the whole remaining budget.
    remaining = series["limit_seconds"] - series["used_seconds"]
    reservation = remaining / len(jobs)
    active = [
        {
            "member": member,
            "fold": fold,
            "reserved_seconds": reservation,
            "started_unix": time.time(),
        }
        for member, fold in jobs
    ]
    series.update(status="running", active_parallel=active)
    write(root / "series.json", series)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {}
        for record in active:
            member, fold = record["member"], record["fold"]
            path = root / "fits" / member / str(fold)
            path.mkdir(parents=True, exist_ok=True)
            future = pool.submit(
                execute,
                command_factory(root, member, fold, reservation),
                path / "fit.log",
                reservation,
            )
            futures[future] = record
        failed = False
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                # Conservatively retain the reservation if no reliable elapsed time exists.
                result = {
                    "returncode": 125,
                    "used_seconds": record["reserved_seconds"],
                    "error": str(exc),
                }
            series["used_seconds"] += result["used_seconds"]
            series["attempts"].append({**record, **result, "execution": "parallel"})
            series["active_parallel"] = [r for r in series["active_parallel"] if r != record]
            failed = failed or result["returncode"] != 0
            series["status"] = "failed" if failed else "running"
            write(root / "series.json", series)
    series.pop("active_parallel")
    write(root / "series.json", series)
    return not failed


def run(root, workers=2):
    root = Path(root).resolve()
    if workers not in (1, 2, 3, 4):
        raise ValueError("Concurrency must be between one and four")
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_contract(root)
        series = read(root / "series.json")
        if series.get("active"):
            raise ValueError("Settle the serial worker before transferring supervision")
        for record in series.pop("active_parallel", []):
            series["used_seconds"] += record["reserved_seconds"]
            series["attempts"].append({**record, "status": "interrupted_reservation_charged"})
        series["concurrency"] = workers
        series["accounting"] = "sum_of_worker_wall_seconds_on_one_shared_gpu"
        series["orchestrator_sha256"] = sha256_file(Path(__file__))
        write(root / "series.json", series)
        jobs = []
        for member in MEMBERS:
            for fold in range(5):
                if (root / "fits" / member / str(fold) / "complete.json").exists():
                    verify_complete(root, member, fold)
                else:
                    jobs.append((member, fold))
        while jobs:
            check_contract(root)
            batch = jobs[:workers]
            if (series["limit_seconds"] - series["used_seconds"]) / len(batch) < 90:
                series["status"] = "budget_exhausted"
                write(root / "series.json", series)
                return series
            if not wave(root, batch, series):
                raise RuntimeError(
                    "A fold failed; preserved completed artifacts and stopped dispatch"
                )
            for member, fold in batch:
                verify_complete(root, member, fold)
            jobs = jobs[len(batch) :]
        series["status"] = "training_complete"
        write(root / "series.json", series)
        try:
            result = export(root)
        except Exception as exc:
            series.update(status="failed", error=f"Prediction export failed: {exc}")
            write(root / "series.json", series)
            raise
        series.update(
            status="predictions_prepared_not_submitted",
            submission_sha256=result["submission_sha256"],
        )
        write(root / "series.json", series)
        return series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.workers)


if __name__ == "__main__":
    main()
