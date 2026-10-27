"""Persistent reservations for one explicitly initialized 12-GPU-hour allocation."""

import fcntl
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

from .download import sha256_file

LIMIT_HOURS = 12.0


class CampaignLedger:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ValueError("Campaign ledger must be an absolute shared path")
        self.path = self.path.resolve()

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _write(self, state):
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w") as handle:
            json.dump(state, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    def _read(self):
        if not self.path.exists():
            raise ValueError("Initialize the campaign ledger from the completed study first")
        state = json.loads(self.path.read_text())
        if state["version"] != 1 or state["limit_hours"] != LIMIT_HOURS:
            raise ValueError("Campaign allocation contract mismatch")
        charges = [state["historical_hours"]] + [
            r.get("used_hours", r["reserved_hours"]) for r in state["reservations"].values()
        ]
        if any(not math.isfinite(v) or v < 0 for v in charges):
            raise ValueError("Invalid campaign charge")
        return state

    def import_study(self, study_path):
        """One-time migration; never reset an existing allocation."""
        study_path = Path(study_path).resolve()
        study = json.loads(study_path.read_text())
        hours = study["used_hours"]
        if (
            study["status"] != "complete_predictions_prepared_not_submitted"
            or study["gpu_limit_hours"] != LIMIT_HOURS
            or not math.isfinite(hours)
            or not 0 <= hours <= LIMIT_HOURS
            or any(r["status"] == "running" for r in study["runs"])
        ):
            raise ValueError("Import requires a completed finite campaign accounting snapshot")
        with self.locked():
            if self.path.exists():
                raise FileExistsError("Campaign ledger already exists; refusing to reset spending")
            self._write(
                {
                    "version": 1,
                    "limit_hours": LIMIT_HOURS,
                    "historical_hours": hours,
                    "imported_study": str(study_path),
                    "imported_study_sha256": sha256_file(study_path),
                    "reservations": {},
                }
            )
        return self.snapshot()

    @staticmethod
    def _totals(state):
        records = state["reservations"].values()
        used = state["historical_hours"] + sum(r.get("used_hours", 0) for r in records)
        reserved = sum(r["reserved_hours"] for r in records if "used_hours" not in r)
        return {"used_hours": used, "reserved_hours": reserved, "committed_hours": used + reserved}

    def snapshot(self):
        with self.locked():
            return self._totals(self._read())

    def reserve_many(self, allocations):
        """Reserve the whole launch group atomically, before any subprocess starts."""
        allocations = [(str(Path(run).resolve()), hours) for run, hours in allocations]
        if len({run for run, _ in allocations}) != len(allocations) or any(
            not math.isfinite(hours) or hours <= 0 for _, hours in allocations
        ):
            raise ValueError("Invalid campaign reservations")
        with self.locked():
            state = self._read()
            if any(run in state["reservations"] for run, _ in allocations):
                raise ValueError("Run already reserved")
            if (
                self._totals(state)["committed_hours"] + sum(h for _, h in allocations)
                > LIMIT_HOURS
            ):
                raise ValueError("Campaign allocation exceeds 12 hours")
            for run, hours in allocations:
                state["reservations"][run] = {"reserved_hours": hours, "status": "reserved"}
            self._write(state)

    def claim_worker(self, run, hours):
        """A training worker must own one unclaimed coordinator reservation."""
        with self.locked():
            state = self._read()
            record = state["reservations"].get(str(Path(run).resolve()))
            if (
                record is None
                or record["status"] != "reserved"
                or record["reserved_hours"] != hours
            ):
                raise ValueError("Training requires an unclaimed matching campaign reservation")
            record.update(status="running", worker_pid=os.getpid())
            self._write(state)

    def settle(self, run, hours, status):
        """Only the coordinator releases a reservation after the worker has exited."""
        if not math.isfinite(hours) or hours < 0:
            raise ValueError("Invalid elapsed charge")
        with self.locked():
            state = self._read()
            record = state["reservations"][str(Path(run).resolve())]
            if "used_hours" in record:
                raise ValueError("Run already settled")
            record.update(used_hours=hours, status=status)
            self._write(state)


class WorkerBudget:
    """Measure worker time without releasing the coordinator's full-process reservation."""

    def __init__(self, ledger, run, hours):
        self.ledger, self.run, self.hours = CampaignLedger(ledger), run, hours

    def reserve(self):
        self.ledger.claim_worker(self.run, self.hours)
        self.started = time.monotonic()

    def finish(self):
        return (time.monotonic() - self.started) / 3600
