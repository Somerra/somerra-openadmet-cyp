"""File-locked GPU time reservations shared by regression experiments."""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path


class GPUBudget:
    def __init__(self, path: Path, run: Path, hours: float, limit: float = 72):
        self.path, self.run, self.hours, self.limit = path, str(run.resolve()), hours, limit
        self.started = None

    def _update(self, operation):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            text = handle.read()
            records = json.loads(text) if text else {}
            operation(records)
            handle.seek(0)
            handle.truncate()
            json.dump(records, handle, indent=2)
            handle.flush()

    def reserve(self):
        def update(records):
            if self.run in records:
                raise ValueError("This run already has a GPU reservation")
            committed = sum(r.get("used_hours", r["reserved_hours"]) for r in records.values())
            # Account for completed earlier runs, including those created before the ledger.
            paths = list(self.path.parent.glob("regression-*/run_manifest.json"))
            paths += [
                p
                for p in self.path.parent.glob("regression-*/manifest.json")
                if not (p.parent / "run_manifest.json").exists()
            ]
            for path in paths:
                if str(path.parent.resolve()) not in records:
                    record = json.loads(path.read_text())
                    committed += float(record.get("gpu_hours", 0))
            if self.hours <= 0 or committed + self.hours > self.limit:
                raise ValueError(f"GPU allocation exceeds {self.limit} hour campaign limit")
            records[self.run] = {"reserved_hours": self.hours, "status": "running"}

        self._update(update)
        self.started = time.monotonic()

    def finish(self):
        used = (time.monotonic() - self.started) / 3600

        def update(records):
            records[self.run].update(used_hours=used, status="finished")

        self._update(update)
        return used

    def record_interruption(self, evidence: str):
        """After external process verification, retain the full unknown-time charge."""
        if not evidence.strip():
            raise ValueError("Interruption evidence is required")

        def update(records):
            record = records[self.run]
            if "used_hours" in record:
                raise ValueError("Cannot mark a finalized allocation interrupted")
            record.update(status="interrupted_unknown_usage", interruption_evidence=evidence)

        self._update(update)
