"""Preserve the original four-hour pilot history alongside current supervision."""

import math
from pathlib import Path

from .budget import GPUBudget

CAP = 4.0


def is_pilot_charge(path):
    """Keep historical charges even when their recorded checkout path differs."""
    return any(
        part == "regression-frozen-representation-pilot-v1"
        or part.startswith("regression-atom-jepa-representations-v1")
        for part in Path(path).parts
    )


class PilotReservation(GPUBudget):
    """Atomic sub-budget reservation; unknown/interrupted usage stays charged."""

    def __init__(self, config):
        ledger = Path(config.get("ledger", "runs/regression-gpu-budget.json")).resolve()
        super().__init__(ledger, Path(config["output_dir"]), float(config["max_gpu_hours"]))
        self.root = str(ledger.parent / "regression-frozen-representation-pilot-v1") + "/"
        self.extraction = str(ledger.parent / "regression-atom-jepa-representations-v1")
        if not self.contains(self.run):
            raise ValueError("Frozen pilot output must remain in its budget-accounted namespace")
        if not math.isfinite(self.hours) or not 0 < self.hours <= CAP:
            raise ValueError("Invalid frozen pilot allocation")

    def contains(self, path):
        return path.startswith(self.root) or path.startswith(self.extraction)

    def _update(self, operation):
        def guarded(records):
            if self.run not in records:
                charges = [
                    record.get("used_hours", record["reserved_hours"])
                    for path, record in records.items()
                    if is_pilot_charge(path)
                ]
                if any(not math.isfinite(v) or v < 0 for v in charges):
                    raise ValueError("Invalid historical pilot charge")
                if sum(charges) + self.hours > CAP:
                    raise ValueError("Original four-GPU-hour pilot cap exceeded")
            operation(records)

        super()._update(guarded)
