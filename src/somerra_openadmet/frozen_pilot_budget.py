"""Preserve the original four-hour pilot history alongside current supervision."""

import fcntl
import json
import math
from pathlib import Path

from .budget import GPUBudget
from .campaign_budget import CampaignLedger
from .download import sha256_file

CAP = 4.0


def pilot_ledger_path(config):
    """Read the immutable history binding in the shared campaign, never the cwd."""
    from .representation_budget import ledger_path

    campaign = CampaignLedger(ledger_path(config))
    with campaign.locked():
        binding = campaign._read().get("frozen_pilot_history")
    if not binding:
        raise ValueError("Bind the existing pilot history to the shared campaign before launching")
    path = Path(binding["ledger"])
    if not path.is_absolute():
        raise ValueError("Shared pilot history binding must be absolute")
    configured = config.get("ledger")
    if configured is not None:
        supplied = Path(configured)
        if not supplied.is_absolute() or supplied.resolve() != path.resolve():
            raise ValueError("Pilot ledger must match the absolute history bound to the campaign")
    if not path.is_file():
        raise ValueError("Restore the existing pilot history bound to the shared campaign")
    return path.resolve()


def bind_pilot_history(config, history):
    """Record the existing ledger's location once, without copying/resetting charges."""
    from .representation_budget import ledger_path

    path = Path(history)
    if not path.is_absolute():
        raise ValueError("Pilot history must be an absolute existing path")
    path = path.resolve()
    campaign = CampaignLedger(ledger_path(config))
    with campaign.locked():
        state = campaign._read()
        binding = state.get("frozen_pilot_history")
        if binding:
            if binding["ledger"] != str(path):
                raise ValueError("Pilot history already bound; refusing to redirect spending")
            return
        with path.open() as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            records = json.load(handle)
            charges = [
                record.get("used_hours", record["reserved_hours"])
                for name, record in records.items()
                if is_pilot_charge(name)
            ]
            if not charges or any(not math.isfinite(v) or v < 0 for v in charges):
                raise ValueError("Existing pilot history is missing or invalid")
            state["frozen_pilot_history"] = {
                "ledger": str(path),
                "bound_sha256": sha256_file(path),
                "bound_committed_hours": sum(charges),
            }
            campaign._write(state)


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
        ledger = pilot_ledger_path(config)
        super().__init__(ledger, Path(config["output_dir"]), float(config["max_gpu_hours"]))
        if not self.contains(self.run):
            raise ValueError("Frozen pilot output must remain in its budget-accounted namespace")
        if not math.isfinite(self.hours) or not 0 < self.hours <= CAP:
            raise ValueError("Invalid frozen pilot allocation")

    def contains(self, path):
        return is_pilot_charge(path)

    def _update(self, operation):
        def guarded(records):
            if not any(is_pilot_charge(path) for path in records):
                raise ValueError("Existing pilot history is missing; refusing to reset spending")
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

        # Unlike the legacy writer, never create an empty ledger if history is lost.
        with self.path.open("r+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            records = json.load(handle)
            guarded(records)
            handle.seek(0)
            handle.truncate()
            json.dump(records, handle, indent=2, allow_nan=False)
            handle.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bind existing pilot history; allocate no GPU time"
    )
    parser.add_argument("--campaign-ledger", required=True)
    parser.add_argument("--bind-history", required=True)
    args = parser.parse_args()
    bind_pilot_history({"campaign_ledger": args.campaign_ledger}, args.bind_history)
