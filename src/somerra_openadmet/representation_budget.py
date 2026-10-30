"""Supervise representation workers against the existing shared campaign allocation."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .campaign_budget import CampaignLedger


def ledger_path(config):
    value = (
        os.environ.get("SOMERRA_CAMPAIGN_LEDGER")
        or os.environ.get("SOMERRA_GPU_LEDGER")
        or config.get("campaign_ledger")
    )
    if not value:
        raise ValueError("Set SOMERRA_CAMPAIGN_LEDGER to the initialized shared 12-hour ledger")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Shared campaign ledger path must be absolute")
    CampaignLedger(path).snapshot()  # Reject legacy ledgers and never initialize/reset here.
    return path.resolve()


def maybe_supervise(operation, config_path):
    """Return a completed child result for GPU calls; None means CPU/claimed worker."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    device = config.get("device", "auto")
    if device == "cpu":
        return None
    if device == "auto":
        import torch

        if not torch.cuda.is_available():
            return None
    output = Path(config["output_dir"]).resolve()
    if os.environ.get("SOMERRA_REPRESENTATION_WORKER") == str(output):
        return None
    if output.exists():
        raise FileExistsError(output)
    hours = float(config["max_gpu_hours"])
    ledger = CampaignLedger(ledger_path(config))
    output.parent.mkdir(parents=True, exist_ok=True)
    pilot_allocation = None
    if operation in {
        "frozen-pilot",
        "atom-pilot-cache",
        "conformer-cache",
        "conformer-fit",
        "frozen-probe",
    }:
        from .frozen_pilot_budget import PilotReservation

        pilot_allocation = PilotReservation(config)
        pilot_allocation.reserve()
    try:
        ledger.reserve_many([(output, hours)])
    except BaseException:
        if pilot_allocation is not None:
            pilot_allocation.finish()
        raise
    start, process = time.monotonic(), None
    command = [sys.executable, "-m", __name__, operation, str(config_path)]
    try:
        with output.with_suffix(".worker.log").open("x") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ,
                    "SOMERRA_CAMPAIGN_LEDGER": str(ledger.path),
                    "SOMERRA_REPRESENTATION_WORKER": str(output),
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                },
            )
            try:
                code = process.wait(timeout=max(0.01, hours * 3600 - (time.monotonic() - start)))
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    "Representation worker exhausted its full-process allocation"
                ) from exc
        if code:
            raise RuntimeError(
                f"Representation worker failed ({code}); see {output.with_suffix('.worker.log')}"
            )
        return json.loads((output / "worker-result.json").read_text())
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        spent = (time.monotonic() - start) / 3600 if process is not None else 0.0
        ledger.settle(output, spent, "exited" if process is not None else "not_started")
        if pilot_allocation is not None:
            pilot_allocation.finish()
        evidence = {
            "campaign_ledger": str(ledger.path),
            "charged_hours": spent,
            "reserved_hours": hours,
            "returncode": None if process is None else process.returncode,
            "scope": "full child lifetime; reservation settled after worker exit",
        }
        output.with_suffix(".accounting.json").write_text(json.dumps(evidence, indent=2) + "\n")


def main():
    from .atom_jepa import cache_representations as atom_cache
    from .conformer_sensitivity import cache_stage, fit_stage
    from .frozen_benchmark import screen as frozen_screen
    from .frozen_probe import screen as probe_screen
    from .representation_training import (
        predict_representations,
        profile_representations,
        screen_representations,
    )
    from .representations import atomic_json, cache_representations

    operation, config_path = sys.argv[1:]
    handlers = {
        "cache": cache_representations,
        "screen": screen_representations,
        "predict": predict_representations,
        "profile": profile_representations,
        "frozen-pilot": frozen_screen,
        "frozen-probe": probe_screen,
        "atom-pilot-cache": atom_cache,
        "conformer-cache": cache_stage,
        "conformer-fit": fit_stage,
    }
    config = json.loads(Path(config_path).read_text())
    if os.environ.get("SOMERRA_REPRESENTATION_WORKER") != str(Path(config["output_dir"]).resolve()):
        raise ValueError("Workers must be launched by the campaign supervisor")
    result = handlers[operation](Path(config_path))
    atomic_json(Path(config["output_dir"]) / "worker-result.json", result)


if __name__ == "__main__":
    main()
