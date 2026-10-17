"""Read-only summaries of persisted campaign artifacts; no process-state guesses."""

import json
from pathlib import Path


def status(runs_dir: Path) -> dict:
    rows = []
    ledger_path = runs_dir / "regression-gpu-budget.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    accounted = set()
    used = 0.0
    reserved = 0.0
    for name, record in ledger.items():
        accounted.add(str(Path(name).resolve()))
        if "used_hours" in record:
            used += record["used_hours"]
        else:
            reserved += record["reserved_hours"]
    for directory in sorted(runs_dir.glob("regression-*")):
        if not directory.is_dir():
            continue
        metrics_path = directory / "metrics.json"
        manifest_path = directory / "run_manifest.json"
        if not manifest_path.exists():
            manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if str(directory.resolve()) not in accounted:
            used += float(manifest.get("gpu_hours", 0))
        if (
            not metrics_path.exists()
            and not (directory / "oof_predictions.csv").exists()
            and "selected_folds" not in manifest
        ):
            continue
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        score = metrics.get("MA-ST-RAE", metrics.get("macro", {}).get("nested_ensemble"))
        rows.append(
            {
                "run": str(directory),
                "artifact_state": "complete" if score is not None and manifest else "partial",
                "MA-ST-RAE": score,
                "recorded_gpu_hours": manifest.get("gpu_hours"),
                "gpu_reservation": ledger.get(str(directory.resolve())),
                "selected_folds": manifest.get("selected_folds"),
                "completed_folds": manifest.get("completed_folds"),
                "requested_folds_complete": manifest.get("requested_folds_complete"),
            }
        )
    return {
        "runs": rows,
        "recorded_gpu_hours": used,
        "outstanding_reserved_gpu_hours": reserved,
        "note": "Artifact state only; a reservation does not prove the training process is alive.",
    }
