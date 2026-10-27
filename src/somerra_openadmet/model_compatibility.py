"""Recorded fitted identity and explicit checkpoint inference compatibility."""

import json
import math
from pathlib import Path

from .constants import ACTIVITY_ENDPOINTS
from .download import sha256_file

IMPLEMENTATION_FILES = [
    "src/somerra_openadmet/hillclimb_model.py",
    "src/somerra_openadmet/model_recipe.py",
    # Legacy references built their architecture here; this also owns training.
    "src/somerra_openadmet/reference_member.py",
]
PARITY_TOLERANCE = 1e-4


def fitted_identity(record):
    source = record.get("source_sha256")
    if not isinstance(source, dict) or not source:
        raise ValueError("Fitted run has no recorded source identity")
    # Preserve missing legacy modules as missing, never substitute today's files.
    return {
        path: source.get(str(Path(path).relative_to("src/somerra_openadmet")))
        for path in IMPLEMENTATION_FILES
    }


def compatibility_binding(entry, bundle, current, runtime):
    return {
        "fitted_source_sha256": entry["model_source_sha256"],
        "compatible_source_sha256": current,
        "artifacts": entry["artifacts"],
        "model_config": entry["model_config"],
        "bundle_sha256": sha256_file(Path(bundle) / "manifest.json"),
        "runtime": runtime,
    }


def verify_prediction_parity(truth, expected, actual):
    import numpy as np

    keys = ["Molecule_Name", "parent"]
    if (
        truth.empty
        or truth.Molecule_Name.duplicated().any()
        or not truth[keys].equals(expected[keys])
    ):
        raise ValueError("Compatibility prediction identities/order mismatch")
    if list(actual.columns) != list(ACTIVITY_ENDPOINTS):
        raise ValueError("Compatibility endpoint order mismatch")
    observed = actual.to_numpy()
    saved = expected[list(ACTIVITY_ENDPOINTS)].to_numpy()
    if (
        observed.shape != saved.shape
        or not np.isfinite(observed).all()
        or not np.isfinite(saved).all()
    ):
        raise ValueError("Invalid compatibility predictions")
    error = float(np.max(np.abs(observed - saved)))
    if error > PARITY_TOLERANCE:
        raise ValueError(f"Model compatibility prediction parity failed: {error}")
    return {"rows": len(truth), "max_abs_error": error, "tolerance": PARITY_TOLERANCE}


def check_compatibility(entry, bundle, current, runtime):
    import pandas as pd
    import torch

    from .reference_member import load_model, predict, prediction_loader
    from .reference_member_data import load_frames

    bundle, run = Path(bundle), Path(entry["run"])
    record = json.loads((run / "run_manifest.json").read_text())
    if record["data_manifest_sha256"] != sha256_file(bundle / "manifest.json"):
        raise ValueError("Compatibility requires the fitted data bundle")
    load_frames(bundle)
    truth = pd.read_csv(bundle / "official.csv")
    truth = truth.loc[truth.partition == "internal_test"].reset_index(drop=True)
    expected = pd.read_csv(run / "internal-test.csv")
    model = load_model(run).to("cuda" if torch.cuda.is_available() else "cpu")
    actual = predict(model, prediction_loader(truth.parent), record["tasks"])
    parity = verify_prediction_parity(truth, expected, actual)
    return {
        **compatibility_binding(entry, bundle, current, runtime),
        **parity,
        "kind": "checkpoint_internal_test_prediction_parity_v1",
        "scope": "Checkpoint inference compatibility; not proof of equivalent retraining",
    }


def require_compatible(entry, bundle, current, runtime):
    if entry["model_source_sha256"] == current:
        return
    evidence = entry.get("implementation_compatibility", {})
    expected = compatibility_binding(entry, bundle, current, runtime)
    error = evidence.get("max_abs_error")
    if (
        any(evidence.get(key) != value for key, value in expected.items())
        or evidence.get("kind") != "checkpoint_internal_test_prediction_parity_v1"
        or evidence.get("tolerance") != PARITY_TOLERANCE
        or not isinstance(error, (int, float))
        or not math.isfinite(error)
        or not 0 <= error <= PARITY_TOLERANCE
        or not isinstance(evidence.get("rows"), int)
        or evidence["rows"] <= 0
    ):
        raise ValueError(
            "Model implementation differs from fitted source; restore it or initialize a new "
            "task with --check-model-compatibility"
        )
