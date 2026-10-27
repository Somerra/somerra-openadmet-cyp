from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.model_compatibility import (
    compatibility_binding,
    require_compatible,
    verify_prediction_parity,
)


def test_prediction_compatibility_checks_values_and_identities():
    truth = pd.DataFrame({"Molecule_Name": ["a", "b"], "parent": ["C", "CC"]})
    actual = pd.DataFrame(np.ones((2, 4)), columns=ACTIVITY_ENDPOINTS)
    saved = pd.concat([truth, actual], axis=1)
    assert verify_prediction_parity(truth, saved, actual)["max_abs_error"] == 0
    with pytest.raises(ValueError, match="parity failed"):
        verify_prediction_parity(truth, saved, actual + 0.001)
    with pytest.raises(ValueError, match="identities/order"):
        verify_prediction_parity(truth, saved.iloc[::-1].reset_index(drop=True), actual)
    actual.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid compatibility"):
        verify_prediction_parity(truth, saved, actual)


@pytest.mark.parametrize("changed", ["source", "artifacts", "bundle", "runtime", "recipe", "error"])
def test_compatibility_evidence_is_bound_to_exact_inputs(tmp_path, changed):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    entry = {
        "model_source_sha256": {"model.py": "old"},
        "artifacts": {"best.ckpt": "checkpoint"},
        "model_config": {"depth": 4},
    }
    current, runtime = {"model.py": "new"}, {"torch": "version"}
    entry["implementation_compatibility"] = deepcopy(
        {
            **compatibility_binding(entry, tmp_path, current, runtime),
            "kind": "checkpoint_internal_test_prediction_parity_v1",
            "rows": 981,
            "max_abs_error": 1e-6,
            "tolerance": 1e-4,
        }
    )
    require_compatible(entry, tmp_path, current, runtime)
    if changed == "source":
        current["model.py"] = "newer"
    elif changed == "artifacts":
        entry["artifacts"]["best.ckpt"] = "different"
    elif changed == "bundle":
        manifest.write_text('{"changed": true}')
    elif changed == "runtime":
        runtime["torch"] = "different"
    elif changed == "recipe":
        entry["model_config"]["depth"] = 3
    else:
        entry["implementation_compatibility"]["max_abs_error"] = float("nan")
    with pytest.raises(ValueError, match="check-model-compatibility"):
        require_compatible(entry, tmp_path, current, runtime)


def test_exact_fitted_implementation_needs_no_compatibility_record(tmp_path):
    require_compatible(
        {"model_source_sha256": {"model.py": "same"}}, tmp_path, {"model.py": "same"}, {}
    )
