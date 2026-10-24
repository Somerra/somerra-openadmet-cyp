import json

import numpy as np
import pandas as pd
import pytest

from somerra_openadmet import reference
from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.download import sha256_file


def predictions():
    return pd.DataFrame(
        {
            "SMILES": ["C", "CC"],
            "Molecule_Name": ["a", "b"],
            **{ep: [4.0, 5.0] for ep in ACTIVITY_ENDPOINTS},
        }
    )


def test_rebuild_accepts_numeric_drift_but_not_changed_predictions():
    expected, actual = predictions(), predictions()
    actual[ACTIVITY_ENDPOINTS[0]] += 5e-6
    result = reference.prediction_agreement(expected, actual)
    assert result["predictions"] == 8
    assert result["max_absolute_difference"] == pytest.approx(5e-6)
    actual[ACTIVITY_ENDPOINTS[1]] += 0.01
    with pytest.raises(ValueError, match="tolerance"):
        reference.prediction_agreement(expected, actual)


@pytest.mark.parametrize("change", ["order", "nan", "duplicate", "smiles", "missing"])
def test_rebuild_rejects_invalid_coverage(change):
    actual = predictions()
    if change == "order":
        actual = actual.iloc[::-1].reset_index(drop=True)
    elif change == "nan":
        actual.loc[0, ACTIVITY_ENDPOINTS[0]] = np.nan
    elif change == "duplicate":
        actual.loc[1, "Molecule_Name"] = "a"
    elif change == "smiles":
        actual.loc[0, "SMILES"] = "N"
    else:
        actual = actual.iloc[:1]
    with pytest.raises(ValueError):
        reference.prediction_agreement(predictions(), actual)


def test_reference_refuses_existing_output_before_execution(tmp_path):
    with pytest.raises(FileExistsError):
        reference.reproduce(tmp_path / "source", tmp_path)


def test_reference_refuses_unpinned_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(reference.subprocess, "check_output", lambda *a, **k: "changed")
    with pytest.raises(ValueError, match="Expected reference commit"):
        reference.reproduce(tmp_path / "source", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_reference_records_timeout_without_completion(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "data").mkdir()
    script = source / "scripts/build_r25.py"
    script.write_text("# fixture")
    (source / "data/MANIFEST.json").write_text("{}")
    monkeypatch.setattr(reference, "BUILD_SHA256", sha256_file(script))
    monkeypatch.setattr(
        reference.subprocess, "check_output", lambda *a, **k: reference.REFERENCE_COMMIT
    )

    def run(command, **kwargs):
        if command[0] != "git":
            raise reference.subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(reference.subprocess, "run", run)
    output = tmp_path / "output"
    with pytest.raises(reference.subprocess.TimeoutExpired):
        reference.reproduce(source, output, timeout_seconds=1)
    record = json.loads((output / "manifest.json").read_text())
    assert record["status"] == "failed"
    assert record["error_type"] == "TimeoutExpired"
    assert record["independent_local_score"] is None
    assert "score" not in record
