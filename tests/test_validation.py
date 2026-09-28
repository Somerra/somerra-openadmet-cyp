from pathlib import Path

import pandas as pd

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS, TDI_ENDPOINTS
from somerra_openadmet.validation import validate_submission


def _test_frame() -> pd.DataFrame:
    return pd.DataFrame({"Molecule_Name": ["mol-1", "mol-2"], "SMILES": ["CCO", "c1ccccc1"]})


def test_valid_activity_submission(tmp_path: Path):
    frame = _test_frame()
    submission = frame.copy()
    for endpoint in ACTIVITY_ENDPOINTS:
        submission[endpoint] = [4.0, 5.0]
    path = tmp_path / "activity.csv"
    submission.to_csv(path, index=False)
    assert validate_submission(path, frame, "activity").valid


def test_tdi_rejects_invalid_boolean(tmp_path: Path):
    frame = _test_frame()
    submission = frame.copy()
    for endpoint in TDI_ENDPOINTS:
        submission[endpoint] = [True, "maybe"]
    path = tmp_path / "tdi.csv"
    submission.to_csv(path, index=False)
    report = validate_submission(path, frame, "tdi")
    assert not report.valid
    assert any("Unrecognized boolean" in error for error in report.errors)
