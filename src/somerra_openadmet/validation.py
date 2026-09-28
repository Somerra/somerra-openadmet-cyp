"""Strict local validation for activity and TDI submission files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ACTIVITY_ENDPOINTS, IDENTIFIER_COLUMNS, TDI_ENDPOINTS
from .data import coerce_boolean_labels


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    track: str
    rows: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_submission(
    submission_path: Path,
    test_frame: pd.DataFrame,
    track: str,
) -> ValidationReport:
    path = Path(submission_path)
    if not path.is_file():
        return ValidationReport(False, track, 0, (f"File does not exist: {path}",), ())
    try:
        submission = pd.read_csv(path)
    except Exception as exc:
        return ValidationReport(False, track, 0, (f"Could not read CSV: {exc}",), ())

    endpoints = ACTIVITY_ENDPOINTS if track == "activity" else TDI_ENDPOINTS
    required = (*IDENTIFIER_COLUMNS, *endpoints)
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(set(required) - set(submission.columns))
    if missing:
        return ValidationReport(False, track, len(submission), (f"Missing columns: {missing}",), ())

    if len(submission) != len(test_frame):
        errors.append(f"Expected {len(test_frame)} rows, found {len(submission)}")
    if submission[list(IDENTIFIER_COLUMNS)].isna().any(axis=None):
        errors.append("Identifier columns contain missing values")
    if submission["Molecule_Name"].duplicated().any():
        errors.append("Molecule_Name contains duplicates")

    expected = test_frame.set_index("Molecule_Name")["SMILES"]
    observed = submission.set_index("Molecule_Name")["SMILES"]
    missing_ids = sorted(set(expected.index) - set(observed.index))
    extra_ids = sorted(set(observed.index) - set(expected.index))
    if missing_ids:
        errors.append(f"Missing {len(missing_ids)} expected Molecule_Name value(s)")
    if extra_ids:
        errors.append(f"Found {len(extra_ids)} unexpected Molecule_Name value(s)")
    shared = expected.index.intersection(observed.index)
    mismatched_smiles = int((expected.loc[shared] != observed.loc[shared]).sum())
    if mismatched_smiles:
        errors.append(f"SMILES mismatch for {mismatched_smiles} molecule(s)")
    if list(submission["Molecule_Name"]) != list(test_frame["Molecule_Name"]):
        warnings.append("Row order differs from the official blinded test file")

    if track == "activity":
        for endpoint in endpoints:
            values = pd.to_numeric(submission[endpoint], errors="coerce").to_numpy()
            invalid = int((~np.isfinite(values)).sum())
            if invalid:
                errors.append(f"{endpoint} contains {invalid} missing/non-finite value(s)")
            if np.isfinite(values).any() and float(np.nanstd(values)) == 0.0:
                warnings.append(f"{endpoint} is constant")
    elif track == "tdi":
        for endpoint in endpoints:
            try:
                values = coerce_boolean_labels(submission[endpoint])
            except ValueError as exc:
                errors.append(f"{endpoint}: {exc}")
                continue
            if values.isna().any():
                errors.append(f"{endpoint} contains missing values")
            elif values.nunique() < 2:
                warnings.append(f"{endpoint} predicts only one class")
    else:
        errors.append(f"Unknown track: {track!r}")

    return ValidationReport(not errors, track, len(submission), tuple(errors), tuple(warnings))
