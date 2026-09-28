"""Data loading and integrity checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .constants import (
    ACTIVITY_ENDPOINTS,
    ACTIVITY_TRAIN_FILE,
    EMAX_TRAIN_FILE,
    EXPECTED_TEST_ROWS,
    IDENTIFIER_COLUMNS,
    SINGLE_CONCENTRATION_FILE,
    TDI_ENDPOINTS,
    TDI_TRAIN_FILE,
    TEST_FILE,
    lower_bound_column,
    upper_bound_column,
)


@dataclass(frozen=True)
class DataAudit:
    activity_rows: int
    tdi_rows: int
    emax_rows: int
    single_concentration_rows: int
    test_rows: int
    activity_label_counts: dict[str, int]
    tdi_label_counts: dict[str, int]
    tdi_positive_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing challenge file: {path}")
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def coerce_boolean_labels(series: pd.Series) -> pd.Series:
    """Normalize common CSV boolean encodings while preserving missing values."""
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
    }

    def convert(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        key = str(value).strip().lower()
        if key not in mapping:
            raise ValueError(f"Unrecognized boolean label: {value!r}")
        return mapping[key]

    return series.map(convert).astype("boolean")


def load_activity(data_dir: Path) -> pd.DataFrame:
    df = _read_csv(Path(data_dir) / ACTIVITY_TRAIN_FILE)
    required = (*IDENTIFIER_COLUMNS, *ACTIVITY_ENDPOINTS)
    bounds = tuple(
        column
        for endpoint in ACTIVITY_ENDPOINTS
        for column in (lower_bound_column(endpoint), upper_bound_column(endpoint))
    )
    _require_columns(df, (*required, *bounds), "activity training data")
    return df


def load_tdi(data_dir: Path) -> pd.DataFrame:
    df = _read_csv(Path(data_dir) / TDI_TRAIN_FILE)
    _require_columns(df, (*IDENTIFIER_COLUMNS, *TDI_ENDPOINTS), "TDI training data")
    for endpoint in TDI_ENDPOINTS:
        df[endpoint] = coerce_boolean_labels(df[endpoint])
    return df


def load_test(data_dir: Path) -> pd.DataFrame:
    df = _read_csv(Path(data_dir) / TEST_FILE)
    _require_columns(df, IDENTIFIER_COLUMNS, "blinded test data")
    if len(df) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Expected {EXPECTED_TEST_ROWS} test rows, found {len(df)}")
    if df["Molecule_Name"].duplicated().any():
        raise ValueError("Blinded test data contains duplicate Molecule_Name values")
    return df


def audit_data(data_dir: Path) -> DataAudit:
    data_dir = Path(data_dir)
    activity = load_activity(data_dir)
    tdi = load_tdi(data_dir)
    test = load_test(data_dir)
    emax = _read_csv(data_dir / EMAX_TRAIN_FILE)
    single = _read_csv(data_dir / SINGLE_CONCENTRATION_FILE)

    for label, frame in (("activity", activity), ("TDI", tdi), ("test", test)):
        invalid_ids = frame[list(IDENTIFIER_COLUMNS)].isna().any(axis=1).sum()
        if invalid_ids:
            raise ValueError(f"{label} data has {invalid_ids} rows with missing identifiers")

    return DataAudit(
        activity_rows=len(activity),
        tdi_rows=len(tdi),
        emax_rows=len(emax),
        single_concentration_rows=len(single),
        test_rows=len(test),
        activity_label_counts={col: int(activity[col].notna().sum()) for col in ACTIVITY_ENDPOINTS},
        tdi_label_counts={col: int(tdi[col].notna().sum()) for col in TDI_ENDPOINTS},
        tdi_positive_counts={col: int(tdi[col].fillna(False).sum()) for col in TDI_ENDPOINTS},
    )
