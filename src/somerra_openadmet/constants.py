"""Challenge constants pinned to the 2026-09-23 activity-data release."""

from __future__ import annotations

HF_DATASET = "openadmet/cyp-challenge-train-test"
DEFAULT_DATA_REVISION = "146282d2ce1532f1803679a834a9bec9dac1e482"

ACTIVITY_TRAIN_FILE = "cyp-challenge-TRAIN_inhibition.csv"
TDI_TRAIN_FILE = "cyp-challenge-TRAIN_TDI.csv"
EMAX_TRAIN_FILE = "cyp-challenge-TRAIN_Emax.csv"
SINGLE_CONCENTRATION_FILE = "cyp-challenge-single-concentration-TRAIN.csv"
TEST_FILE = "cyp-challenge-TEST-BLINDED.csv"
DATASET_CARD_FILE = "README.md"

# Deliberately activity-only: the structure-track file is not downloaded by this package.
ACTIVITY_DATA_FILES = (
    ACTIVITY_TRAIN_FILE,
    TDI_TRAIN_FILE,
    EMAX_TRAIN_FILE,
    SINGLE_CONCENTRATION_FILE,
    TEST_FILE,
    DATASET_CARD_FILE,
)

IDENTIFIER_COLUMNS = ("Molecule_Name", "SMILES")
ACTIVITY_ENDPOINTS = (
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
)
TDI_ENDPOINTS = ("CYP2D6_is_TDI", "CYP3A4_is_TDI")
EXPECTED_TEST_ROWS = 750


def lower_bound_column(endpoint: str) -> str:
    return f"{endpoint}_conf_low"


def upper_bound_column(endpoint: str) -> str:
    return f"{endpoint}_conf_high"
