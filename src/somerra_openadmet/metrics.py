"""Local implementations of the challenge's primary activity metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from .constants import (
    ACTIVITY_ENDPOINTS,
    TDI_ENDPOINTS,
    lower_bound_column,
    upper_bound_column,
)
from .data import coerce_boolean_labels


def soft_threshold_rae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
) -> float:
    """Relative absolute error measured from the nearest credible-interval edge.

    This matches the current official OpenADMET scoring implementation: predictions
    inside a compound's interval contribute zero, and the constant-mean denominator
    is evaluated with the same interval rule.
    """
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    lower = true if y_lower is None else np.asarray(y_lower, dtype=float)
    upper = true if y_upper is None else np.asarray(y_upper, dtype=float)
    valid = np.isfinite(true) & np.isfinite(pred) & np.isfinite(lower) & np.isfinite(upper)
    if not valid.any():
        raise ValueError("No finite observations available for ST-RAE")
    true, pred, lower, upper = true[valid], pred[valid], lower[valid], upper[valid]
    if np.any(lower > upper):
        raise ValueError("Credible-interval lower bounds exceed upper bounds")

    error = np.maximum(pred - upper, 0.0) + np.maximum(lower - pred, 0.0)
    mean_true = float(np.mean(true))
    baseline = np.maximum(mean_true - upper, 0.0) + np.maximum(lower - mean_true, 0.0)
    denominator = float(np.sum(baseline))
    if denominator <= 0:
        raise ValueError("ST-RAE denominator is zero")
    return float(np.sum(error) / denominator)


def score_activity(truth: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    merged = truth.merge(
        predictions[["Molecule_Name", *ACTIVITY_ENDPOINTS]],
        on="Molecule_Name",
        how="inner",
        suffixes=("_truth", "_pred"),
        validate="one_to_one",
    )
    if len(merged) != len(truth):
        raise ValueError("Activity predictions do not cover every truth row")
    endpoints: dict[str, float] = {}
    for endpoint in ACTIVITY_ENDPOINTS:
        endpoints[endpoint] = soft_threshold_rae(
            merged[f"{endpoint}_truth"].to_numpy(),
            merged[f"{endpoint}_pred"].to_numpy(),
            merged[lower_bound_column(endpoint)].to_numpy(),
            merged[upper_bound_column(endpoint)].to_numpy(),
        )
    return {"ST-RAE": endpoints, "MA-ST-RAE": float(np.mean(list(endpoints.values())))}


def score_tdi(truth: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    merged = truth.merge(
        predictions[["Molecule_Name", *TDI_ENDPOINTS]],
        on="Molecule_Name",
        how="inner",
        suffixes=("_truth", "_pred"),
        validate="one_to_one",
    )
    if len(merged) != len(truth):
        raise ValueError("TDI predictions do not cover every truth row")
    endpoints: dict[str, float] = {}
    for endpoint in TDI_ENDPOINTS:
        y_true = coerce_boolean_labels(merged[f"{endpoint}_truth"])
        y_pred = coerce_boolean_labels(merged[f"{endpoint}_pred"])
        valid = y_true.notna() & y_pred.notna()
        endpoints[endpoint] = float(
            matthews_corrcoef(y_true[valid].astype(int), y_pred[valid].astype(int))
        )
    return {"MCC": endpoints, "MA-MCC": float(np.mean(list(endpoints.values())))}
