"""Binary classification diagnostics and exact MCC threshold selection."""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def _binary(y):
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Both binary classes must be present")
    return y.astype(int)


def _probabilities(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return p


def _mcc_counts(counts):
    tn, fp, fn, tp = np.asarray(counts, dtype=float).T
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(
        tp * tn - fp * fn,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )


def best_threshold(y, probability):
    """Exact MCC search; ties prefer the threshold closest to 0.5."""
    y, p = _binary(y), _probabilities(probability)
    if p.shape != y.shape:
        raise ValueError("Label and probability shapes differ")
    order = np.argsort(p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
    cumulative = np.r_[0, np.cumsum(sorted_y)]
    fn = cumulative[starts]
    tn = starts - fn
    tp = y.sum() - fn
    fp = len(y) - y.sum() - tn
    thresholds = sorted_p[starts]
    counts = np.column_stack([tn, fp, fn, tp])
    # Include an all-negative rule even when a fitted model returns exactly one.
    thresholds = np.r_[thresholds, np.nextafter(sorted_p[-1], np.inf)]
    counts = np.vstack([counts, [len(y) - y.sum(), 0, y.sum(), 0]])
    scores = _mcc_counts(counts)
    ties = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
    selected = ties[np.argmin(np.abs(thresholds[ties] - 0.5))]
    return float(thresholds[selected]), float(scores[selected])


def diagnostics(y, probability, prediction):
    y, p = _binary(y), _probabilities(probability)
    counts = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "labels": len(y),
        "positives": int(y.sum()),
        "mcc": float(_mcc_counts(counts)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "predicted_positive_rate": float(np.mean(prediction)),
        "confusion": dict(zip(("tn", "fp", "fn", "tp"), counts.tolist(), strict=True)),
    }
