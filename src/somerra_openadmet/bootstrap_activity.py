"""Official-style fixed-seed bootstrap reporting for the regression track."""

import numpy as np

from .constants import ACTIVITY_ENDPOINTS


def bootstrap_activity(truth, predictions, repeats=1000):
    if repeats < 2 or len(truth) != len(predictions):
        raise ValueError("Invalid bootstrap shape or repeat count")
    values = []
    endpoints = {}
    for ep in ACTIVITY_ENDPOINTS:
        observed = truth[ep].notna().to_numpy()
        y = truth.loc[observed, ep].to_numpy(dtype=float)
        lo = truth.loc[observed, ep + "_conf_low"].to_numpy(dtype=float)
        hi = truth.loc[observed, ep + "_conf_high"].to_numpy(dtype=float)
        pred = predictions.loc[observed, ep].to_numpy(dtype=float)
        if not len(y) or not np.isfinite(np.c_[y, lo, hi, pred]).all() or (lo > hi).any():
            raise ValueError("Invalid bootstrap observations")
        # Official evaluator restarts seed 0 for each endpoint's sample size.
        idx = np.random.default_rng(0).choice(len(y), size=(repeats, len(y)), replace=True)
        actual, low, high, predicted = y[idx], lo[idx], hi[idx], pred[idx]
        numerator = (np.maximum(predicted - high, 0) + np.maximum(low - predicted, 0)).sum(axis=1)
        mean = actual.mean(axis=1, keepdims=True)
        denominator = (np.maximum(mean - high, 0) + np.maximum(low - mean, 0)).sum(axis=1)
        if (denominator <= 0).any():
            raise ValueError("Degenerate bootstrap denominator")
        score = numerator / denominator
        values.append(score)
        endpoints[ep] = {"mean": float(score.mean()), "std": float(score.std(ddof=1))}
    macro = np.mean(values, axis=0)
    return {
        "MA-ST-RAE_bootstrap_mean": float(macro.mean()),
        "MA-ST-RAE_bootstrap_std": float(macro.std(ddof=1)),
        "MA-ST-RAE_bootstrap_percentile_95": np.quantile(macro, [0.025, 0.975]).tolist(),
        "endpoints": endpoints,
        "repeats": repeats,
        "seed": 0,
    }
