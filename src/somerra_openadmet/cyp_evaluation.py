"""Fixed challenge evaluator shared by strategy discovery and submission building."""

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from .metrics import soft_threshold_rae


def regression_metrics(truth, prediction, endpoint):
    """Challenge metrics in raw pIC50; invalid predictions never shrink the cohort."""
    pred = np.asarray(prediction, float)
    if pred.shape != (len(truth),) or not np.isfinite(pred).all():
        raise ValueError("Invalid primary predictions")
    y = truth[endpoint].to_numpy(float)
    lo, hi = [truth[endpoint + s].to_numpy(float) for s in ["_conf_low", "_conf_high"]]
    observed = np.isfinite(y)
    if not observed.any() or not (np.isfinite(lo[observed]) & np.isfinite(hi[observed])).all():
        raise ValueError("Missing labels or credible intervals")
    y, pred, lo, hi = [v[observed] for v in [y, pred, lo, hi]]
    varied = len(y) > 1 and np.ptp(y) > 0
    ranks_defined = varied and np.ptp(pred) > 0
    return {
        "ST-RAE": soft_threshold_rae(y, pred, lo, hi),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)) if varied else None,
        "Spearman_R": float(spearmanr(y, pred).statistic) if ranks_defined else None,
        "Kendall_Tau": float(kendalltau(y, pred).statistic) if ranks_defined else None,
        "n": len(y),
    }
