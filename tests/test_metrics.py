import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS, TDI_ENDPOINTS
from somerra_openadmet.metrics import score_activity, score_tdi, soft_threshold_rae


def test_soft_threshold_predictions_inside_interval_have_zero_error():
    truth = np.array([4.0, 5.0, 6.0])
    pred = np.array([4.1, 4.9, 6.0])
    lower = np.array([3.8, 4.8, 5.8])
    upper = np.array([4.2, 5.2, 6.2])
    assert soft_threshold_rae(truth, pred, lower, upper) == pytest.approx(0.0)


def test_soft_threshold_mean_baseline_scores_one():
    truth = np.array([3.0, 5.0, 8.0])
    mean_prediction = np.repeat(truth.mean(), len(truth))
    assert soft_threshold_rae(truth, mean_prediction) == pytest.approx(1.0)


def test_track_scores_macro_average():
    truth = pd.DataFrame({"Molecule_Name": ["a", "b", "c"]})
    pred = pd.DataFrame({"Molecule_Name": ["a", "b", "c"]})
    for endpoint in ACTIVITY_ENDPOINTS:
        truth[endpoint] = [3.0, 5.0, 8.0]
        truth[f"{endpoint}_conf_low"] = [2.9, 4.9, 7.9]
        truth[f"{endpoint}_conf_high"] = [3.1, 5.1, 8.1]
        pred[endpoint] = [3.0, 5.0, 8.0]
    assert score_activity(truth, pred)["MA-ST-RAE"] == pytest.approx(0.0)

    tdi_truth = pd.DataFrame({"Molecule_Name": ["a", "b", "c", "d"]})
    tdi_pred = pd.DataFrame({"Molecule_Name": ["a", "b", "c", "d"]})
    for endpoint in TDI_ENDPOINTS:
        tdi_truth[endpoint] = [False, True, False, True]
        tdi_pred[endpoint] = [False, True, False, True]
    assert score_tdi(tdi_truth, tdi_pred)["MA-MCC"] == pytest.approx(1.0)
