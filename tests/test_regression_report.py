import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.regression_report import compare, paired_bootstrap


def test_paired_cluster_bootstrap_rewards_consistent_improvement():
    rows = []
    for endpoint in ACTIVITY_ENDPOINTS:
        for i in range(30):
            y = float(i % 10)
            rows.append(
                {
                    "endpoint": endpoint,
                    "cluster": i // 2,
                    "y_true": y,
                    "conf_low": y - 0.1,
                    "conf_high": y + 0.1,
                    "candidate": y,
                    "reference": y + 1,
                }
            )
    result = paired_bootstrap(pd.DataFrame(rows), "candidate", "reference", samples=100)
    assert result["delta_95ci"][1] < 0
    assert result["probability_improvement"] == 1
    assert np.isfinite(result["delta_95ci"]).all()


def test_cross_run_comparison_supports_neural_and_rejects_changed_folds(tmp_path):
    candidate, reference = tmp_path / "neural", tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    rows = []
    for endpoint in ACTIVITY_ENDPOINTS:
        for i in range(30):
            rows.append(
                {
                    "Molecule_Name": f"m{i}",
                    "endpoint": endpoint,
                    "cluster": i,
                    "fold": i % 5,
                    "y_true": float(i % 10),
                    "conf_low": float(i % 10) - 0.1,
                    "conf_high": float(i % 10) + 0.1,
                    "y_pred": float(i % 10),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(candidate / "oof_predictions.csv", index=False)
    frame["nested_ensemble"] = frame.y_pred + 1
    frame.to_csv(reference / "oof_predictions.csv", index=False)
    result = compare(candidate, "nested_ensemble", reference)
    assert result["passes_numerical_gate"]
    saved = (candidate / "comparison.json").read_bytes()
    report = tmp_path / "matched-comparison.json"
    separate = compare(candidate, "nested_ensemble", reference, report)
    assert separate["candidate_macro"] == result["candidate_macro"]
    assert (candidate / "comparison.json").read_bytes() == saved
    with pytest.raises(FileExistsError):
        compare(candidate, "nested_ensemble", reference, report)
    with pytest.raises(FileExistsError):
        compare(candidate, "nested_ensemble", reference)
    assert (candidate / "comparison.json").read_bytes() == saved
    frame.loc[0, "fold"] = 4
    frame.to_csv(reference / "oof_predictions.csv", index=False)
    with pytest.raises(ValueError, match="different"):
        compare(candidate, "nested_ensemble", reference, tmp_path / "changed.json")
