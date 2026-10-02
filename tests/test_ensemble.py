from pathlib import Path

import numpy as np

from somerra_openadmet.cli import build_parser
from somerra_openadmet.ensemble import _metric_weights


def test_metric_weights_prefer_better_predictor():
    labels = np.asarray([4.0, 5.0, 6.0, 7.0])
    lower = labels - 0.1
    upper = labels + 0.1
    predictions = np.column_stack([labels, np.full_like(labels, 5.5)])
    weights, score = _metric_weights(
        predictions,
        labels,
        lower,
        upper,
        seed=7,
        search_samples=128,
    )
    assert score == 0.0
    assert weights[0] > 0.9
    assert np.isclose(weights.sum(), 1.0)


def test_ensemble_cli_accepts_grouped_and_repeated_run_dirs():
    parser = build_parser()
    grouped = parser.parse_args(
        ["ensemble", "activity", "--run-dir", "one", "two", "--output-dir", "out"]
    )
    repeated = parser.parse_args(
        [
            "ensemble",
            "activity",
            "--run-dir",
            "one",
            "--run-dir",
            "two",
            "--output-dir",
            "out",
        ]
    )
    assert grouped.run_dir == [[Path("one"), Path("two")]]
    assert repeated.run_dir == [[Path("one")], [Path("two")]]
