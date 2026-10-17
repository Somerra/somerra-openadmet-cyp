import json

import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.download import sha256_file
from somerra_openadmet.recovery import merge_recovered, validate_fragment


def fixture_frames():
    train = pd.DataFrame(
        {
            "Molecule_Name": [f"m{i}" for i in range(10)],
            "parent": ["CC"] * 10,
            "cluster": np.arange(10),
            "fold": np.arange(10) % 5,
        }
    )
    parts = []
    for endpoint in ACTIVITY_ENDPOINTS:
        train[endpoint] = np.arange(10, dtype=float)
        train[endpoint + "_conf_low"] = train[endpoint] - 0.1
        train[endpoint + "_conf_high"] = train[endpoint] + 0.1
        part = train[["Molecule_Name", "parent", "cluster", "fold"]].copy()
        part["endpoint"] = endpoint
        part["y_true"] = train[endpoint]
        part["conf_low"] = train[endpoint + "_conf_low"]
        part["conf_high"] = train[endpoint + "_conf_high"]
        part["y_pred"] = train[endpoint] + 0.2
        parts.append(part)
    return train, pd.concat(parts, ignore_index=True)


def test_fragment_requires_exact_observed_coverage_and_finite_predictions():
    train, oof = fixture_frames()
    fragment = oof.loc[oof.fold < 4].copy()
    assert validate_fragment(fragment, train) == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="match"):
        validate_fragment(fragment.iloc[1:], train)
    with pytest.raises(ValueError, match="duplicate"):
        validate_fragment(pd.concat([fragment, fragment.iloc[:1]]), train)
    fragment.iloc[0, fragment.columns.get_loc("y_pred")] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        validate_fragment(fragment, train)


def test_merge_preserves_original_and_marks_provenance_limitation(tmp_path):
    train, oof = fixture_frames()
    prepared, original, recovered = (tmp_path / p for p in ("data", "original", "recovered"))
    for directory in (prepared, original, recovered):
        directory.mkdir()
    train.to_csv(prepared / "training.csv", index=False)
    config = {"prepared_dir": str(prepared), "output_dir": str(original), "epochs": 10}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    for directory, folds in ((original, [0, 1, 2, 3]), (recovered, [4])):
        oof.loc[oof.fold.isin(folds)].to_csv(directory / "oof_predictions.csv", index=False)
        for fold in folds:
            (directory / f"fold-{fold}.pt").write_bytes(b"fixture")
    manifest = {
        "config": {**config, "output_dir": str(recovered), "selected_folds": [4]},
        "requested_folds_complete": True,
        "completed_folds": [4],
        "prepared_manifest": {"training_sha256": sha256_file(prepared / "training.csv")},
    }
    (recovered / "run_manifest.json").write_text(json.dumps(manifest))
    before = sha256_file(original / "oof_predictions.csv")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "prediction_validation": {"status": "passed"},
                "sha256_before_validation": {str(original / "oof_predictions.csv"): before},
            }
        )
    )
    output = tmp_path / "completed"
    result = merge_recovered(config_path, recovered, output, audit)
    assert np.isfinite(result["MA-ST-RAE"])
    assert len(pd.read_csv(output / "oof_predictions.csv")) == len(oof)
    assert sha256_file(original / "oof_predictions.csv") == before
    evidence = json.loads((output / "run_manifest.json").read_text())
    assert evidence["complete"] and not evidence["promotion_eligible"]
    manifest["config"]["epochs"] = 11
    (recovered / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="parameters"):
        merge_recovered(config_path, recovered, tmp_path / "wrong", audit)
    manifest["config"]["epochs"] = 10
    (recovered / "run_manifest.json").write_text(json.dumps(manifest))
    original_oof = original / "oof_predictions.csv"
    original_oof.write_text(original_oof.read_text() + "\n")
    with pytest.raises(ValueError, match="Audited artifact changed"):
        merge_recovered(config_path, recovered, tmp_path / "tampered", audit)
