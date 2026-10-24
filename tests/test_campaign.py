import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.campaign import balanced_folds, nested_endpoint, parent_smiles


def test_parent_identity_removes_salts_and_stereo_only_for_grouping():
    assert parent_smiles("CC[NH3+].[Cl-]") == parent_smiles("CCN")
    assert parent_smiles("C[C@H](O)CC") == parent_smiles("C[C@@H](O)CC")
    assert parent_smiles("C[C@H](O)CC", stereo=True) != parent_smiles("C[C@@H](O)CC", stereo=True)
    with pytest.raises(ValueError):
        parent_smiles("not a molecule")


def test_balanced_folds_keep_groups_together():
    groups = np.repeat(np.arange(12), 2)
    observed = np.ones((24, 4), dtype=bool)
    folds = balanced_folds(groups, observed, 3)
    assert set(folds) == {0, 1, 2}
    for group in np.unique(groups):
        assert len(set(folds[groups == group])) == 1
    np.testing.assert_array_equal(folds, balanced_folds(groups, observed, 3))


def test_nested_predictions_never_fit_outer_labels(monkeypatch):
    import somerra_openadmet.campaign as campaign

    n = 18
    frame = pd.DataFrame(
        {
            "y_true": np.arange(n, dtype=float),
            "conf_low": np.arange(n) - 0.1,
            "conf_high": np.arange(n) + 0.1,
            "cluster": np.arange(n),
            "fold": np.arange(n) % 3,
        }
    )
    x = np.arange(n).reshape(-1, 1)
    calls = []

    def predict(name, fit_x, y, query, seed, n_jobs):
        calls.append((set(fit_x[:, 0]), set(query[:, 0])))
        assert not set(fit_x[:, 0]) & set(query[:, 0])
        return None, np.full(len(query), np.mean(y))

    monkeypatch.setattr(campaign, "fit_predict", predict)
    base, blend, weights = nested_endpoint(frame, x, ["constant", "ridge"], inner_folds=2)
    assert np.isfinite(base).all() and np.isfinite(blend).all()
    assert len(weights) == 3
    # First outer fold has four inner fits followed by two outer refits.
    outer_holdout = set(x[frame.fold == 0, 0])
    assert all(not (fit & outer_holdout) for fit, _ in calls[:6])


def test_nested_rejects_group_overlap():
    frame = pd.DataFrame(
        {
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "conf_low": [1.0, 2.0, 3.0, 4.0],
            "conf_high": [1.0, 2.0, 3.0, 4.0],
            "cluster": [0, 0, 1, 1],
            "fold": [0, 1, 0, 1],
        }
    )
    with pytest.raises(ValueError, match="overlap"):
        nested_endpoint(frame, np.ones((4, 2)), ["constant", "ridge"])


@pytest.mark.parametrize("width", [2058, 27])
def test_radius_specific_models_receive_only_their_feature_block(monkeypatch, width):
    import somerra_openadmet.campaign as campaign

    class Model:
        def fit(self, x, y):
            assert x.shape == (3, width)
            np.testing.assert_array_equal(x, 2)

        def predict(self, x):
            np.testing.assert_array_equal(x, 2)
            return np.zeros(len(x))

    monkeypatch.setattr(campaign, "make_regressor", lambda *args: Model())
    x = np.concatenate([np.ones((3, width)), np.full((3, width), 2)], axis=1)
    campaign.fit_predict("lightgbm_r3", x, np.arange(3), x, 42, 1)


def test_evaluation_writes_complete_oof_and_rejects_tampered_split(tmp_path, monkeypatch):
    import json

    from somerra_openadmet.campaign import evaluate
    from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
    from somerra_openadmet.download import sha256_file

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows = []
    for i in range(18):
        row = {
            "Molecule_Name": f"m{i}",
            "SMILES": "C" * (i + 1),
            "parent": "C" * (i + 1),
            "cluster": i,
            "fold": i % 3,
        }
        for endpoint in ACTIVITY_ENDPOINTS:
            row[endpoint] = float(i % 7)
            row[endpoint + "_conf_low"] = float(i % 7) - 0.1
            row[endpoint + "_conf_high"] = float(i % 7) + 0.1
        rows.append(row)
    training = prepared / "training.csv"
    pd.DataFrame(rows).to_csv(training, index=False)
    (prepared / "manifest.json").write_text(json.dumps({"training_sha256": sha256_file(training)}))
    config = tmp_path / "config.json"
    output = tmp_path / "evaluation"
    config.write_text(
        json.dumps(
            {
                "prepared_dir": str(prepared),
                "output_dir": str(output),
                "models": ["constant", "ridge"],
                "inner_folds": 2,
            }
        )
    )
    import somerra_openadmet.campaign as campaign

    original_features = campaign.featurize_smiles

    def features(smiles):
        record = json.loads((output / "run_manifest.json").read_text())
        assert record["git_revision"] == "startup-revision"
        assert record["source_sha256"]
        assert record["complete"] is False
        monkeypatch.setattr(campaign, "git_revision", lambda: "later-revision")
        return original_features(smiles)

    monkeypatch.setattr(campaign, "git_revision", lambda: "startup-revision")
    monkeypatch.setattr(campaign, "featurize_smiles", features)
    result = evaluate(config)
    record = json.loads((output / "run_manifest.json").read_text())
    assert record["git_revision"] == "startup-revision"
    assert record["complete"] is True
    assert np.isfinite(result["macro"]["nested_ensemble"])
    assert len(pd.read_csv(output / "oof_predictions.csv")) == 18 * 4
    assert (output / "run_manifest.json").is_file()
    training.write_text(training.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        evaluate(config)


@pytest.mark.parametrize(
    "error", [FileNotFoundError(), __import__("subprocess").CalledProcessError(128, "git")]
)
def test_git_revision_is_optional_outside_checkout(monkeypatch, error):
    import somerra_openadmet.campaign as campaign

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(campaign.subprocess, "check_output", unavailable)
    assert campaign.git_revision() is None
