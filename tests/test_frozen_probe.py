import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from somerra_openadmet import frozen_probe as probe  # noqa: E402
from somerra_openadmet.constants import ACTIVITY_ENDPOINTS  # noqa: E402


def fixture_data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 5)).astype(np.float32)
    y = (x[:, :4] * [1, 2, 3, 4] + [4, 5, 6, 7]).astype(np.float32)
    frame = pd.DataFrame(
        {
            "Molecule_Name": [f"m{i}" for i in range(100)],
            "parent": [f"p{i}" for i in range(100)],
            "cluster": np.arange(100) // 2,
            "fold": np.arange(100) // 2 % 5,
        }
    )
    for i, endpoint in enumerate(ACTIVITY_ENDPOINTS):
        frame[endpoint] = y[:, i]
        frame[endpoint + "_conf_low"] = y[:, i] - 0.01
        frame[endpoint + "_conf_high"] = y[:, i] + 0.01
    return frame, x, y


def run_stub(path):
    path.mkdir()
    torch.set_num_threads(1)
    return SimpleNamespace(device="cpu", output=path, check=lambda: None)


def test_inner_split_is_group_disjoint_and_independent_of_outer_labels():
    frame, _, _ = fixture_data()
    first = probe.split_indices(frame)
    frame.loc[frame.fold == 0, list(ACTIVITY_ENDPOINTS)] = 999
    second = probe.split_indices(frame)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    assert set(np.concatenate(first)) == set(range(len(frame)))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert not set(frame.iloc[first[a]].cluster) & set(frame.iloc[first[b]].cluster)


def test_train_only_statistics_and_early_stopping_restore_best_epoch(tmp_path, monkeypatch):
    _, x, y = fixture_data()
    y[0, 0] = np.nan
    snapshots = []
    original_predict = probe.predict_head

    def predict(model, features):
        snapshots.append(copy.deepcopy(model.state_dict()))
        return original_predict(model, features)

    values = iter([1.0, 0.5, 0.6, 0.7])
    monkeypatch.setattr(probe, "predict_head", predict)
    monkeypatch.setattr(probe, "metric", lambda *args: next(values))
    recipe = {**probe.RECIPE, "max_epochs": 10, "patience": 2}
    model, scaler, record = probe.fit_mlp(
        x[:70],
        y[:70],
        x[70:] + 100,
        y[70:] + 100,
        np.stack([y[70:] - 0.01, y[70:] + 0.01], axis=2),
        42,
        run_stub(tmp_path / "fit"),
        recipe,
    )
    assert record["selected_epoch"] == 2 and record["epochs_run"] == 4
    np.testing.assert_allclose(scaler.mean_, x[:70].mean(0), atol=1e-7)
    np.testing.assert_allclose(model.target_mean.numpy(), np.nanmean(y[:70], axis=0))
    np.testing.assert_allclose(model.target_scale.numpy(), np.nanstd(y[:70], axis=0))
    assert model.encoder is None
    assert any(isinstance(layer, torch.nn.Dropout) for layer in model.head)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, snapshots[1][name], rtol=0, atol=0)


def test_outer_labels_do_not_change_selected_checkpoint_and_resume_is_verified(
    tmp_path, monkeypatch
):
    frame, x, _ = fixture_data()
    prepared = {"n_folds": 5, "training_sha256": "original"}
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    monkeypatch.setattr(probe, "prepared_data", lambda directory: (frame, prepared))
    monkeypatch.setattr(probe, "load_features", lambda config, data: (x, {}))
    monkeypatch.setattr(probe, "RECIPE", {**probe.RECIPE, "max_epochs": 5, "patience": 2})
    config = {
        "protocol": probe.PROTOCOL,
        "learner": "mlp",
        "encoder": "atom_jepa",
        "representation": "scalar",
        "cache_dir": str(cache),
        "prepared_dir": "unused",
        "output_dir": str(tmp_path / "fit"),
        "device": "cpu",
        "max_gpu_hours": 0.1,
        "seed": 42,
        "threads": 1,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    first = probe.screen(path)
    assert first["pilot_complete"] and not first["complete"]
    assert probe.completed(config)
    _, saved = probe.restore_probe(tmp_path / "fit/head.pt")
    frame.loc[frame.fold == 0, list(ACTIVITY_ENDPOINTS)] += 1000
    x[frame.fold == 0] += 1000
    other = {**config, "output_dir": str(tmp_path / "other")}
    path.write_text(json.dumps(other))
    second = probe.screen(path)
    _, changed = probe.restore_probe(tmp_path / "other/head.pt")
    assert first["selection"] == second["selection"]
    for name, value in saved["state_dict"].items():
        torch.testing.assert_close(value, changed["state_dict"][name], rtol=0, atol=0)
    prepared["training_sha256"] = "changed"
    with pytest.raises(ValueError, match="Changed v2 probe inputs"):
        probe.completed(config)
    prepared["training_sha256"] = "original"
    (tmp_path / "fit/head.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed v2 probe artifact"):
        probe.completed(config)


def test_controls_select_on_inner_validation_and_replay(tmp_path):
    import joblib

    _, x, y = fixture_data()
    recipe = {**probe.RECIPE, "tree_max_estimators": 5, "patience": 2}
    controls, scaler, means, scales = probe.fit_controls(
        x[:70],
        y[:70],
        x[70:85],
        y[70:85],
        np.stack([y[70:85] - 0.01, y[70:85] + 0.01], axis=2),
        run_stub(tmp_path / "controls"),
        recipe,
    )
    assert set(controls) == {"mean", "ridge", "lightgbm"}
    np.testing.assert_allclose(means, y[:70].mean(0))
    for name in ("ridge", "lightgbm"):
        control = controls[name]
        assert control["selection"]["inner_MA-ST-RAE"] == min(
            row["inner_MA-ST-RAE"] for row in control["trials"]
        )
    path = tmp_path / "controls.joblib"
    joblib.dump((controls, scaler, means, scales), path)
    restored, rs, rm, rt = joblib.load(path)
    for name, control in controls.items():
        prediction = probe.control_prediction(control, x[85:], scaler, means, scales)
        assert np.isfinite(prediction).all()
        np.testing.assert_array_equal(
            prediction, probe.control_prediction(restored[name], x[85:], rs, rm, rt)
        )
