import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from somerra_openadmet import workbench_probe as probe  # noqa: E402


def sample():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(50, 6)).astype(np.float32)
    y = rng.normal(size=(50, 8)).astype(np.float32)
    y[:10, 0] = np.nan
    return x, y


def test_saved_folds_reject_scaffold_or_parent_leakage():
    frame = pd.DataFrame(
        {
            "fold": np.arange(10) % 5,
            "scaffold": list("abcdefghij"),
            "connectivity": list("abcdefghij"),
        }
    )
    for train, val in probe.validate_folds(frame):
        assert not set(train) & set(val)
        assert len(train) == 8 and len(val) == 2
    for column in ("scaffold", "connectivity"):
        changed = frame.copy()
        changed.loc[1, column] = changed.loc[0, column]
        with pytest.raises(ValueError, match="crosses folds"):
            probe.validate_folds(changed)
    probe.validate_folds(changed, {"a": [0, 1]})
    with pytest.raises(ValueError, match="crosses folds"):
        probe.validate_folds(changed, {"a": [0, 2]})


def test_training_weights_and_masked_loss_use_observed_labels():
    x, y = sample()
    stats = probe.training_statistics(x[:30], y[:30])
    inverse = 1 / np.isfinite(y[:30, :4]).sum(0)
    np.testing.assert_allclose(stats["task_weights"][:4], inverse / inverse.mean())
    np.testing.assert_allclose(stats["task_weights"][4:], 0.3)
    model = probe.Head(6, probe.RECIPE, stats)
    target = torch.tensor([[1.0, float("nan"), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    prediction = torch.ones_like(target, requires_grad=True)
    mask = torch.isfinite(target)
    expected = (((prediction - target).abs() / model.target_scale) * model.task_weights)[
        mask
    ].mean()
    actual = probe.weighted_loss(model, prediction, target)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 1] == 0
    y[:, 7] = np.nan
    with pytest.raises(ValueError, match="needs labels"):
        probe.training_statistics(x, y)


def test_train_only_scaling_best_checkpoint_and_replay(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    x, y = sample()
    x[30:] += 100
    y[30:] += 100
    snapshots = []
    values = iter([1.0, 0.5, 0.7, 0.8])

    def validation(model, *_):
        model.eval()
        snapshots.append(copy.deepcopy(model.state_dict()))
        return next(values)

    monkeypatch.setattr(probe, "validation_loss", validation)
    recipe = {**probe.RECIPE, "max_epochs": 10, "patience": 2, "batch_size": 16}
    run = SimpleNamespace(device="cpu", check=lambda: None)
    predictions, record = probe.fit(
        x, y, np.arange(30), np.arange(30, 50), 42, run, tmp_path / "fit", recipe
    )
    restored, _ = probe.restore(tmp_path / "fit/best.pt")
    assert record["selected_epoch"] == 2 and record["epochs_run"] == 4
    for key, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, snapshots[1][key], rtol=0, atol=0)
    np.testing.assert_allclose(restored.feature_mean.numpy(), x[:30].mean(0))
    np.testing.assert_allclose(restored.target_mean.numpy(), np.nanmean(y[:30], axis=0))
    np.testing.assert_allclose(predictions, probe.predict_head(restored, x[30:]), atol=1e-6)
    assert record["cpu_replay_max_abs"] == 0
    assert len(json.loads((tmp_path / "fit/curve.json").read_text())) == 4


def test_no_calibration_floor_and_scored_output_order():
    model = probe.Head(6, probe.RECIPE).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head[-1].bias.fill_(-2)
        values = model(torch.zeros(3, 6)).numpy()
    assert (values == -2).all()
    values = np.tile(np.arange(8), (3, 1))
    scored = probe.scored_predictions(values)
    for ep in probe.ACTIVITY_ENDPOINTS:
        assert (scored[ep] == probe.TASKS.index(ep.lower())).all()
