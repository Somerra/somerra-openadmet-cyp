import json

import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.bootstrap_activity import bootstrap_activity
from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.data_sources.common import write_json
from somerra_openadmet.download import sha256_file
from somerra_openadmet.hillclimb import qualifies
from somerra_openadmet.reference_member_data import load_frames


def test_loss_parity_interval_masking_and_auxiliary_weight():
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from chemprop import nn

    from somerra_openadmet.hillclimb_model import training_loss

    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]], requires_grad=True)
    target = torch.tensor([[1.0, 1.0, float("nan"), 4.0, 2.0], [1.0, 3.0, 4.0, 4.0, 3.0]])
    weights = torch.tensor([2.0, 0.5])
    mask = target.isfinite()
    expected = nn.metrics.MAE()(pred, target.nan_to_num(), mask, weights)
    assert training_loss(pred, target, weights).item() == pytest.approx(expected.item())
    lower = target[:, :4] - 1
    upper = target[:, :4] + 1
    bounded = torch.cat([target, lower, upper], dim=1)
    loss = training_loss(pred, bounded, weights, "interval", 0.5)
    assert loss.item() == pytest.approx((3 * 2 + 3 * 0.5) * 0.5 / mask.sum().item())
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad[:, :4]) == 0
    bounded[0, 5] = 10
    with pytest.raises(ValueError, match="Invalid observed"):
        training_loss(pred, bounded, weights, "interval")


def test_bootstrap_matches_official_sampling_and_ratio():
    truth = pd.DataFrame()
    pred = pd.DataFrame()
    for i, ep in enumerate(ACTIVITY_ENDPOINTS):
        truth[ep] = np.linspace(2, 9, 100) + i * 0.1
        truth[ep + "_conf_low"] = truth[ep] - 0.1
        truth[ep + "_conf_high"] = truth[ep] + 0.2
        pred[ep] = truth[ep] + 0.5
    report = bootstrap_activity(truth, pred, repeats=100)
    idx = np.random.default_rng(0).choice(100, size=(100, 100), replace=True)
    y = truth[ACTIVITY_ENDPOINTS[0]].to_numpy()[idx]
    baseline = (
        np.maximum(y.mean(1)[:, None] - (y + 0.2), 0) + np.maximum(y - 0.1 - y.mean(1)[:, None], 0)
    ).sum(1)
    expected = 30 / baseline
    assert report["MA-ST-RAE_bootstrap_mean"] == pytest.approx(expected.mean())
    assert report["MA-ST-RAE_bootstrap_std"] == pytest.approx(expected.std(ddof=1))
    pred.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid bootstrap"):
        bootstrap_activity(truth, pred)


def test_both_holdouts_and_blind_excluded_across_every_task(tmp_path):
    official = pd.DataFrame(
        {
            "Molecule_Name": ["a", "b", "c"],
            "parent": ["CC", "CCC", "CCO"],
            "partition": ["train", "validation", "internal_test"],
        }
    )
    pool = pd.DataFrame(
        {
            "Molecule_Name": ["a", "b", "c", "ext", "blind"],
            "parent": ["CC", "CCC", "CCO", "CCN", "CCCC"],
        }
    )
    for ep in ACTIVITY_ENDPOINTS:
        pool[ep] = [1, 2, 3, np.nan, 5]
    pool["aux"] = 2
    official.to_csv(tmp_path / "official.csv", index=False)
    pool.to_csv(tmp_path / "pool.csv", index=False)
    pd.DataFrame({"parent": ["CCCC"]}).to_csv(tmp_path / "test.csv", index=False)
    manifest = {
        "protocol": "molecular-60-20-20-v1",
        "exclusion_policy": "exact_parent",
        "tasks": list(ACTIVITY_ENDPOINTS) + ["aux"],
        "groups": {},
        "file_sha256": {
            n: sha256_file(tmp_path / n) for n in ["official.csv", "pool.csv", "test.csv"]
        },
    }
    write_json(tmp_path / "manifest.json", manifest)
    fit, _, _, _ = load_frames(tmp_path)
    assert set(fit.Molecule_Name) == {"a", "ext"}
    final, _, _, _ = load_frames(tmp_path, final=True)
    assert set(final.Molecule_Name) == {"a", "b", "c", "ext"}
    manifest["exclusion_policy"] = "analogues"
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="policy mismatch"):
        load_frames(tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text())["protocol"]


def test_strategy_gate_uses_internal_test_not_validation():
    ref = {"internal_test_metrics": {"MA-ST-RAE": 1.0, "endpoints": {"a": 1.0, "b": 1.0}}}
    new = {
        "complete": True,
        "best_score": 0.1,
        "internal_test_metrics": {"MA-ST-RAE": 1.1, "endpoints": {"a": 1.0, "b": 1.2}},
    }
    assert not qualifies(ref, new)
    new["internal_test_metrics"] = {"MA-ST-RAE": 0.95, "endpoints": {"a": 0.9, "b": 1.0}}
    assert not qualifies(ref, new)  # Point gain alone is insufficient.
    new["paired_comparison"] = {"delta_95ci": [-0.1, 0.01]}
    assert not qualifies(ref, new)
    new["paired_comparison"] = {"delta_95ci": [-0.1, -0.01]}
    assert qualifies(ref, new)
    new["paired_comparison"] = {"delta_95ci": [-0.1, float("nan")]}
    assert not qualifies(ref, new)


def test_prediction_restores_training_precision_even_after_failure():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from somerra_openadmet.reference_member import predict

    class Model:
        training = True
        device = "cpu"
        fail = False

        def eval(self):
            self.training = False

        def train(self, mode):
            self.training = mode

        def __call__(self, graph):
            assert torch.get_float32_matmul_precision() == "highest"
            assert not self.training
            if self.fail:
                raise ValueError("test failure")
            return torch.ones((2, 4))

    model = Model()
    batch = SimpleNamespace(bmg=SimpleNamespace(to=lambda device: None))
    old = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        assert predict(model, [batch], list(ACTIVITY_ENDPOINTS)).shape == (2, 4)
        assert model.training and torch.get_float32_matmul_precision() == "high"
        model.fail = True
        with pytest.raises(ValueError, match="test failure"):
            predict(model, [batch], list(ACTIVITY_ENDPOINTS))
        assert model.training and torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(old)


def test_paired_bootstrap_uses_identical_molecules_and_detects_consistent_improvement():
    from somerra_openadmet.hillclimb_comparison import paired_comparison

    truth = pd.DataFrame(
        {
            "Molecule_Name": [f"m{i}" for i in range(32)],
            "parent": ["C" * (2 + i // 2) for i in range(32)],
        }
    )
    reference, candidate = truth.copy(), truth.copy()
    for ep in ACTIVITY_ENDPOINTS:
        truth[ep] = np.linspace(2, 9, 32)
        truth[ep + "_conf_low"] = truth[ep] - 0.1
        truth[ep + "_conf_high"] = truth[ep] + 0.1
        reference[ep] = truth[ep] + 1
        candidate[ep] = truth[ep]
    paired = paired_comparison(truth, reference, candidate, samples=100)
    assert paired["groups"] == 16
    assert paired["point_delta"] < 0 and paired["delta_95ci"][1] < 0
    same = paired_comparison(truth, reference, reference, samples=100)
    assert same["delta_95ci"] == [0, 0]
    with pytest.raises(ValueError, match="identities/order"):
        paired_comparison(truth, reference, candidate.iloc[::-1].reset_index(drop=True))
    candidate.loc[0, ACTIVITY_ENDPOINTS[0]] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        paired_comparison(truth, reference, candidate)


def test_training_frequency_normalization_equals_mean_task_loss():
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from somerra_openadmet.hillclimb_model import training_loss

    predictions = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]], requires_grad=True)
    targets = torch.tensor(
        [[0.0, 0.0], [0.0, float("nan")], [0.0, float("nan")], [0.0, float("nan")]]
    )
    weights = torch.tensor([2.0, 1.0, 1.0, 1.0])
    # Four training records, two tasks, observation counts [4, 1].
    scales = torch.tensor([0.5, 2.0])
    loss = training_loss(
        predictions, targets, weights, loss_normalization="tasks", task_scales=scales
    )
    assert loss.item() == pytest.approx(((2 + 2 + 3 + 4) / 4 + 8) / 2)
    parts = [
        training_loss(
            predictions[i : i + 2],
            targets[i : i + 2],
            weights[i : i + 2],
            loss_normalization="tasks",
            task_scales=scales,
        )
        for i in [0, 2]
    ]
    assert ((parts[0] + parts[1]) / 2).item() == pytest.approx(loss.item())
    loss.backward()
    assert torch.isfinite(predictions.grad).all()
    assert torch.count_nonzero(predictions.grad[1:, 1]) == 0
    with pytest.raises(ValueError, match="training-only"):
        training_loss(predictions, targets, weights, loss_normalization="tasks")
