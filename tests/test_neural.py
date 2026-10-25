import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chemprop")

from somerra_openadmet.neural import (  # noqa: E402
    interval_loss,
    masked_huber,
    source_balanced_huber,
)


def test_masked_huber_ignores_missing_targets_and_balances_heads():
    prediction = torch.zeros((2, 2), requires_grad=True)
    target = torch.tensor([[1.0, float("nan")], [1.0, 2.0]])
    loss = masked_huber(prediction, target)
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert prediction.grad[0, 1].item() == 0
    assert torch.isfinite(prediction.grad).all()


def test_empty_target_batch_rejected():
    with pytest.raises(ValueError, match="no observed"):
        masked_huber(torch.zeros((2, 2)), torch.full((2, 2), float("nan")))


def test_source_balanced_huber_with_finite_missing_gradients():
    prediction = torch.zeros((2, 2), requires_grad=True)
    target = torch.tensor([[1.0, 1.0], [float("nan"), 0.0]])
    loss = source_balanced_huber(prediction, target, torch.tensor([0, 1]))
    assert loss.item() == pytest.approx((0.5 + 0.25) / 2)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[1, 0] == 0


def test_interval_loss_masks_missing_and_penalizes_outside_bounds():
    prediction = torch.tensor([[0.0, 3.0, 2.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0, float("nan")]])
    lower = torch.tensor([[-1.0, -1.0, float("nan")]])
    upper = torch.tensor([[1.0, 1.0, float("nan")]])
    loss = interval_loss(prediction, target, lower, upper)
    assert loss.item() == pytest.approx(1.125)
    loss.backward()
    assert prediction.grad[0, 2] == 0
    assert torch.isfinite(prediction.grad).all()


def test_matched_interval_loss_keeps_auxiliary_mae_and_masks_missing():
    from somerra_openadmet.neural_models import masked_mae

    pred = torch.tensor([[0.0, 3.0, 0.0, 0.0, 2.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 0.0, float("nan"), float("nan"), 1.0]])
    lower = torch.tensor([[-1.0, -1.0, float("nan"), float("nan")]])
    upper = torch.tensor([[1.0, 1.0, float("nan"), float("nan")]])
    weights = torch.ones(5)
    loss = masked_mae(pred, targets, weights, bounds=(lower, upper))
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0, 2] == 0
    assert pred.grad[0, 4] != 0
    assert masked_mae(pred, targets, weights).item() == pytest.approx(4 / 3)


def test_eight_head_checkpoint_roundtrip_on_cpu(tmp_path):
    from somerra_openadmet.neural import loader
    from somerra_openadmet.neural_export import load_checkpoint
    from somerra_openadmet.neural_models import build_model

    mp, agg, head = build_model("scigantic_mt_aux8", 8)
    mp.eval()
    head.eval()
    batch = next(iter(loader(["CCN", "CCO"])))
    with torch.no_grad():
        expected = head(agg(mp(batch.bmg), batch.bmg.batch))
    buffer = tmp_path / "model.pt"
    torch.save(
        {
            "encoder_state": mp.state_dict(),
            "head_state": head.state_dict(),
            "encoder_parameters": mp.hparams,
        },
        buffer,
    )
    saved = load_checkpoint(buffer)
    new_mp, new_agg, new_head = build_model("scigantic_mt_aux8", 8)
    new_mp.load_state_dict(saved["encoder_state"])
    new_head.load_state_dict(saved["head_state"])
    new_mp.eval()
    new_head.eval()
    with torch.no_grad():
        actual = new_head(new_agg(new_mp(batch.bmg), batch.bmg.batch))
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 8)


def test_validation_loader_keeps_singleton_last_batch():
    from somerra_openadmet.neural import loader

    batches = list(loader(["C", "CC", "CCC"], batch_size=2))
    assert len(batches) == 2


@pytest.mark.parametrize("kind", ["swapped", "missing", "auxiliary_first"])
def test_rejects_misaligned_official_task_order(kind):
    from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
    from somerra_openadmet.neural import validate_task_order

    tasks = list(ACTIVITY_ENDPOINTS)
    validate_task_order(tasks + ["auxiliary"])
    if kind == "swapped":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif kind == "missing":
        tasks.pop()
    else:
        tasks.insert(0, "auxiliary")
    with pytest.raises(ValueError, match="ACTIVITY_ENDPOINTS order"):
        validate_task_order(tasks)
