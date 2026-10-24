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
