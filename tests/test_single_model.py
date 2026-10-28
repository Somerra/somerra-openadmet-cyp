from copy import deepcopy

import pytest

from somerra_openadmet.foundation import encoder_identity, transfer_state
from somerra_openadmet.single_model import selected_epochs


def test_encoder_rejects_unrecognized_file_before_loading(tmp_path):
    path = tmp_path / "untrusted.pt"
    path.write_bytes(b"not the pinned encoder")
    with pytest.raises(ValueError, match="Unrecognized"):
        encoder_identity(path)


def test_exact_encoder_transfer_rejects_partial_mismatched_and_nonfinite_weights():
    torch = pytest.importorskip("torch")
    target = torch.nn.Linear(2, 3)
    source = torch.nn.Linear(2, 3)
    result = transfer_state(target, source.state_dict())
    assert result["tensors"] == 2
    assert torch.equal(target.weight, source.weight)
    with pytest.raises(ValueError, match="keys"):
        transfer_state(target, {"weight": source.weight})
    with pytest.raises(ValueError, match="shape"):
        transfer_state(target, {"weight": torch.ones(3, 4), "bias": source.bias})
    with pytest.raises(ValueError, match="values"):
        transfer_state(target, {"weight": torch.full((3, 2), float("nan")), "bias": source.bias})


def test_refit_epoch_gate_requires_matching_complete_selection():
    plan = {
        "selection_epochs": 100,
        "initialization": {"kind": "chemeleon", "sha256": "pinned"},
        "data_manifest_sha256": "data",
        "seed": 42,
        "scheduler_horizon": 200,
    }
    model = {"depth": 4}
    record = {
        **plan,
        "complete": True,
        "final": False,
        "completed_epochs": 100,
        "epoch_limit": 100,
        "model_config": model,
        "best_epoch": 73,
    }
    assert selected_epochs(record, plan, model) == 73
    for key, value in [
        ("complete", False),
        ("final", True),
        ("completed_epochs", 99),
        ("initialization", None),
        ("data_manifest_sha256", "other"),
        ("model_config", {"depth": 6}),
        ("best_epoch", 0),
        ("best_epoch", 101),
        ("seed", 43),
        ("scheduler_horizon", 100),
    ]:
        bad = deepcopy(record)
        bad[key] = value
        with pytest.raises(ValueError):
            selected_epochs(bad, plan, model)
