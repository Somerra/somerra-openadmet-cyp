"""Behavioral checks for the isolated matched transformer pilot."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chemprop")

from somerra_openadmet.smiles_transformer import (  # noqa: E402
    TOKENIZER,
    ReferenceMember,
    SmilesTransformer,
    count_parameters,
    decode,
    load_checkpoint,
    masked_mae,
    matched_architecture,
    tokenize,
)
from somerra_openadmet.transformer_pilot import TASKS, Inputs, batches, prepare  # noqa: E402


def test_tokenizer_roundtrip_and_rejection():
    strings = ["ClC[C@@H](O)c1ccccc1", "[NH4+].[Cl-]", "C" * 1100]
    encoded = tokenize(strings)
    assert [decode(row) for row in encoded] == strings
    for invalid in [[], [""], ["Cé"]]:
        with pytest.raises(ValueError):
            tokenize(invalid)


def test_padding_and_batch_composition_invariance():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = SmilesTransformer(32).eval()
    with torch.no_grad():
        alone = model(tokenize(["CCO"]))[0]
        together = model(tokenize(["CCO", "c1ccccc1CCCCCCCC"]))[0]
    torch.testing.assert_close(alone, together, atol=2e-6, rtol=2e-6)


def test_parameter_match_and_checkpoint(tmp_path):
    reference = count_parameters(ReferenceMember())
    architecture = matched_architecture(reference)
    model = SmilesTransformer(architecture["feedforward"]).eval()
    assert abs(count_parameters(model) / reference - 1) <= 0.05
    path = tmp_path / "model.pt"
    torch.save(
        {
            "architecture": architecture,
            "tokenizer": TOKENIZER,
            "tasks": TASKS,
            "means": [0.0] * 8,
            "scales": [1.0] * 8,
            "state_dict": model.state_dict(),
        },
        path,
    )
    restored, saved = load_checkpoint(path)
    x = tokenize(["CCO", "CCN"])
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored(x))
    assert saved["tasks"] == TASKS


def test_masked_loss_gradients():
    pred = torch.tensor([[2.0, 10.0], [4.0, 3.0]], requires_grad=True)
    target = torch.tensor([[1.0, float("nan")], [float("nan"), 1.0]])
    loss = masked_mae(pred, target, torch.tensor([1.0, 3.0]))
    assert loss.item() == 1.75
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[0.25, 0.0], [0.0, 0.75]]))
    with pytest.raises(ValueError):
        masked_mae(pred, torch.full_like(pred, float("nan")), torch.ones(2))


def test_both_families_backward_and_reference_roundtrip(tmp_path):
    torch.set_num_threads(2)
    inputs = Inputs(["CCO", "CCN", "c1ccccc1"])
    model = ReferenceMember()
    x = inputs.batch(np.arange(3), "mt_aux8", "cpu")
    loss = masked_mae(model(x), torch.zeros(3, 8), torch.ones(8))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    path = tmp_path / "reference.pt"
    torch.save(
        {
            "architecture": {"family": "mt_aux8"},
            "tokenizer": TOKENIZER,
            "state_dict": model.state_dict(),
        },
        path,
    )
    restored, _ = load_checkpoint(path)
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored(x))
    transformer = SmilesTransformer(32)
    masked_mae(transformer(tokenize(inputs.smiles)), torch.zeros(3, 8), torch.ones(8)).backward()
    assert all(torch.isfinite(p.grad).all() for p in transformer.parameters() if p.grad is not None)


def test_real_split_isolation_and_train_only_scaling():
    config = json.load(open("configs/regression-smiles-pilot.json"))
    from pathlib import Path

    if not Path(config["prepared_dir"]).exists():
        pytest.skip("Local prepared corpus not present")
    frame, target, means, scales, splits, _ = prepare(config)
    for key in ["parent", "cluster"]:
        fit, val, outer = [set(frame.iloc[ix][key]) for ix in splits]
        assert not (fit & val or fit & outer or val & outer)
    np.testing.assert_allclose(means, np.nanmean(frame.iloc[splits[0]][TASKS], axis=0), rtol=1e-5)
    np.testing.assert_allclose(np.nanmean(target[splits[0]], axis=0), 0, atol=1e-5)
    assert (scales > 0).all()
    assert set(frame.iloc[splits[2]].fold) == {0}
    ix = np.arange(129)
    groups = batches(ix)
    assert min(map(len, groups)) > 1
    np.testing.assert_array_equal(np.concatenate(groups), ix)
