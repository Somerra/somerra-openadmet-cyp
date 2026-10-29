import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from somerra_openadmet.mol_jepa import (  # noqa: E402
    AttentionWithLoRA,
    TokenHead,
    load_view,
    write_views,
)
from somerra_openadmet.representations import identities  # noqa: E402


def test_attention_projection_parity_and_gradient():
    torch.manual_seed(42)
    original = torch.nn.MultiheadAttention(16, 4, batch_first=True, dropout=0).eval()
    adapter = AttentionWithLoRA(original).eval()
    x = torch.randn(3, 5, 16)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[:, -1] = True
    expected = original(x, x, x, key_padding_mask=mask)[0]
    actual = adapter(x, x, x, key_padding_mask=mask)[0]
    torch.testing.assert_close(expected, actual)
    actual.sum().backward()
    assert original.out_proj.weight.grad is not None
    assert torch.isfinite(original.out_proj.weight.grad).all()


def test_token_axis():
    head = TokenHead().eval()
    assert head(torch.randn(2, 12, 512)).shape == (2, 4)
    with pytest.raises(ValueError):
        head(torch.randn(2, 512, 12))


def test_cache_alignment_and_immutability(tmp_path):
    rows = identities(pd.DataFrame({"Molecule_Name": ["a", "b"], "SMILES": ["CCO", "CCN"]}))
    views = {
        key: np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        for key, shape in {
            "cls": (2, 512),
            "predictions": (2, 12, 512),
            "embeddings": (2, 13, 512),
        }.items()
    }
    metadata = {"encoder": "mol_jepa", "model_revision": "pinned"}
    write_views(tmp_path, rows, views, metadata)
    values, _ = load_view(tmp_path, rows.iloc[::-1], "predictions", metadata)
    np.testing.assert_array_equal(values, views["predictions"][::-1])
    with pytest.raises(FileExistsError):
        write_views(tmp_path, rows, views, metadata)
    with pytest.raises(ValueError, match="Incompatible"):
        load_view(tmp_path, rows, "cls", {"model_revision": "wrong"})
    changed = rows.copy()
    changed.loc[0, "SMILES"] = "CCC"
    with pytest.raises(ValueError, match="SMILES"):
        load_view(tmp_path, changed, "cls", metadata)


@pytest.mark.parametrize("arm", ["full", "lora"])
def test_real_model_gradients_and_frozen_state(arm):
    from pathlib import Path

    from somerra_openadmet.mol_jepa import Regressor

    directory = Path("runs/mol-jepa-model")
    if not (directory / "provenance.json").exists():
        pytest.skip("Pinned model not installed")
    torch.set_num_threads(4)
    model = Regressor(arm, directory).train()
    before = {n: p.clone() for n, p in model.encoder.state_dict().items()} if arm == "lora" else {}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-5)
    for _ in range(2):
        optimizer.zero_grad()
        model(["CCO", "CCN"]).square().mean().backward()
        if arm == "lora":
            for n, p in model.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None, n
                    assert torch.isfinite(p.grad).all(), n
        else:
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.encoder.model.encoders["graph"].parameters()
            )
        optimizer.step()
    if arm == "lora":
        for n, p in model.encoder.state_dict().items():
            if "lora_" not in n:
                torch.testing.assert_close(p, before[n], rtol=0, atol=0)


@pytest.mark.parametrize("arm", ["frozen", "tokens", "full", "lora"])
def test_checkpoint_replay(tmp_path, arm, monkeypatch):
    from pathlib import Path

    from somerra_openadmet.mol_jepa import Regressor
    from somerra_openadmet.representation_training import restore_checkpoint, save_checkpoint

    directory = Path("runs/mol-jepa-model")
    if arm in {"full", "lora"} and not (directory / "provenance.json").exists():
        pytest.skip("Pinned model not installed")
    torch.set_num_threads(4)
    model = Regressor(arm, directory).eval()
    value = torch.randn(2, 12, 512) if arm == "tokens" else torch.randn(2, 512)
    if arm in {"full", "lora"}:
        value = ["CCO", "CCN"]
    with torch.no_grad():
        expected = model(value)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "fold.pt"
    save_checkpoint(path, model, optimizer, {"adaptation": arm, "model_dir": str(directory)})
    monkeypatch.chdir(tmp_path)
    replay, _ = restore_checkpoint(path)
    with torch.no_grad():
        torch.testing.assert_close(replay(value), expected, rtol=0, atol=1e-4)


def test_precomputed_graphs_match_online_outputs_and_gradients():
    from pathlib import Path

    from somerra_openadmet.mol_jepa import Inputs, encode, load_foundation

    directory = Path("runs/mol-jepa-model")
    if not (directory / "provenance.json").exists():
        pytest.skip("Pinned model not installed")
    torch.set_num_threads(4)
    frame = pd.DataFrame({"SMILES": ["CCO", "CCN"]})
    model, _ = load_foundation(directory)
    model.eval()
    inputs = Inputs(frame, "full", "cpu", model_dir=directory)
    online = encode(model, frame.SMILES.tolist())["cls"]
    online.sum().backward()
    reference = next(model.model.encoders["graph"].parameters()).grad.clone()
    model.zero_grad(set_to_none=True)
    cached = encode(model, inputs.batch([0, 1]))["cls"]
    torch.testing.assert_close(cached, online, rtol=0, atol=0)
    cached.sum().backward()
    torch.testing.assert_close(next(model.model.encoders["graph"].parameters()).grad, reference)


def test_zero_lora_matches_pinned_model():
    from pathlib import Path

    from somerra_openadmet.mol_jepa import encode, install_lora, load_foundation

    directory = Path("runs/mol-jepa-model")
    if not (directory / "provenance.json").exists():
        pytest.skip("Pinned model not installed")
    torch.set_num_threads(4)
    model, _ = load_foundation(directory)
    model.eval()
    with torch.no_grad():
        expected = encode(model, ["CCO", "CCN"])["cls"]
    install_lora(model)
    model.eval()
    with torch.no_grad():
        actual = encode(model, ["CCO", "CCN"])["cls"]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_probe_falls_back_for_nonfinite_bf16_only():
    from somerra_openadmet.representation_training import precision_probe
    from somerra_openadmet.representations import NonFiniteRepresentationError

    class Probe(torch.nn.Linear):
        def forward(self, value):
            if torch.is_autocast_enabled("cpu"):
                raise NonFiniteRepresentationError("non-finite probe")
            return super().forward(value)

    class Inputs:
        device = "cpu"

        def batch(self, indices):
            return torch.ones(len(indices), 2)

    model = Probe(2, 4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    precision, record = precision_probe(model, Inputs(), np.array([0, 1]), np.ones((2, 4)), "bf16")
    assert precision == "float32"
    assert "non-finite" in record["fallback_reason"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_restore_rejects_different_foundation(tmp_path):
    from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
    from somerra_openadmet.representation_training import restore_checkpoint

    path = tmp_path / "bad.pt"
    torch.save(
        {
            "schema_version": 2,
            "encoder": "mol_jepa",
            "tasks": list(ACTIVITY_ENDPOINTS),
            "checkpoint_sha256": "wrong",
        },
        path,
    )
    with pytest.raises(ValueError, match="foundation provenance"):
        restore_checkpoint(path)


def test_full_arm_uses_real_training_step():
    from pathlib import Path

    from somerra_openadmet.mol_jepa import Inputs, Regressor
    from somerra_openadmet.representation_training import step

    directory = Path("runs/mol-jepa-model")
    if not (directory / "provenance.json").exists():
        pytest.skip("Pinned model not installed")
    torch.set_num_threads(4)
    model = Regressor("full", directory).train()
    inputs = Inputs(pd.DataFrame({"SMILES": ["CCO", "CCN"]}), "full", "cpu", model_dir=directory)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss, _ = step(model, optimizer, inputs, np.array([0, 1]), np.ones((2, 4)), "float32")
    assert np.isfinite(loss)
