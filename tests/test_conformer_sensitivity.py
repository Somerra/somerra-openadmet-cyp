import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from somerra_openadmet.atom_jepa import (  # noqa: E402
    MODEL_REVISION,
    SOURCE_REVISION,
    WEIGHTS_SHA256,
    generate_conformer,
)
from somerra_openadmet.conformer_sensitivity import (  # noqa: E402
    CONFORMER_SEEDS,
    load_conformer_features,
    predict_conformers,
)
from somerra_openadmet.download import sha256_file  # noqa: E402
from somerra_openadmet.frozen_benchmark import fit_head, predict_head  # noqa: E402
from somerra_openadmet.representations import identities  # noqa: E402


def stub(path):
    path.mkdir()
    return SimpleNamespace(device="cpu", output=path, check=lambda *a: None, sync=lambda: None)


def test_one_conformer_extension_reproduces_original_training_exactly(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(41)
    x = rng.normal(size=(280, 8)).astype(np.float32)
    y = rng.normal(size=(280, 4)).astype(np.float32)
    fit = np.arange(257)
    old, first = fit_head(x, y, fit, 43, "standard", stub(tmp_path / "old"), epochs=3)
    new, second = fit_head(x[:, None], y, fit, 43, "standard", stub(tmp_path / "new"), epochs=3)
    for key in old.state_dict():
        torch.testing.assert_close(old.state_dict()[key], new.state_dict()[key], rtol=0, atol=0)
    assert first["batch_order_sha256"] == second["batch_order_sha256"]
    assert first["successful_updates"] == second["successful_updates"] == 3


def test_conformer_sampling_preserves_budget_and_has_no_outer_leakage(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(40)
    x = rng.normal(size=(280, 10, 8)).astype(np.float32)
    x[:, 1:] += 10  # Additional conformers must not enter scaler statistics.
    y = rng.normal(size=(280, 4)).astype(np.float32)
    fit = np.arange(270)
    first, a = fit_head(x, y, fit, 44, "standard", stub(tmp_path / "a"), epochs=3)
    np.testing.assert_allclose(first.mean.numpy(), x[fit, 0].mean(0), atol=1e-7)
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[270:] = 1e6
    changed_y[270:] = 1e6
    second, b = fit_head(changed_x, changed_y, fit, 44, "standard", stub(tmp_path / "b"), epochs=3)
    for key in first.state_dict():
        torch.testing.assert_close(
            first.state_dict()[key], second.state_dict()[key], rtol=0, atol=0
        )
    _, c = fit_head(x[:, :2, :4], y, fit, 44, "standard", stub(tmp_path / "c"), epochs=3)
    _, d = fit_head(x[:, :, :4], y, fit, 44, "standard", stub(tmp_path / "d"), epochs=3)
    assert a["batch_order_sha256"] == b["batch_order_sha256"] == c["batch_order_sha256"]
    assert a["successful_updates"] == c["successful_updates"] == 6
    assert a["conformer_order_sha256"] == b["conformer_order_sha256"] == d["conformer_order_sha256"]
    assert a["conformer_order_sha256"] != c["conformer_order_sha256"]


def test_evaluation_preserves_conformer_axis_and_averages_predictions():
    class Square(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return x.square() * self.scale

    model = Square()
    x = np.array([[[-2], [2]], [[1], [3]]], dtype=np.float32)
    values = predict_conformers(model, x)
    np.testing.assert_array_equal(values[:, 0], predict_head(model, x[:, 0]))
    np.testing.assert_array_equal(values.mean(1), [[4], [5]])
    assert not np.array_equal(values.mean(1), predict_head(model, x.mean(1)))


def test_geometry_seed_is_reproducible_and_changes_flexible_molecule():
    z, xyz, _ = generate_conformer("CCCCCCO", 43)
    z2, xyz2, _ = generate_conformer("CCCCCCO", 43)
    z3, xyz3, _ = generate_conformer("CCCCCCO", 44)
    np.testing.assert_array_equal(z, z2)
    np.testing.assert_array_equal(z, z3)
    np.testing.assert_array_equal(xyz, xyz2)
    # Compare pairwise distances, not orientations.
    d1 = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
    d2 = np.linalg.norm(xyz3[:, None] - xyz3[None, :], axis=-1)
    assert np.max(np.abs(d1 - d2)) > 0.01
    with pytest.raises(ValueError, match="seed"):
        generate_conformer("CCO", -1)


def test_multiconformer_cache_alignment_and_corruption(tmp_path):
    frame = pd.DataFrame({"Molecule_Name": ["a", "b"], "SMILES": ["CCO", "CC"]})
    identities(frame).to_csv(tmp_path / "rows.csv", index=False)
    x = np.arange(2 * 10 * 768, dtype=np.float32).reshape(2, 10, 768)
    np.save(tmp_path / "embeddings.npy", x)
    manifest = {
        "schema_version": 4,
        "kind": "atom_jepa_conformer_features",
        "complete": True,
        "conformer_seeds": list(CONFORMER_SEEDS),
        "checkpoint_sha256": WEIGHTS_SHA256,
        "source_revision": SOURCE_REVISION,
        "model_revision": MODEL_REVISION,
        "files": {n: sha256_file(tmp_path / n) for n in ("rows.csv", "embeddings.npy")},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    loaded, _ = load_conformer_features(tmp_path, frame.iloc[::-1])
    np.testing.assert_array_equal(loaded, x[::-1])
    np.save(tmp_path / "embeddings.npy", x + 1)
    with pytest.raises(ValueError, match="Changed conformer"):
        load_conformer_features(tmp_path, frame)


def test_remaining_budget_includes_prior_pilot_and_all_extension_stages(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_atom_jepa_conformer_sensitivity as extension
    import run_frozen_representation_pilot as pilot

    root, extraction, ledger = tmp_path / "pilot", tmp_path / "extract", tmp_path / "ledger.json"
    monkeypatch.setattr(pilot, "ROOT", root)
    monkeypatch.setattr(pilot, "EXTRACTION_PREFIX", extraction)
    monkeypatch.setattr(pilot, "LEDGER", ledger)
    records = {
        str(extraction): {"used_hours": 0.1, "reserved_hours": 1.5},
        str(root / "original-fit"): {"used_hours": 0.2, "reserved_hours": 0.3},
        str(root / "conformer-sensitivity-v1" / "failed"): {"reserved_hours": 3.6},
    }
    ledger.write_text(json.dumps(records))
    assert extension.allocation(1.5) == pytest.approx(0.1)
    records[str(root / "other")] = {"reserved_hours": 0.2}
    ledger.write_text(json.dumps(records))
    with pytest.raises(TimeoutError, match="four-GPU-hour"):
        extension.allocation(0.1)


def test_cpu_conformer_stage_saves_replayable_partial_fold(tmp_path):
    from somerra_openadmet.conformer_sensitivity import fit_stage
    from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
    from somerra_openadmet.frozen_benchmark import PROTOCOL, restore_head
    from somerra_openadmet.frozen_resume import completed_head

    prepared, cache, output = [tmp_path / name for name in ("prepared", "cache", "output")]
    prepared.mkdir()
    cache.mkdir()
    frame = pd.DataFrame(
        {"Molecule_Name": list("abcdefghij"), "SMILES": ["C" * i for i in range(1, 11)]}
    )
    frame["parent"] = identities(frame).parent
    frame["fold"] = np.arange(10) % 5
    frame["cluster"] = np.arange(10)
    for endpoint in ACTIVITY_ENDPOINTS:
        frame[endpoint] = np.arange(10, dtype=float)
        frame[endpoint + "_conf_low"] = frame[endpoint] - 0.2
        frame[endpoint + "_conf_high"] = frame[endpoint] + 0.2
    frame.to_csv(prepared / "training.csv", index=False)
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "n_folds": 5,
                "training_sha256": sha256_file(prepared / "training.csv"),
            }
        )
    )
    identities(frame).to_csv(cache / "rows.csv", index=False)
    x = np.random.default_rng(42).normal(size=(10, 10, 768)).astype(np.float32)
    np.save(cache / "embeddings.npy", x)
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "kind": "atom_jepa_conformer_features",
                "complete": True,
                "conformer_seeds": list(CONFORMER_SEEDS),
                "checkpoint_sha256": WEIGHTS_SHA256,
                "source_revision": SOURCE_REVISION,
                "model_revision": MODEL_REVISION,
                "files": {
                    name: sha256_file(cache / name) for name in ("rows.csv", "embeddings.npy")
                },
            }
        )
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "adaptation": "frozen",
                "device": "cpu",
                "threads": 1,
                "precision": "float32",
                "scaling": "standard",
                "scaler_conformer": 0,
                "selected_folds": [0],
                "representation": "scalar",
                "train_conformers": 2,
                "seed": 42,
                "output_dir": str(output),
                "prepared_dir": str(prepared),
                "cache_dir": str(cache),
                "max_gpu_hours": 0.1,
            }
        )
    )
    result = fit_stage(config)
    assert result["pilot_complete"] and not result["complete"]
    assert result["gpu_hours"] == 0 and result["encoder_trainable"] is False
    assert completed_head(output)
    model, _ = restore_head(output / "fold-0.pt")
    assert predict_conformers(model, x[:1, :, :256]).shape == (1, 10, 4)
