import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS  # noqa: E402
from somerra_openadmet.frozen_benchmark import (  # noqa: E402
    PROTOCOL,
    fit_head,
    make_head,
    predict_head,
    restore_head,
)


def run_stub(path):
    path.mkdir()
    return SimpleNamespace(device="cpu", output=path, check=lambda *a: None, sync=lambda: None)


def test_scaler_is_train_only_and_state_is_restorable(tmp_path):
    x = np.arange(120, dtype=np.float32).reshape(40, 3)
    x[30:] = 1e7
    fit = np.arange(30)
    model = make_head(x, fit, "standard", 42)
    np.testing.assert_allclose(model.mean.numpy(), x[:30].mean(0))
    assert all(name.startswith("head.") for name, _ in model.named_parameters())
    path = tmp_path / "head.pt"
    torch.save(
        {
            "schema_version": 3,
            "protocol": PROTOCOL,
            "tasks": list(ACTIVITY_ENDPOINTS),
            "dimension": 3,
            "state_dict": model.state_dict(),
        },
        path,
    )
    restored, _ = restore_head(path)
    np.testing.assert_allclose(predict_head(model, x), predict_head(restored, x), atol=1e-6)


def test_outer_labels_do_not_affect_fitting_and_batches_match_across_dimensions(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 3)).astype(np.float32)
    targets = rng.normal(size=(40, 4)).astype(np.float32)
    targets[0, 1] = np.nan
    fit = np.arange(30)
    first, record = fit_head(x, targets, fit, 42, "none", run_stub(tmp_path / "a"), epochs=3)
    changed = targets.copy()
    changed[30:] = 999
    second, other = fit_head(x, changed, fit, 42, "none", run_stub(tmp_path / "b"), epochs=3)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
    _, third = fit_head(
        np.tile(x, (1, 2)), targets, fit, 42, "none", run_stub(tmp_path / "c"), epochs=3
    )
    assert (
        record["batch_order_sha256"] == other["batch_order_sha256"] == third["batch_order_sha256"]
    )
    assert record["successful_updates"] == third["successful_updates"] == 3


def test_singleton_batch_merges_and_only_head_changes(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(258, 5)).astype(np.float32)
    y = rng.normal(size=(258, 4)).astype(np.float32)
    original = x.copy()
    model, record = fit_head(x, y, np.arange(257), 44, "none", run_stub(tmp_path / "fit"), epochs=2)
    assert record["successful_updates"] == 2
    np.testing.assert_array_equal(original, x)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_pilot_dispatch_rejects_finetuning(tmp_path):
    from somerra_openadmet.representation_training import screen_representations

    config = {
        "protocol": PROTOCOL,
        "encoder": "atom_jepa",
        "representation": "invariant",
        "prepared_dir": "unused",
        "cache_dir": "unused",
        "output_dir": "unused",
        "max_gpu_hours": 1,
        "scaling": "none",
        "selected_folds": [0],
        "seed": 42,
        "adaptation": "full",
        "ledger": "unused-legacy-pilot-ledger.json",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="only fits frozen"):
        screen_representations(path)


def test_bootstrap_pairs_seeds_without_turning_them_into_an_ensemble():
    import pandas as pd

    from somerra_openadmet.frozen_benchmark import paired_seed_bootstrap

    rows = []
    for seed, offset in [(42, -2), (43, 0), (44, 2)]:
        for i in range(20):
            for ep in ACTIVITY_ENDPOINTS:
                truth = float(i % 5)
                for arm, error in [("a", offset), ("b", 0.0)]:
                    rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "Molecule_Name": str(i),
                            "endpoint": ep,
                            "cluster": i,
                            "fold": 0,
                            "y_true": truth,
                            "conf_low": truth - 0.1,
                            "conf_high": truth + 0.1,
                            "y_pred": truth + error,
                        }
                    )
    frame = pd.DataFrame(rows)
    result = paired_seed_bootstrap(frame, "a", "b", samples=100)
    # Mean predictions are identical, but mean individual-seed errors are not.
    assert result["delta_95ci"][0] > 0
    frame.loc[0, "y_true"] += 0.01
    with pytest.raises(ValueError, match="do not match"):
        paired_seed_bootstrap(frame, "a", "b", samples=100)
