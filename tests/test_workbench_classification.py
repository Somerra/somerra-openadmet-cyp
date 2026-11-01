import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef

from somerra_openadmet.classification_metrics import best_threshold
from somerra_openadmet.workbench_classification import make_model, scaffold_folds, task_weights


def test_exact_threshold_matches_exhaustive_mcc():
    rng = np.random.default_rng(17)
    for _ in range(20):
        y = rng.integers(2, size=40)
        p = rng.choice([0, 0.02, 0.14, 0.5, 0.97, 1], size=40)
        threshold, score = best_threshold(y, p)
        expected = max(
            matthews_corrcoef(y, p >= t) for t in np.r_[np.unique(p), np.nextafter(1.0, np.inf)]
        )
        assert score == pytest.approx(expected)
        assert matthews_corrcoef(y, p >= threshold) == pytest.approx(expected)


def test_threshold_rejects_invalid_labels_and_probabilities():
    with pytest.raises(ValueError, match="Both binary"):
        best_threshold([1, 1], [0.2, 0.7])
    with pytest.raises(ValueError, match="finite"):
        best_threshold([0, 1], [np.nan, 0.7])


def test_workbench_scaffolds_remain_together_and_repeat():
    smiles = ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCCC1", "c1ccoc1"]
    folds, keys = scaffold_folds(smiles)
    again, _ = scaffold_folds(smiles)
    np.testing.assert_array_equal(folds, again)
    assert keys[0] == keys[1]
    assert folds[0] == folds[1]
    assert set(folds) == set(range(5))


def test_weights_balance_observed_heads_and_reject_missing_classes():
    y = np.array([[0, 0], [1, 1], [np.nan, 0], [np.nan, 1]])
    weights = np.array(task_weights(y))
    assert weights.mean() == pytest.approx(1)
    np.testing.assert_allclose(weights * np.isfinite(y).sum(axis=0), [8 / 3, 8 / 3])
    with pytest.raises(ValueError, match="both binary"):
        task_weights([[0, np.nan], [1, np.nan]])


def test_classification_head_masks_missing_labels_and_replays(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from chemprop import data, nn

    config = json.loads(Path("configs/workbench-classification.json").read_text())
    torch.manual_seed(42)
    expected_encoder = nn.BondMessagePassing(d_h=700, depth=5, dropout=0.1)
    torch.manual_seed(42)
    model = make_model(config, [1, 1])
    for key, tensor in expected_encoder.state_dict().items():
        torch.testing.assert_close(model.message_passing.state_dict()[key], tensor, rtol=0, atol=0)
    # Missing labels must contribute no loss or gradient to their output logit.
    logits = torch.tensor([[0.3, -0.6], [1.2, 0.2]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[True, False], [True, True]])
    loss = nn.metrics.BCELoss()(logits, targets, mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], targets[mask])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 1] == 0
    # Inference is sigmoid probabilities, without regression unscaling.
    ds = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in ["CC", "CCC"]])
    batch = next(iter(data.build_dataloader(ds, shuffle=False)))
    model.eval()
    with torch.no_grad():
        probabilities = model(batch.bmg)
        latent = model.fingerprint(batch.bmg)
        torch.testing.assert_close(probabilities, model.predictor.train_step(latent).sigmoid())
    assert probabilities.shape == (2, 2)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    checkpoint = tmp_path / "weights.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = make_model(config, [1, 1])
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(probabilities, restored(batch.bmg), rtol=0, atol=0)
