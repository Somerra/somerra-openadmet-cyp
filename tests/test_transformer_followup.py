import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chemprop")

from somerra_openadmet.smiles_transformer import decode, tokenize  # noqa: E402
from somerra_openadmet.transformer_followup import CachedInputs  # noqa: E402


def test_cached_batch_matches_tokenizer():
    smiles = ["CCO", "CCN", "c1ccccc1"]
    inputs = CachedInputs(smiles, np.array([0, 1]), device="cpu")
    for indices in [np.array([0]), np.array([2, 0])]:
        torch.testing.assert_close(
            inputs.batch(indices, "smiles_transformer", "cpu"),
            tokenize([smiles[i] for i in indices]),
        )


def test_augmentation_preserves_identity_and_cannot_touch_validation():
    from rdkit import Chem

    smiles = ["CCO", "C[C@H](N)C(=O)O", "c1ccccc1"]
    fit = np.array([0, 1])
    inputs = CachedInputs(smiles, fit, device="cpu", augment=True)
    again = CachedInputs(smiles, fit, device="cpu", augment=True)
    for epoch in range(1, 10):
        batch = inputs.training_batch(fit, "smiles_transformer", "cpu", epoch)
        torch.testing.assert_close(
            batch, again.training_batch(fit, "smiles_transformer", "cpu", epoch)
        )
        for row, expected in zip(batch, smiles, strict=False):
            assert (
                Chem.MolToSmiles(Chem.MolFromSmiles(decode(row)), isomericSmiles=True) == expected
            )
    with pytest.raises(ValueError, match="fit partition"):
        inputs.training_batch(np.array([2]), "smiles_transformer", "cpu", 1)
    torch.testing.assert_close(
        inputs.batch(np.array([2]), "smiles_transformer", "cpu"), tokenize([smiles[2]])
    )
