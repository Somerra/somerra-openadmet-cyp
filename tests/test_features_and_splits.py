import numpy as np

from somerra_openadmet.features import featurize_smiles, scaffold_key
from somerra_openadmet.splits import assert_group_disjoint, regression_splits

SMILES = [
    "c1ccccc1O",
    "c1ccccc1Cl",
    "C1CCCCC1O",
    "C1CCCCC1N",
    "CC(=O)O",
    "CCC(=O)O",
    "c1ncccc1",
    "c1ncccc1C",
    "C1CCCC1",
    "C1CCCC1F",
]


def test_featurizer_shape_and_finiteness():
    features = featurize_smiles(SMILES[:2], n_bits=64)
    assert features.shape == (2, 74)
    assert np.isfinite(features).all()


def test_generic_scaffold_groups_analogues():
    assert scaffold_key("c1ccccc1O") == scaffold_key("c1ccccc1Cl")


def test_group_split_has_no_scaffold_overlap():
    for train_idx, val_idx in regression_splits(SMILES, n_splits=4):
        assert_group_disjoint(SMILES, train_idx, val_idx)
