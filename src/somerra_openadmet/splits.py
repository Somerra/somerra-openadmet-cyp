"""Chemistry-aware cross-validation helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .features import scaffold_key


def scaffold_groups(smiles: Sequence[str]) -> np.ndarray:
    return np.asarray([scaffold_key(value) for value in smiles], dtype=object)


def regression_splits(
    smiles: Sequence[str], n_splits: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    groups = scaffold_groups(smiles)
    splitter = GroupKFold(n_splits=n_splits)
    positions = np.arange(len(groups))
    yield from splitter.split(positions, groups=groups)


def classification_splits(
    smiles: Sequence[str],
    labels: Sequence[int],
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    groups = scaffold_groups(smiles)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    positions = np.arange(len(groups))
    yield from splitter.split(positions, labels, groups=groups)


def assert_group_disjoint(
    smiles: Sequence[str], train_idx: np.ndarray, val_idx: np.ndarray
) -> None:
    groups = scaffold_groups(smiles)
    overlap = set(groups[train_idx]) & set(groups[val_idx])
    if overlap:
        raise AssertionError(f"Scaffold leakage detected for {len(overlap)} group(s)")
