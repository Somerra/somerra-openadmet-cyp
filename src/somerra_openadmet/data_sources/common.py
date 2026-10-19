"""Shared helpers for preparing molecular data from distinct sources."""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Cluster import Butina


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def parent_smiles(value: str, *, stereo: bool = False) -> str:
    mol = Chem.MolFromSmiles(value)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid molecular structure: {value!r}")
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=stereo)


def cluster_ids(smiles: list[str], distance: float = 0.4) -> np.ndarray:
    """Deterministic Butina groups; input order is canonicalized by the caller."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    distances = []
    for i in range(1, len(fps)):
        distances.extend(1.0 - np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])))
    clusters = Butina.ClusterData(distances, len(fps), distance, isDistData=True)
    result = np.empty(len(fps), dtype=int)
    for group, indices in enumerate(clusters):
        result[list(indices)] = group
    return result


def balanced_folds(groups: np.ndarray, observed: np.ndarray, n_folds: int) -> np.ndarray:
    unique = np.unique(groups)
    if len(unique) < n_folds or n_folds < 2:
        raise ValueError("Need at least two folds and at least one molecular group per fold")
    counts = np.stack([observed[groups == group].sum(axis=0) for group in unique])
    sizes = np.asarray([(groups == group).sum() for group in unique])
    loads = np.zeros((n_folds, observed.shape[1] + 1))
    totals = np.maximum(np.r_[counts.sum(axis=0), sizes.sum()], 1)
    assignment = {}
    order = sorted(range(len(unique)), key=lambda i: (-int(counts[i].sum()), -sizes[i], i))
    for position, index in enumerate(order):
        weight = np.r_[counts[index], sizes[index]]
        costs = []
        for fold in range(n_folds):
            proposed = loads.copy()
            proposed[fold] += weight
            costs.append(float(np.sum((proposed / totals) ** 2)))
        fold = position if position < n_folds else int(np.argmin(costs))
        assignment[unique[index]] = fold
        loads[fold] += weight
    return np.asarray([assignment[group] for group in groups])


def potency(value: float, units: str, relation: str = "=") -> tuple[float, str]:
    """Convert molar concentration to p-scale, reversing inequality direction."""
    factors = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12}
    if units not in factors or relation not in {"=", "<", ">", "<=", ">="}:
        raise ValueError("Unsupported potency units or censoring relation")
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Potency must be finite and positive")
    reverse = {"=": "=", "<": ">", ">": "<", "<=": ">=", ">=": "<="}
    return -float(np.log10(value * factors[units])), reverse[relation]


def allowed_auxiliary(parents: list[str], heldout: list[str], threshold: float = 0.6):
    """Exclude exact parents and analogs of every held-out molecule, across sources.

    Unlike assigning a new molecule to its nearest cluster center, this prevents
    an external molecule bridging a training cluster and a validation cluster.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    holdout_fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in sorted(set(heldout))]
    holdout_set = set(heldout)
    result = []
    for parent in parents:
        fp = generator.GetFingerprint(Chem.MolFromSmiles(parent))
        similar = (
            bool(holdout_fps)
            and max(DataStructs.BulkTanimotoSimilarity(fp, holdout_fps)) >= threshold
        )
        result.append(parent not in holdout_set and not similar)
    return np.asarray(result, dtype=bool)


def request_json(url):
    """Retrieve a JSON response with bounded retries for transient source errors."""
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
