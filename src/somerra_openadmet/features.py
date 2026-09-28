"""2D molecular features for transparent CPU baselines."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "HDonors",
    "HAcceptors",
    "RotatableBonds",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "FormalCharge",
)


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def canonical_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(mol_from_smiles(smiles), canonical=True)


def scaffold_key(smiles: str) -> str:
    """Return a generic Bemis-Murcko scaffold; acyclic compounds use canonical SMILES."""
    mol = mol_from_smiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return f"ACYCLIC::{Chem.MolToSmiles(mol, canonical=True)}"
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(generic, canonical=True)


def _descriptors(mol: Chem.Mol) -> np.ndarray:
    return np.asarray(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Descriptors.TPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.RingCount(mol),
            Descriptors.FractionCSP3(mol),
            Lipinski.HeavyAtomCount(mol),
            Chem.GetFormalCharge(mol),
        ],
        dtype=np.float32,
    )


def featurize_smiles(
    smiles: Iterable[str],
    *,
    radius: int = 2,
    n_bits: int = 2048,
    include_descriptors: bool = True,
) -> np.ndarray:
    """Generate ECFP fingerprints, optionally followed by 10 physical descriptors."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows: list[np.ndarray] = []
    for value in smiles:
        mol = mol_from_smiles(value)
        fingerprint = generator.GetFingerprint(mol)
        bits = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, bits)
        if include_descriptors:
            bits = np.concatenate((bits, _descriptors(mol)))
        rows.append(bits)
    if not rows:
        return np.empty((0, n_bits + (len(DESCRIPTOR_NAMES) if include_descriptors else 0)))
    return np.stack(rows)
