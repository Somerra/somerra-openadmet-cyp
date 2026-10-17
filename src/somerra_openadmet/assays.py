"""Official auxiliary regression targets and fold-safe molecular exclusions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .campaign import parent_smiles, write_json
from .constants import EMAX_TRAIN_FILE, SINGLE_CONCENTRATION_FILE, TDI_TRAIN_FILE
from .data import load_test
from .download import sha256_file


def potency(value: float, units: str, relation: str = "=") -> tuple[float, str]:
    """Convert molar concentration to p-scale, reversing inequality direction."""
    factors = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12}
    if units not in factors or relation not in {"=", "<", ">", "<=", ">="}:
        raise ValueError("Unsupported potency units or censoring relation")
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Potency must be finite and positive")
    reverse = {"=": "=", "<": ">", ">": "<", "<=": ">=", ">=": "<="}
    return -float(np.log10(value * factors[units])), reverse[relation]


def auxiliary(data_dir: Path, output_dir: Path) -> dict:
    records = []
    for filename, token in (
        (TDI_TRAIN_FILE, "_pIC50_TDI_condition"),
        (EMAX_TRAIN_FILE, "_EmaxVsPosCtrl_"),
    ):
        frame = pd.read_csv(data_dir / filename)
        columns = [
            c for c in frame if token in c and not any(suffix in c for suffix in ("_conf_", "_std"))
        ]
        for column in columns:
            selected = frame.loc[frame[column].notna(), ["Molecule_Name", "SMILES", column]]
            for name, smiles, value in selected.itertuples(index=False, name=None):
                records.append(
                    {
                        "Molecule_Name": name,
                        "SMILES": smiles,
                        "task": column,
                        "value": float(value),
                        "source": filename,
                    }
                )
    frame = pd.read_csv(data_dir / SINGLE_CONCENTRATION_FILE)
    for row in frame.itertuples(index=False):
        if np.isfinite(row.log2fc_estimate) and np.isfinite(row.concentration_M):
            records.append(
                {
                    "Molecule_Name": row.Molecule_Name,
                    "SMILES": row.SMILES,
                    "task": f"{row.enzyme}_log2fc_{row.concentration_M:.8g}M",
                    "value": float(row.log2fc_estimate),
                    "source": SINGLE_CONCENTRATION_FILE,
                }
            )
    data = pd.DataFrame(records)
    structures = {s: parent_smiles(s) for s in data.SMILES.unique()}
    data["parent"] = data.SMILES.map(structures)
    test_parents = set(load_test(data_dir).SMILES.map(parent_smiles))
    excluded = data.parent.isin(test_parents)
    data = data.loc[~excluded].copy()
    if not np.isfinite(data.value).all():
        raise ValueError("Auxiliary targets contain non-finite values")
    # Replicates remain within the same source, molecular parent, and assay head.
    data = data.groupby(["parent", "task", "source"], as_index=False).agg(
        value=("value", "median"),
        SMILES=("SMILES", "first"),
        Molecule_Name=("Molecule_Name", "first"),
        replicates=("value", "size"),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    data.to_csv(output_dir / "assays.csv", index=False)
    manifest = {
        "rows": len(data),
        "parents": data.parent.nunique(),
        "excluded_test_rows": int(excluded.sum()),
        "task_counts": data.groupby("task").size().to_dict(),
        "input_sha256": {
            f: sha256_file(data_dir / f)
            for f in (TDI_TRAIN_FILE, EMAX_TRAIN_FILE, SINGLE_CONCENTRATION_FILE)
        },
        "assays_sha256": sha256_file(output_dir / "assays.csv"),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


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
