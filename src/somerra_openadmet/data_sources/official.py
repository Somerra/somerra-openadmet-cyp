"""Prepare the official challenge data and its official auxiliary assays."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import (
    ACTIVITY_ENDPOINTS,
    ACTIVITY_TRAIN_FILE,
    EMAX_TRAIN_FILE,
    SINGLE_CONCENTRATION_FILE,
    TDI_TRAIN_FILE,
    TEST_FILE,
)
from ..data import load_activity, load_test
from ..download import sha256_file, verify_activity_data
from .common import balanced_folds, cluster_ids, parent_smiles, write_json


def prepare(data_dir: Path, output_dir: Path, n_folds: int = 5) -> dict:
    report = verify_activity_data(data_dir)
    if not report.valid:
        raise ValueError(f"Dataset verification failed: {report.errors}")
    train, test = load_activity(data_dir), load_test(data_dir)
    train["parent"] = train.SMILES.map(parent_smiles)
    test_parents = set(test.SMILES.map(parent_smiles))
    overlap = train.parent.isin(test_parents)
    excluded = train.loc[overlap, ["Molecule_Name", "parent"]]
    train = train.loc[~overlap].copy()
    parents = sorted(train.parent.unique())
    groups = dict(zip(parents, cluster_ids(parents), strict=True))
    train["cluster"] = train.parent.map(groups)
    train = train.sort_values(["parent", "Molecule_Name"]).reset_index(drop=True)
    train["fold"] = balanced_folds(
        train.cluster.to_numpy(), train[list(ACTIVITY_ENDPOINTS)].notna().to_numpy(), n_folds
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    train.to_csv(output_dir / "training.csv", index=False)
    excluded.to_csv(output_dir / "excluded_test_overlap.csv", index=False)
    manifest = {
        "schema_version": 1,
        "method": "ecfp4_2048_butina_distance_0.4",
        "n_folds": n_folds,
        "rows": len(train),
        "parents": len(parents),
        "clusters": len(set(groups.values())),
        "excluded_test_overlap": len(excluded),
        "input_sha256": {
            name: sha256_file(data_dir / name) for name in (ACTIVITY_TRAIN_FILE, TEST_FILE)
        },
        "training_sha256": sha256_file(output_dir / "training.csv"),
        "rdkit_version": version("rdkit"),
        "fold_label_counts": train.groupby("fold")[list(ACTIVITY_ENDPOINTS)].count().to_dict(),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


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
