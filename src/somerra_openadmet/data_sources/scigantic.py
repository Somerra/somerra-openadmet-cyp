"""Import frozen Scigantic auxiliary labels, never its scored labels or folds.

This is an attributed derivative snapshot, not a fresh ChEMBL/PubChem download.
Official direct and auxiliary labels continue to come from the official bundle.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..download import sha256_file
from .common import allowed_auxiliary, parent_smiles, write_json

REVISION = "e70a2920d611684c9aaff07400d951e2e79dd5f6"
GROUPS = {
    "chembl": [f"{ep}_pic50_chembl" for ep in ("cyp1a2", "cyp2c9", "cyp2d6", "cyp3a4", "cyp2c19")],
    "pubchem_response": [
        f"{ep}_max_response" for ep in ("cyp1a2", "cyp2c9", "cyp2d6", "cyp3a4", "cyp2c19")
    ],
    "pubchem_ac50": [f"{ep}_pic50_qhts" for ep in ("cyp1a2", "cyp2c9", "cyp2d6", "cyp3a4")],
}
SOURCE_TASKS = {
    "scigantic_chembl": GROUPS["chembl"],
    "scigantic_pubchem": GROUPS["pubchem_response"] + GROUPS["pubchem_ac50"],
}


def validate_snapshot(frame, evidence):
    sources = set(frame.source) if "source" in frame else set()
    if len(sources) != 1 or not sources.issubset(SOURCE_TASKS):
        raise ValueError("Unknown Scigantic derivative source")
    source = next(iter(sources))
    if evidence.get("upstream_revision") != REVISION or evidence.get("source") != source:
        raise ValueError("Scigantic derivative requires pinned provenance")
    if set(frame.task) - set(SOURCE_TASKS[source]) or not np.isfinite(frame.value).all():
        raise ValueError("Invalid Scigantic auxiliary labels")
    if frame.duplicated(["parent", "task"]).any():
        raise ValueError("Duplicate auxiliary parent/task")


def prepare(source_dir: Path, test_smiles, output_dir: Path):
    import subprocess

    revision = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError("Unexpected upstream revision")
    relative = "data/training/chemeleon_26_tdi/train_table.csv"
    if subprocess.check_output(
        ["git", "-C", str(source_dir), "status", "--porcelain", "--", relative], text=True
    ).strip():
        raise ValueError("Modified upstream table")
    path = source_dir / relative
    frame = pd.read_csv(path)
    structures = {s: parent_smiles(s) for s in sorted(frame.smiles.unique())}
    frame["parent"] = frame.smiles.map(structures)
    parents = sorted(frame.parent.unique())
    blind = sorted({parent_smiles(s) for s in test_smiles})
    allowed = allowed_auxiliary(parents, blind)
    permitted = set(np.asarray(parents)[allowed])
    excluded_frame = frame.loc[~frame.parent.isin(permitted)]
    exact = excluded_frame.parent.isin(blind)
    excluded = excluded_frame[["Molecule_Name", "parent"]]
    frame = frame.loc[frame.parent.isin(permitted)]
    output_dir.mkdir(parents=True, exist_ok=False)
    excluded.to_csv(output_dir / "blind_overlap_exclusions.csv", index=False)
    outputs = {}
    for source, tasks in SOURCE_TASKS.items():
        long = frame.melt(
            id_vars=["parent"], value_vars=tasks, var_name="task", value_name="value"
        ).dropna(subset=["value"])
        long = long.groupby(["parent", "task"], as_index=False).value.median()
        long["source"] = source
        dest = output_dir / source
        dest.mkdir()
        long.to_csv(dest / "assays.csv", index=False)
        evidence = {
            "source": source,
            "upstream_revision": REVISION,
            "upstream_table_sha256": sha256_file(path),
            "upstream_table": relative,
            "provenance_limit": "Published derivative; raw assay reconstruction unverified",
            "blind_exclusion": "standardized parent and ECFP4 Tanimoto >= 0.6",
            "excluded_upstream_rows": len(excluded),
            "excluded_exact_parent_rows": int(exact.sum()),
            "excluded_analogue_only_rows": int((~exact).sum()),
            "excluded_exact_parents": excluded_frame.loc[exact, "parent"].nunique(),
            "excluded_analogue_only_parents": excluded_frame.loc[~exact, "parent"].nunique(),
            "excluded_source_observations": int(excluded_frame[tasks].notna().sum().sum()),
            "rows": len(long),
            "parents": long.parent.nunique(),
            "task_counts": long.groupby("task").size().to_dict(),
            "assays_sha256": sha256_file(dest / "assays.csv"),
        }
        validate_snapshot(long, evidence)
        write_json(dest / "manifest.json", evidence)
        outputs[source] = str(dest.resolve())
    return outputs
