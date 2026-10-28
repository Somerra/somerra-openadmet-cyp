"""Audit Scigantic's published component OOF predictions on common official rows.

This does not fit a stacker or claim an independent OOF score for the final R25 blend.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ACTIVITY_ENDPOINTS
from .data_sources.common import write_json
from .data_sources.scigantic import REVISION
from .download import sha256_file
from .reference_member import score_predictions

MEMBERS = [
    "armA",
    "armB",
    "armC",
    "armD",
    "ivB2048",
    "ivC2048",
    "mu2048",
    "ii2048",
    "mt_aux8",
    "union18",
    "aux_ac50",
    "cp_interval",
]


def normalized_predictions(frame):
    if frame.Molecule_Name.duplicated().any():
        raise ValueError("Duplicate reference prediction IDs")
    result = frame[["Molecule_Name"]].copy()
    for ep in ACTIVITY_ENDPOINTS:
        options = [ep.lower() + "_pred", ep.lower(), ep]
        column = next((c for c in options if c in frame), None)
        if column is None:
            raise ValueError(f"Missing reference endpoint {ep}")
        result[ep] = pd.to_numeric(frame[column], errors="raise")
    return result


def audit(source, bundle, output):
    source, bundle, output = map(lambda p: Path(p).resolve(), (source, bundle, output))
    if output.exists():
        raise FileExistsError(output)
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError("Unpinned Scigantic source")
    if subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--", "data", "scripts/build_r25.py"], text=True
    ):
        raise ValueError("Changed published reference inputs")
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, digest in manifest["file_sha256"].items():
        if sha256_file(bundle / name) != digest:
            raise ValueError("Changed official evaluation bundle")
    truth = pd.read_csv(bundle / "official.csv")
    if truth.Molecule_Name.duplicated().any():
        raise ValueError("Duplicate official evaluation IDs")
    data = source / "data"
    table = pd.read_csv(data / "training/chemeleon_26_tdi/train_table.csv")
    folds = np.load(data / "training/chemeleon_26_tdi/folds.npy", allow_pickle=False)
    if (
        table.Molecule_Name.duplicated().any()
        or len(folds) != len(table)
        or set(folds) != set(range(5))
    ):
        raise ValueError("Invalid published folds")
    membership = pd.Series(folds, index=table.Molecule_Name)
    truth["reference_fold"] = truth.Molecule_Name.map(membership)
    if truth.reference_fold.isna().any():
        raise ValueError("Official records missing reference folds")
    direct_label_audit = {}
    for ep in ACTIVITY_ENDPOINTS:
        original = (
            table.set_index("Molecule_Name")[ep.lower()].reindex(truth.Molecule_Name).to_numpy()
        )
        current = truth[ep].to_numpy()
        matched = np.isclose(original, current, equal_nan=True, rtol=0, atol=1e-7)
        direct_label_audit[ep] = {
            "matching": int(matched.sum()),
            "differing": int((~matched).sum()),
        }
    frames, metadata = {}, {}
    for member in MEMBERS:
        frame = pd.read_csv(data / f"members/{member}/oof.csv")
        pred = normalized_predictions(frame).set_index("Molecule_Name")
        frames[member] = pred
        fold_column = next((c for c in ["fold", "_briford_fold"] if c in frame), None)
        differences = None
        if fold_column:
            local = frame.set_index("Molecule_Name")[fold_column].reindex(truth.Molecule_Name)
            observed = local.notna()
            differences = int(
                (
                    local[observed].to_numpy()
                    != truth.loc[observed.to_numpy(), "reference_fold"].to_numpy()
                ).sum()
            )
        metadata[member] = {
            "rows": len(pred),
            "fold_column": fold_column,
            "official_fold_id_differences": differences,
            "oof_sha256": sha256_file(data / f"members/{member}/oof.csv"),
        }
    common = set(truth.Molecule_Name)
    for frame in frames.values():
        common &= set(
            frame.index[np.isfinite(frame[list(ACTIVITY_ENDPOINTS)].to_numpy()).all(axis=1)]
        )
    matched_truth = truth.loc[truth.Molecule_Name.isin(common)].reset_index(drop=True)
    if matched_truth.empty or set(matched_truth.reference_fold) != set(range(5)):
        raise ValueError("Incomplete common-fold evaluation coverage")
    rows = []
    for member, frame in frames.items():
        for k in [None, *range(5)]:
            held = (
                matched_truth
                if k is None
                else matched_truth.loc[matched_truth.reference_fold == k].reset_index(drop=True)
            )
            pred = frame.loc[held.Molecule_Name, list(ACTIVITY_ENDPOINTS)].reset_index(drop=True)
            score = score_predictions(held, pred)
            rows.append(
                {
                    "member": member,
                    "fold": "pooled" if k is None else k,
                    "official_rows": len(held),
                    "MA-ST-RAE": score["MA-ST-RAE"],
                    **score["endpoints"],
                }
            )
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output / "component-scores.csv", index=False)
    truth[["Molecule_Name", "parent", "reference_fold"]].to_csv(
        output / "official-folds.csv", index=False
    )
    table[["Molecule_Name", "smiles"]].assign(reference_fold=folds).to_csv(
        output / "pool-folds.csv", index=False
    )
    report = {
        "reference_revision": revision,
        "official_rows": len(truth),
        "common_official_rows": len(matched_truth),
        "members": metadata,
        "direct_label_audit": direct_label_audit,
        "official_fold_sizes": truth.reference_fold.value_counts().sort_index().to_dict(),
        "pool_fold_sizes": pd.Series(folds).value_counts().sort_index().to_dict(),
        "input_sha256": {
            "official": sha256_file(bundle / "official.csv"),
            "table": sha256_file(data / "training/chemeleon_26_tdi/train_table.csv"),
            "folds": sha256_file(data / "training/chemeleon_26_tdi/folds.npy"),
        },
        "limitations": [
            "Canonical CheMeleon fold strata do not imply matching training sets for every member.",
            "OOF is relative to each member fine-tuning stage; "
            "pretrained label exposure is not independently audited.",
            "g26_tdi_armB has published blind predictions but no published OOF file.",
            "R25 blend coefficients were fitted on OOF rows; its fitted OOF score is optimistic.",
            "There is no independent five-fold final-R25 score in this audit.",
            "These are published third-party predictions, not models trained by this project.",
        ],
        "full_R25_historical_official_score": 0.4436,
        "training_launched": False,
        "submission_authorized": False,
    }
    write_json(output / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "bundle", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.bundle, args.output), indent=2))


if __name__ == "__main__":
    main()
