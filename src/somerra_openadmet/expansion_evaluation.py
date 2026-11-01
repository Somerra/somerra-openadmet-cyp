"""Declared hit-expansion proxy; separate from the preserved Workbench folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from .briford import calibrated as historical_calibrated
from .constants import ACTIVITY_ENDPOINTS
from .cyp_evaluation import regression_metrics
from .cyp_research import evaluation_rows
from .data_sources.common import parent_smiles
from .download import sha256_file

SEED_TARGETS = tuple(e for e in ACTIVITY_ENDPOINTS if not e.startswith("CYP2D6"))
ROLES = ("validation", "calibration", "test")


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def structure_key(smiles):
    """Stereo-insensitive parent: all equivalent records share one partition."""
    mol = Chem.MolFromSmiles(parent_smiles(smiles))
    if mol is None:
        raise ValueError("Invalid molecule")
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def make_assignments(official, blind, seed=42, hits=25, neighbors=10, min_similarity=0.0):
    """Hit labels are consulted only in a structure-randomized training seed pool.

    Candidate analog selection uses fingerprints, never candidate labels. Shared analogs
    are assigned to their closest nominating seed; all analogs of that anchor share a role.
    """
    if hits < 1 or neighbors < 1 or not 0 <= min_similarity <= 1:
        raise ValueError("Invalid expansion parameters")
    if official.Molecule_Name.isna().any() or official.Molecule_Name.duplicated().any():
        raise ValueError("Official identities must be unique")
    frame = official.sort_values("Molecule_Name").copy().reset_index(drop=True)
    frame["expansion_parent"] = frame.SMILES.map(structure_key)
    blocked = set(blind.SMILES.map(structure_key))
    parents = sorted(set(frame.expansion_parent) - blocked)
    if len(parents) < 12:
        raise ValueError("Insufficient molecular parents")
    shuffled = np.random.default_rng(seed).permutation(parents)
    seed_pool = set(shuffled[: len(shuffled) // 2])
    candidates = sorted(set(parents) - seed_pool)
    frame["partition"] = np.where(frame.expansion_parent.isin(blocked), "excluded_blind", "train")
    frame["seed_pool"] = frame.expansion_parent.isin(seed_pool)
    seed_records = []
    for ep in SEED_TARGETS:
        # Selection is restricted to seed-pool observations; held-out labels are not consulted.
        hit_rows = (
            frame.loc[frame.seed_pool & frame[ep].notna()]
            .sort_values([ep, "Molecule_Name"], ascending=[False, True])
            .drop_duplicates("expansion_parent")
            .head(hits)
        )
        if len(hit_rows) != hits:
            raise ValueError(f"Fewer than {hits} seed hits for {ep}")
        for _, row in hit_rows.iterrows():
            seed_records.append(
                {
                    "seed_parent": row.expansion_parent,
                    "seed_id": row.Molecule_Name,
                    "selection_endpoint": ep,
                    "seed_pIC50": float(row[ep]),
                }
            )
    selected_seeds = sorted({r["seed_parent"] for r in seed_records})
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {
        p: generator.GetFingerprint(Chem.MolFromSmiles(p)) for p in [*selected_seeds, *candidates]
    }
    candidate_fps = [fps[p] for p in candidates]
    links = []
    for anchor in selected_seeds:
        sim = DataStructs.BulkTanimotoSimilarity(fps[anchor], candidate_fps)
        ranked = sorted(zip(candidates, sim, strict=True), key=lambda x: (-x[1], x[0]))
        for parent, similarity in ranked[:neighbors]:
            if similarity >= min_similarity:
                links.append(
                    {
                        "seed_parent": anchor,
                        "analog_parent": parent,
                        "similarity": float(similarity),
                    }
                )
    if not links:
        raise ValueError("No analogs meet the predeclared threshold")
    link_frame = pd.DataFrame(links)
    # Pick a deterministic anchor for candidates nominated by multiple hits.
    anchors = link_frame.sort_values(["similarity", "seed_parent"], ascending=[False, True])
    anchors = anchors.drop_duplicates("analog_parent").set_index("analog_parent")
    groups = sorted(set(anchors.seed_parent))
    if len(groups) < 3:
        raise ValueError("Fewer than three analog series")
    roles = {}
    for i, anchor in enumerate(np.random.default_rng(seed + 1).permutation(groups)):
        roles[anchor] = ROLES[i % 3]
    role_by_parent = anchors.seed_parent.map(roles)
    mask = frame.expansion_parent.isin(anchors.index)
    frame.loc[mask, "partition"] = frame.loc[mask, "expansion_parent"].map(role_by_parent)
    frame["anchor_parent"] = frame.expansion_parent.map(anchors.seed_parent)
    frame["anchor_similarity"] = frame.expansion_parent.map(anchors.similarity)
    frame["selected_seed"] = frame.expansion_parent.isin(selected_seeds)
    assert set(frame.loc[frame.selected_seed, "partition"]) == {"train"}
    assert frame.groupby("expansion_parent").partition.nunique().max() == 1
    return frame, pd.DataFrame(seed_records), link_frame


def prepare(source, workbench, output, seed=42, hits=25, neighbors=10, min_similarity=0.0):
    source, workbench, output = map(Path, (source, workbench, output))
    if output.exists():
        raise FileExistsError("Choose a new output directory; never replace a saved split")
    inputs = {}
    for base, names in [
        (source, ["official.csv", "pool.csv", "test.csv"]),
        (workbench, ["mt_aux8.csv"]),
    ]:
        manifest = json.loads((base / "manifest.json").read_text())
        for name in names:
            digest = sha256_file(base / name)
            if digest != manifest["file_sha256"][name]:
                raise ValueError(f"Changed input: {base / name}")
            inputs[str((base / name).resolve())] = digest
    official, pool, blind = [pd.read_csv(source / f"{n}.csv") for n in ["official", "pool", "test"]]
    wb = pd.read_csv(workbench / "mt_aux8.csv")
    if set(official.Molecule_Name) != set(wb.molecule_name):
        raise ValueError("Workbench and official identities differ")
    if pool.Molecule_Name.isna().any() or pool.Molecule_Name.duplicated().any():
        raise ValueError("Duplicate or missing expanded-pool identity")
    indexed = pool.set_index("Molecule_Name").loc[official.Molecule_Name]
    if not np.allclose(
        indexed[list(ACTIVITY_ENDPOINTS)],
        official[list(ACTIVITY_ENDPOINTS)],
        equal_nan=True,
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError("Pool/official primary label mismatch")
    if not np.array_equal(indexed.parent.to_numpy(), official.parent.to_numpy()):
        raise ValueError("Pool/official parent mismatch")
    assigned, seeds, links = make_assignments(
        official, blind, seed, hits, neighbors, min_similarity
    )
    assigned["workbench_fold"] = assigned.Molecule_Name.map(wb.set_index("molecule_name").fold)
    # Remove every alias of a held-out or blind parent from every training source.
    pool["expansion_parent"] = pool.parent.map(structure_key)
    held = set(assigned.loc[assigned.partition.isin(ROLES), "expansion_parent"])
    blocked = set(blind.SMILES.map(structure_key)) | set(blind.parent.map(structure_key))
    train = pool.loc[~pool.expansion_parent.isin(held | blocked)].copy()
    truth = official.set_index("Molecule_Name")
    train["SMILES"] = train.Molecule_Name.map(truth.SMILES).fillna(train.parent)
    bounds = [ep + s for ep in ACTIVITY_ENDPOINTS for s in ["_conf_low", "_conf_high"]]
    for col in bounds:
        train[col] = train.Molecule_Name.map(truth[col])
    train["sample_weight"] = 1.0
    frames = {"train": train}
    for role in ROLES:
        frames[role] = evaluation_rows(official, pool, assigned, role)
    role_parents = {role: set(f.expansion_parent) for role, f in frames.items()}
    for role, parents in role_parents.items():
        if parents & blocked:
            raise ValueError("Blind parent in evaluation bundle")
        for other in role_parents:
            if role != other and parents & role_parents[other]:
                raise ValueError("Parent leakage between partitions")
    counts = {
        role: {ep.split("_")[0]: int(f[ep].notna().sum()) for ep in ACTIVITY_ENDPOINTS}
        for role, f in frames.items()
    }
    issues = []
    for role in ROLES:
        for ep in ACTIVITY_ENDPOINTS:
            f = frames[role]
            if f[ep].notna().sum() < 20:
                issues.append(f"{role}/{ep}: fewer than 20 observed labels")
            try:
                regression_metrics(f, np.full(len(f), 4.0), ep)
            except ValueError as exc:
                issues.append(f"{role}/{ep}: {exc}")
    output.mkdir(parents=True)
    all_frames = {**frames, "assignments": assigned, "seed-hits": seeds, "analog-links": links}
    for name, f in all_frames.items():
        f.to_csv(output / f"{name}.csv", index=False)
    manifest = {
        "protocol": "hit-expansion-proxy-v1",
        "seed": seed,
        "hits_per_selection_endpoint": hits,
        "neighbors_per_seed": neighbors,
        "minimum_tanimoto": min_similarity,
        "seed_pool_fraction": 0.5,
        "fingerprint": "Morgan radius 2, 2048 bits, no chirality; standardized parent",
        "rdkit": rdBase.rdkitVersion,
        "anchor_similarity_quantiles": {
            str(q): float(
                assigned.loc[assigned.partition.isin(ROLES), "anchor_similarity"].quantile(q)
            )
            for q in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        },
        "seed_targets": list(SEED_TARGETS),
        "input_sha256": inputs,
        "implementation_sha256": sha256_file(Path(__file__)),
        "file_sha256": {f"{n}.csv": sha256_file(output / f"{n}.csv") for n in all_frames},
        "rows": {r: len(f) for r, f in frames.items()},
        "label_counts": counts,
        "analog_parent_count": len(held),
        "unique_seed_count": seeds.seed_parent.nunique(),
        "roles": {
            "train": "fit models and scalers; contains all selected seed hits",
            "validation": "monitor and select checkpoints",
            "calibration": "fit declared calibration after checkpoint freezing",
            "test": "score frozen raw/calibrated strategies; never fit parameters",
        },
        "limitations": [
            "Proxy uses existing measured library, not purchased catalog analogs.",
            "Only training seed-pool labels select hits; analog labels never select holdouts.",
            "Sparse endpoint coverage remains; missing labels are not negative labels.",
            "Nearest-anchor groups are disjoint across roles but may remain chemically similar.",
            "Prior exposure exists; fresh fits required, not historical checkpoints.",
            "Repeated strategy feedback makes test scores adaptive development evidence.",
            "Blind structures exclude overlaps; no blind scores/labels select splits.",
        ],
        "readiness_issues": issues,
        "status": "ready_for_fresh_fits" if not issues else "coverage_review_required",
        "training_launched": False,
        "submission_authorized": False,
    }
    # Numpy scalar converted explicitly to maintain a portable JSON artifact.
    manifest["unique_seed_count"] = int(manifest["unique_seed_count"])
    (output / "preparation-implementation.py").write_bytes(Path(__file__).read_bytes())
    save(output / "manifest.json", manifest)
    return manifest


def aligned_predictions(path, truth, endpoint):
    frame = pd.read_csv(path)
    if frame.Molecule_Name.isna().any() or frame.Molecule_Name.duplicated().any():
        raise ValueError("Duplicate/missing prediction IDs")
    if set(frame.Molecule_Name) != set(truth.Molecule_Name):
        raise ValueError("Predictions must cover the exact partition, including unlabeled rows")
    pred = frame.set_index("Molecule_Name").loc[truth.Molecule_Name, endpoint].to_numpy(float)
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite predictions")
    return pred


def evaluate(root, calibration_predictions, test_predictions, fit_record, output):
    """Score raw, own-data affine and authorized historical Workbench transforms."""
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("Evaluation result already exists")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "ready_for_fresh_fits":
        raise ValueError("Resolve coverage before evaluation")
    for name, digest in manifest["file_sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Changed split: {name}")
    record = json.loads(Path(fit_record).read_text())
    # Provenance assertion from the trainer; hashes prevent accidental reuse of old fits.
    if (
        record.get("train_sha256") != manifest["file_sha256"]["train.csv"]
        or record.get("validation_sha256") != manifest["file_sha256"]["validation.csv"]
    ):
        raise ValueError("Fit must reference this split's train and validation files")
    if record.get("calibration_or_test_used_for_training_or_checkpoint_selection") is not False:
        raise ValueError("Training isolation assertion required")
    if set(record.get("checkpoint_sha256", {})) != {ep.split("_")[0] for ep in ACTIVITY_ENDPOINTS}:
        raise ValueError("Record a checkpoint hash for each CYP")
    for digest in record["checkpoint_sha256"].values():
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid checkpoint hash")
    cal, test = [pd.read_csv(root / f"{r}.csv") for r in ["calibration", "test"]]
    result = {
        "interpretation": "development evidence; prior exposure and adaptive strategy selection",
        "historical_calibration": (
            "User authorized borrowed constants 2026-11-01; leaderboard-informed, "
            "not training-only; no organizer-specific clearance asserted."
        ),
        "fit_record_sha256": sha256_file(Path(fit_record)),
        "endpoints": {},
    }
    exported = test[["Molecule_Name", "SMILES"]].copy()
    historical_export = exported.copy()
    raw_export = exported.copy()
    for ep in ACTIVITY_ENDPOINTS:
        x = aligned_predictions(calibration_predictions, cal, ep)
        t = aligned_predictions(test_predictions, test, ep)
        observed = cal[ep].notna().to_numpy()
        y, xx = cal.loc[observed, ep].to_numpy(float), x[observed]
        if len(y) < 20 or np.std(xx) <= 0:
            raise ValueError("Insufficient calibration labels or constant predictions")
        slope = float(np.cov(xx, y, ddof=1)[0, 1] / np.var(xx, ddof=1))
        if slope <= 0:
            raise ValueError("Nonpositive fitted slope; review recipe rather than invert ranking")
        intercept = float(y.mean() - slope * xx.mean())
        pred = np.maximum(intercept + slope * t, 1.0)
        exported[ep] = pred
        historical, historical_params = historical_calibrated(
            t, ep.split("_")[0].lower(), float(np.corrcoef(xx, y)[0, 1])
        )
        historical_export[ep] = historical
        raw_export[ep] = t
        result["endpoints"][ep] = {
            "parameters": {"slope": slope, "intercept": intercept, "floor": 1.0, "n": len(y)},
            "raw": regression_metrics(test, t, ep),
            "calibrated": regression_metrics(test, pred, ep),
            "historical_workbench": regression_metrics(test, historical, ep),
            "historical_parameters": historical_params,
        }
    result["macro_ST_RAE"] = {
        v: float(np.mean([r[v]["ST-RAE"] for r in result["endpoints"].values()]))
        for v in ["raw", "calibrated", "historical_workbench"]
    }
    result["prediction_sha256"] = {
        "calibration": sha256_file(Path(calibration_predictions)),
        "test": sha256_file(Path(test_predictions)),
    }
    output.mkdir(parents=True)
    exported.to_csv(output / "calibrated-test-predictions.csv", index=False)
    historical_export.to_csv(output / "historical-workbench-test-predictions.csv", index=False)
    raw_export.to_csv(output / "raw-test-predictions.csv", index=False)
    save(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--source", required=True)
    p.add_argument("--workbench", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hits", type=int, default=25)
    p.add_argument("--neighbors", type=int, default=10)
    p.add_argument("--min-similarity", type=float, default=0.0)
    p = subs.add_parser("evaluate")
    p.add_argument("--root", required=True)
    p.add_argument("--calibration-predictions", required=True)
    p.add_argument("--test-predictions", required=True)
    p.add_argument("--fit-record", required=True)
    p.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    print(json.dumps((prepare if command == "prepare" else evaluate)(**args), indent=2))


if __name__ == "__main__":
    main()
