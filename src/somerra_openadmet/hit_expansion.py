"""Frozen, retrospective hit-expansion benchmark; never launches training or uploads."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import wasserstein_distance

from .constants import ACTIVITY_ENDPOINTS
from .cyp_evaluation import regression_metrics
from .data_sources.common import parent_smiles
from .download import sha256_file

PROTOCOL = "hit-expansion-close-analogs-v2"
REPEATS = tuple(range(42, 47))
SEED_TARGETS = tuple(e for e in ACTIVITY_ENDPOINTS if not e.startswith("CYP2D6"))
FINGERPRINT = "Morgan radius 2, 2048 bits, no chirality; standardized stereo-insensitive parent"
INTERPRETATION = (
    "Retrospective development evidence: prior adaptive exposure, sparse assays, "
    "blind-structure-informed design, and overlapping repeats; not independent test estimates."
)
SOURCES = {
    "challenge_design": "https://www.openadmet.org/blog/announcing-openadmets-cyp-inhibition-blind-challenge/",
    "assay_design": "https://openadmet.github.io/octant-cyp-inhib-blog-post/",
    "rules": "https://huggingface.co/spaces/openadmet/cyp-challenge/blob/main/config.py",
    "dataset": "https://huggingface.co/datasets/openadmet/cyp-challenge-train-test",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_hashes(hashes, base=None):
    for name, expected in hashes.items():
        path = Path(name) if base is None else Path(base) / name
        if sha256_file(path) != expected:
            raise ValueError(f"Changed artifact: {path}")


@lru_cache(maxsize=100_000)
def structure_key(smiles):
    return parent_smiles(smiles, stereo=False)


def identify(frame):
    frame = frame.copy()
    if frame.Molecule_Name.isna().any() or frame.Molecule_Name.duplicated().any():
        raise ValueError("Missing or duplicate molecule ID")
    column = "SMILES" if "SMILES" in frame else "parent"
    frame["expansion_parent"] = frame[column].map(structure_key)
    if "parent" in frame:
        if not frame.parent.map(structure_key).equals(frame.expansion_parent):
            raise ValueError("SMILES/parent identity mismatch")
    return frame.sort_values("Molecule_Name").reset_index(drop=True)


@lru_cache(maxsize=100_000)
def fingerprint(parent):
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(
        Chem.MolFromSmiles(parent)
    )


def select_analogs(seed_parents, candidate_parents, neighbors=10, minimum_similarity=0.45):
    """Structure-only boundary: no assay table or label mask is accepted here."""
    if neighbors < 1 or not 0 <= minimum_similarity <= 1:
        raise ValueError("Invalid expansion parameters")
    candidates = sorted(set(candidate_parents))
    if set(seed_parents) & set(candidates):
        raise ValueError("Seed/candidate overlap")
    fps = [fingerprint(p) for p in candidates]
    rows = []
    for seed in sorted(set(seed_parents)):
        scores = DataStructs.BulkTanimotoSimilarity(fingerprint(seed), fps)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: (-x[1], x[0]))
        for rank, (parent, similarity) in enumerate(ranked[:neighbors], 1):
            if similarity >= minimum_similarity:
                rows.append((seed, parent, rank, float(similarity)))
    return pd.DataFrame(rows, columns=["seed_parent", "analog_parent", "rank", "similarity"])


def assign_series(links):
    """Connected nomination families stay intact, even with multiple nominating seeds."""
    nodes = sorted(set(links.seed_parent) | set(links.analog_parent))
    uf = {p: p for p in nodes}

    def find(p):
        while uf[p] != p:
            uf[p] = uf[uf[p]]
            p = uf[p]
        return p

    for row in links.itertuples():
        a, b = sorted([find(row.seed_parent), find(row.analog_parent)])
        uf[b] = a
    groups = {}
    for p in sorted(set(links.analog_parent)):
        groups.setdefault(find(p), []).append(p)
    if len(groups) < 2:
        raise ValueError("Need at least two distinct analog families for validation and test")
    loads = {"validation": 0, "test": 0}
    assignments = {}
    for members in sorted(groups.values(), key=lambda g: (-len(g), min(g))):
        role = "validation" if loads["validation"] <= loads["test"] else "test"
        group_id = min(members)
        for p in members:
            assignments[p] = (group_id, role)
        loads[role] += len(members)
    return assignments


def make_assignments(official, blind, seed=42, hits=25, neighbors=10, minimum_similarity=0.45):
    if hits < 1:
        raise ValueError("hits must be positive")
    frame, blind = identify(official), identify(blind)
    blocked = set(blind.expansion_parent)
    parents = sorted(set(frame.expansion_parent) - blocked)
    seed_pool = set(np.random.default_rng(seed).permutation(parents)[: len(parents) // 2])
    frame["seed_pool"] = frame.expansion_parent.isin(seed_pool)
    # Only seed-pool labels enter hit selection. Candidate values/masks cannot affect it.
    seeds = []
    for ep in SEED_TARGETS:
        eligible = frame.loc[frame.seed_pool, ["Molecule_Name", "expansion_parent", ep]]
        eligible = eligible.loc[np.isfinite(eligible[ep])]
        selected = eligible.sort_values([ep, "Molecule_Name"], ascending=[False, True])
        selected = selected.drop_duplicates("expansion_parent").head(hits)
        if len(selected) != hits:
            raise ValueError(f"Fewer than {hits} finite seed hits for {ep}")
        for _, row in selected.iterrows():
            seeds.append((row.expansion_parent, row.Molecule_Name, ep, float(row[ep])))
    seeds = pd.DataFrame(
        seeds, columns=["seed_parent", "seed_id", "selection_endpoint", "seed_pIC50"]
    )
    links = select_analogs(
        seeds.seed_parent, set(parents) - seed_pool, neighbors, minimum_similarity
    )
    series = assign_series(links)
    frame["partition"] = frame.expansion_parent.map(
        lambda p: "excluded_blind" if p in blocked else series.get(p, (None, "train"))[1]
    )
    frame["series_id"] = frame.expansion_parent.map(lambda p: series.get(p, (None, None))[0])
    frame["selected_seed"] = frame.expansion_parent.isin(seeds.seed_parent)
    links["series_id"] = links.analog_parent.map(lambda p: series[p][0])
    links["partition"] = links.analog_parent.map(lambda p: series[p][1])
    if not frame.loc[frame.selected_seed, "partition"].eq("train").all():
        raise ValueError("Seed was held out")
    # Assignments intentionally contain no candidate labels, even for audit consumers.
    cols = [
        "Molecule_Name",
        "SMILES",
        "expansion_parent",
        "seed_pool",
        "selected_seed",
        "series_id",
        "partition",
    ]
    return frame[cols], seeds, links


def build_frames(official, pool, blind, assignments):
    official, pool, blind = identify(official), identify(pool), identify(blind)
    truth = official.set_index("Molecule_Name")
    if not set(truth.index) <= set(pool.Molecule_Name):
        raise ValueError("Pool is missing official records")
    matched = pool.set_index("Molecule_Name").loc[truth.index]
    if not matched.expansion_parent.equals(truth.expansion_parent):
        raise ValueError("Official/pool identity mismatch")
    if not np.allclose(
        matched[list(ACTIVITY_ENDPOINTS)],
        truth[list(ACTIVITY_ENDPOINTS)],
        equal_nan=True,
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError("Official/pool label mismatch")
    held = set(
        assignments.loc[assignments.partition.isin(["validation", "test"]), "expansion_parent"]
    )
    blocked = set(blind.expansion_parent)
    train = pool.loc[~pool.expansion_parent.isin(held | blocked)].copy()
    # Old split columns must not accidentally drive a trainer's row selection.
    train = train.drop(columns=["fold", "cluster", "partition"], errors="ignore")
    train["SMILES"] = train.Molecule_Name.map(truth.SMILES).fillna(train.expansion_parent)
    train["partition"] = "train"
    train["sample_weight"] = 1.0
    for ep in ACTIVITY_ENDPOINTS:
        for suffix in ["_conf_low", "_conf_high"]:
            train[ep + suffix] = train.Molecule_Name.map(truth[ep + suffix])
    frames = {"train": train}
    for role in ["validation", "test"]:
        ids = assignments.loc[assignments.partition == role, "Molecule_Name"]
        f = (
            truth.loc[ids]
            .reset_index()
            .drop(columns=["fold", "cluster", "partition"], errors="ignore")
        )
        f["partition"] = role
        f["series_id"] = f.Molecule_Name.map(assignments.set_index("Molecule_Name").series_id)
        f["sample_weight"] = 1.0
        frames[role] = f
    role_parents = {r: set(f.expansion_parent) for r, f in frames.items()}
    for role, parents in role_parents.items():
        if parents & blocked or any(parents & ps for r, ps in role_parents.items() if r != role):
            raise ValueError("Parent leakage between roles or into blind set")
    removed = pool.loc[pool.expansion_parent.isin(held | blocked)].copy()
    removed["reason"] = np.where(
        removed.expansion_parent.isin(blocked), "official_blind_parent", "evaluation_parent"
    )
    return frames, removed


def coverage(frame):
    result = {}
    for ep in ACTIVITY_ENDPOINTS:
        observed = np.isfinite(frame[ep].to_numpy(float))
        lo, hi = [frame[ep + s].to_numpy(float) for s in ["_conf_low", "_conf_high"]]
        valid_ci = np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
        n = int(observed.sum())
        result[ep] = {
            "n": n,
            "credible_intervals": int((observed & valid_ci).sum()),
            "underpowered": n < 20,
        }
    return result


def input_paths(source_manifest, workbench_root, workbench_hashes):
    source_manifest, workbench_root, workbench_hashes = (
        Path(p).resolve() for p in (source_manifest, workbench_root, workbench_hashes)
    )
    old = read_json(source_manifest)
    protected = dict(old["input_sha256"])
    protected.update({str(source_manifest.parent / n): h for n, h in old["file_sha256"].items()})
    protected[str(source_manifest.resolve())] = sha256_file(source_manifest)
    wb = read_json(workbench_hashes)
    protected.update({str(Path(workbench_root) / n): h for n, h in wb["sha256"].items()})
    protected[str(workbench_hashes.resolve())] = sha256_file(workbench_hashes)
    inputs = {}
    for name in ["official.csv", "pool.csv", "test.csv"]:
        matches = [p for p in old["input_sha256"] if Path(p).name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one pinned input named {name}")
        inputs[name] = matches[0]
    source_metadata = Path(inputs["pool.csv"]).parent / "manifest.json"
    protected[str(source_metadata)] = sha256_file(source_metadata)
    verify_hashes(protected)
    return inputs, protected, source_metadata


def prepare(source_manifest, workbench_root, workbench_hashes, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory; saved splits are immutable")
    inputs, protected, source_metadata = input_paths(
        source_manifest, workbench_root, workbench_hashes
    )
    official, pool, blind = [
        pd.read_csv(inputs[n]) for n in ["official.csv", "pool.csv", "test.csv"]
    ]
    wb = pd.read_csv(Path(workbench_root) / "data/mt_aux8.csv")
    if set(wb.molecule_name) != set(official.Molecule_Name):
        raise ValueError("Workbench/official identity mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".hit-expansion-", dir=output.parent))
    try:
        details = {}
        for seed in REPEATS:
            folder = staging / f"repeat-{seed}"
            folder.mkdir()
            assignments, seeds, links = make_assignments(official, blind, seed)
            assignments["workbench_fold"] = assignments.Molecule_Name.map(
                wb.set_index("molecule_name").fold
            )
            frames, removed = build_frames(official, pool, blind, assignments)
            artifacts = {
                **frames,
                "assignments": assignments,
                "seed-hits": seeds,
                "analog-links": links,
                "excluded-training-records": removed,
            }
            for name, frame in artifacts.items():
                frame.to_csv(folder / f"{name}.csv", index=False)
            details[str(seed)] = {
                "rows": {r: len(f) for r, f in frames.items()},
                "coverage": {r: coverage(f) for r, f in frames.items()},
                "families": int(links.series_id.nunique()),
                "held_parents": int(links.analog_parent.nunique()),
                "excluded_training_records": len(removed),
                "excluded_observations": {
                    c: int(removed[c].notna().sum()) for c in read_json(source_metadata)["tasks"]
                },
            }
        # Snapshot scientific code and source attribution privately, without altering originals.
        for name in ["hit_expansion.py", "cyp_evaluation.py", "metrics.py", "constants.py"]:
            shutil.copyfile(Path(__file__).parent / name, staging / name)
        shutil.copyfile(Path(__file__).parent / "data_sources/common.py", staging / "identity.py")
        shutil.copyfile(source_metadata, staging / "source-provenance.json")
        verify_hashes(protected)
        manifest = {
            "protocol": PROTOCOL,
            "repeats": list(REPEATS),
            "inputs": inputs,
            "protected_sha256": protected,
            "file_sha256": {
                str(p.relative_to(staging)): sha256_file(p)
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            },
            "parameters": {
                "seed_pool_fraction": 0.5,
                "hits_per_endpoint": 25,
                "neighbors_per_seed": 10,
                "minimum_tanimoto": 0.45,
                "seed_targets": list(SEED_TARGETS),
                "fingerprint": FINGERPRINT,
            },
            "versions": {
                "rdkit": rdBase.rdkitVersion,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "roles": {
                "train": "fit fresh models; fit scalers and task weights here only",
                "validation": "checkpoint and development selection only",
                "test": "score frozen raw predictions; never fit or select checkpoints",
            },
            "details": details,
            "interpretation": INTERPRETATION,
            "structural_adaptation": (
                "Fixed 0.45 cutoff chosen after blind-structure comparison; "
                "no blind labels or leaderboard scores used."
            ),
            "limitations": [
                "Existing measured library substitutes for purchased analogs.",
                "Official fingerprint and actual seed IDs are not verified.",
                "Direct pIC50 seed ranking is an explicit reconstruction assumption.",
                "Missing assays are not negatives; repeats are not independent.",
                "Sparse coverage may make checkpoint selection and ST-RAE unstable.",
            ],
            "sources": SOURCES,
            "training_launched": False,
            "submission_authorized": False,
            "status": "prepared_exploratory_underpowered",
            "calibration": "none",
        }
        write_json(staging / "manifest.json", manifest)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_bundle(root):
    root = Path(root)
    m = read_json(root / "manifest.json")
    if m["protocol"] != PROTOCOL or m["repeats"] != list(REPEATS):
        raise ValueError("Unexpected benchmark contract")
    verify_hashes(m["file_sha256"], root)
    verify_hashes(m["protected_sha256"])
    return m


@lru_cache(maxsize=100_000)
def scaffold(parent):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(parent))


def nearest_rows(query, reference, comparison, repeat=None):
    parents = sorted(set(reference.expansion_parent))
    if not parents or query.empty:
        raise ValueError("Empty similarity query or reference")
    fps = [fingerprint(p) for p in parents]
    rows = []
    for row in query.itertuples():
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint(row.expansion_parent), fps)
        )
        best = int(np.argmax(similarities))
        rows.append(
            {
                "repeat": repeat,
                "comparison": comparison,
                "Molecule_Name": row.Molecule_Name,
                "parent": row.expansion_parent,
                "nearest_parent": parents[best],
                "similarity": float(similarities[best]),
            }
        )
    return rows


def audit(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    m = verify_bundle(root)
    official, blind = [identify(pd.read_csv(m["inputs"][n])) for n in ["official.csv", "test.csv"]]
    blocked = set(blind.expansion_parent)
    official = official.loc[~official.expansion_parent.isin(blocked)]
    comparisons = nearest_rows(blind, official, "official_blind_to_official_training")
    exposure, families, membership = [], [], []
    checks = {}
    for seed in REPEATS:
        folder = root / f"repeat-{seed}"
        a, links = [pd.read_csv(folder / n) for n in ["assignments.csv", "analog-links.csv"]]
        frames = {r: pd.read_csv(folder / f"{r}.csv") for r in ["train", "validation", "test"]}
        official_train = official.loc[official.Molecule_Name.isin(frames["train"].Molecule_Name)]
        seeds = official.loc[
            official.expansion_parent.isin(pd.read_csv(folder / "seed-hits.csv").seed_parent)
        ]
        parents = {r: set(f.expansion_parent) for r, f in frames.items()}
        if any(p & blocked for p in parents.values()) or any(
            parents[x] & parents[y]
            for x, y in [("train", "validation"), ("train", "test"), ("validation", "test")]
        ):
            raise ValueError("Parent isolation audit failed")
        if not a.loc[a.selected_seed, "partition"].eq("train").all():
            raise ValueError("Seed isolation audit failed")
        if links.groupby("series_id").partition.nunique().max() != 1:
            raise ValueError("Series isolation audit failed")
        expected_series = assign_series(links)
        for row in links.itertuples():
            if expected_series[row.analog_parent] != (row.series_id, row.partition):
                raise ValueError("Nomination family assignment mismatch")
        for row in a.itertuples():
            if row.partition in ["validation", "test"]:
                if expected_series.get(row.expansion_parent) != (row.series_id, row.partition):
                    raise ValueError("Assignment/nomination mismatch")
        membership.extend(
            {"repeat": seed, "parent": row.expansion_parent, "role": row.partition}
            for row in a.drop_duplicates("expansion_parent").itertuples()
        )
        checks[str(seed)] = {
            "parent_overlap": 0,
            "blind_overlap": 0,
            "seeds_in_train": True,
            "coverage": {r: coverage(f) for r, f in frames.items()},
            "scaffold_overlap": {},
        }
        train_scaffolds = {scaffold(p) for p in parents["train"]}
        for role in ["validation", "test"]:
            f = frames[role]
            for label, ref in [
                ("official_training", official_train),
                ("all_training", frames["train"]),
                ("selected_seeds", seeds),
            ]:
                comparisons.extend(nearest_rows(f, ref, f"{role}_to_{label}", seed))
            checks[str(seed)]["scaffold_overlap"][role] = sum(
                scaffold(p) in train_scaffolds for p in parents[role]
            )
            exposure.extend(
                {"repeat": seed, "role": role, "parent": p} for p in sorted(parents[role])
            )
        for (series, role), group in links.groupby(["series_id", "partition"]):
            families.append(
                {
                    "repeat": seed,
                    "series_id": series,
                    "role": role,
                    "analog_parents": group.analog_parent.nunique(),
                    "seed_parents": group.seed_parent.nunique(),
                }
            )
    nn = pd.DataFrame(comparisons)
    target = nn.loc[nn.comparison == "official_blind_to_official_training", "similarity"]
    summaries = []
    for (repeat, comparison), group in nn.groupby(["repeat", "comparison"], dropna=False):
        s = group.similarity
        summaries.append(
            {
                "repeat": None if pd.isna(repeat) else int(repeat),
                "comparison": comparison,
                "rows": len(group),
                "unique_parents": group.parent.nunique(),
                "mean": float(s.mean()),
                "p10": float(s.quantile(0.1)),
                "median": float(s.median()),
                "p90": float(s.quantile(0.9)),
                "wasserstein_to_official": float(wasserstein_distance(target, s)),
            }
        )
    exp = pd.DataFrame(exposure)
    counts = exp.groupby("parent").agg(
        repeats=("repeat", "nunique"), appearances=("repeat", "size")
    )
    membership_frame = pd.DataFrame(membership)
    role_matrix = membership_frame.pivot(index="parent", columns="repeat", values="role")
    reused_for_training = role_matrix.loc[counts.index].eq("train").any(axis=1)
    result = {
        "interpretation": INTERPRETATION,
        "checks": checks,
        "unique_evaluation_parents": len(counts),
        "evaluation_appearances": len(exp),
        "parents_in_multiple_repeats": int((counts.repeats > 1).sum()),
        "evaluation_parents_used_for_training_in_other_repeats": int(reused_for_training.sum()),
        "summaries": summaries,
        "historical_hashes_verified": True,
        "audit_implementation_sha256": sha256_file(Path(__file__)),
        "benchmark_manifest_sha256": sha256_file(root / "manifest.json"),
    }
    output.mkdir(parents=True)
    nn.to_csv(output / "nearest-neighbors.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "similarity-summary.csv", index=False)
    pd.DataFrame(families).to_csv(output / "families.csv", index=False)
    exp.to_csv(output / "exposure.csv", index=False)
    counts.to_csv(output / "unique-parent-exposure.csv")
    role_matrix.to_csv(output / "cross-repeat-roles.csv")
    write_json(output / "audit.json", result)
    return result


def score_predictions(truth, predictions):
    if predictions.Molecule_Name.isna().any() or predictions.Molecule_Name.duplicated().any():
        raise ValueError("Missing or duplicate prediction IDs")
    if set(predictions.Molecule_Name) != set(truth.Molecule_Name):
        raise ValueError("Predictions must cover exactly every partition row")
    aligned = predictions.set_index("Molecule_Name").loc[truth.Molecule_Name]
    endpoints = {}
    for ep in ACTIVITY_ENDPOINTS:
        pred = aligned[ep].to_numpy(float)
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite predictions, including on unlabeled rows")
        n = int(np.isfinite(truth[ep].to_numpy(float)).sum())
        try:
            metrics = regression_metrics(truth, pred, ep)
            error = None
        except ValueError as exc:
            metrics, error = None, str(exc)
        endpoints[ep] = {
            "n": n,
            "underpowered": n < 20,
            "metrics": metrics,
            "undefined_reason": error,
        }
    values = [v["metrics"]["ST-RAE"] for v in endpoints.values() if v["metrics"] is not None]
    return {
        "endpoints": endpoints,
        "MA-ST-RAE": float(np.mean(values)) if len(values) == len(ACTIVITY_ENDPOINTS) else None,
    }


def evaluate(root, predictions, fit_records, output):
    """One raw prediction CSV and fit record per repeat; all five are required."""
    root, predictions, fit_records, output = map(Path, (root, predictions, fit_records, output))
    if output.exists():
        raise FileExistsError(output)
    m = verify_bundle(root)
    for name in ["cyp_evaluation.py", "metrics.py", "constants.py"]:
        if sha256_file(Path(__file__).parent / name) != m["file_sha256"][name]:
            raise ValueError(f"Scoring implementation changed: {name}")
    results, provenance, test_parents = {}, {}, set()
    appearances = 0
    for seed in REPEATS:
        name = f"repeat-{seed}"
        record_path = fit_records / f"{name}.json"
        prediction_path = predictions / f"{name}.csv"
        record = read_json(record_path)
        for role in ["train", "validation"]:
            if record.get(f"{role}_sha256") != m["file_sha256"][f"{name}/{role}.csv"]:
                raise ValueError("Fit record references a different split")
        if (
            record.get("fresh_fit") is not True
            or record.get("test_used_for_fitting_or_selection") is not False
        ):
            raise ValueError("Fresh-fit and test-isolation assertions required")
        if (
            record.get("calibration") != "none"
            or record.get("scalers_and_task_weights_fit_on") != "train"
        ):
            raise ValueError("Raw predictions and training-only fitted preprocessing required")
        checkpoints = record.get("checkpoint_sha256", {})
        if not checkpoints:
            raise ValueError("Checkpoint paths and hashes required")
        verify_hashes(checkpoints, record_path.parent)
        truth = pd.read_csv(root / name / "test.csv")
        results[str(seed)] = score_predictions(truth, pd.read_csv(prediction_path))
        test_parents.update(truth.expansion_parent)
        appearances += len(truth)
        provenance[str(seed)] = {
            "fit_record_sha256": sha256_file(record_path),
            "prediction_sha256": sha256_file(prediction_path),
        }
    values = [r["MA-ST-RAE"] for r in results.values()]
    complete = all(v is not None for v in values)
    result = {
        "interpretation": INTERPRETATION,
        "repeats": results,
        "macro_across_all_five": {
            "mean": float(np.mean(values)),
            "min": min(values),
            "max": max(values),
        }
        if complete
        else None,
        "undefined_macro_repeats": [s for s, r in results.items() if r["MA-ST-RAE"] is None],
        "unique_test_parents": len(test_parents),
        "test_row_appearances": appearances,
        "provenance": provenance,
        "benchmark_manifest_sha256": sha256_file(root / "manifest.json"),
        "evaluation_implementation_sha256": sha256_file(Path(__file__)),
        "provenance_limit": (
            "Fit records are trainer attestations; hashes establish artifact identity, "
            "not proof of training behavior."
        ),
    }
    output.mkdir(parents=True)
    write_json(output / "evaluation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ["source-manifest", "workbench-root", "workbench-hashes", "output"]:
        p.add_argument(f"--{name}", required=True)
    p = sub.add_parser("audit")
    for name in ["root", "output"]:
        p.add_argument(f"--{name}", required=True)
    p = sub.add_parser("evaluate")
    for name in ["root", "predictions", "fit-records", "output"]:
        p.add_argument(f"--{name}", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = {"prepare": prepare, "audit": audit, "evaluate": evaluate}[command](**args)
    print(
        json.dumps(
            {
                "command": command,
                "output": args["output"],
                "status": result.get("status", "complete"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
