"""Pinned Scigantic armB data adapted to official train/validation boundaries."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ACTIVITY_ENDPOINTS, TEST_FILE
from .data import load_test
from .data_sources.common import parent_smiles, write_json
from .data_sources.scigantic import GROUPS, REVISION
from .download import sha256_file


def task_mapping(columns):
    mapping = {}
    for c in columns:
        if c.endswith("_pic50_direct_inhibition"):
            mapping[c] = c.replace(c.split("_")[0], c.split("_")[0].upper(), 1)
        elif c.endswith("_pic50_tdi_condition"):
            mapping[c] = c.split("_")[0].upper() + "_pIC50_TDI_condition"
        elif c.endswith("_log2fc"):
            mapping[c] = c.split("_")[0].upper() + "_log2fc_4.9504951e-05M"
    return mapping


def prepare(source, prepared, auxiliary, raw, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError("Unpinned reference revision")
    paths = [
        Path(source) / "data/training/chemeleon_26_tdi" / x
        for x in ["train_table.csv", "weights.npy"]
    ]
    for p in paths:
        if subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--", str(p.relative_to(source))],
            text=True,
        ).strip():
            raise ValueError("Modified reference data")
    table = pd.read_csv(paths[0])
    weights = np.load(paths[1], allow_pickle=False)
    if len(weights) != len(table) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid reference sample weights")
    if table.Molecule_Name.duplicated().any():
        raise ValueError("Ambiguous upstream record identities")
    table["sample_weight"] = weights
    structures = {s: parent_smiles(s) for s in sorted(table.smiles.unique())}
    table["parent"] = table.smiles.map(structures)
    table = table.rename(columns=task_mapping(table.columns))
    official = pd.read_csv(Path(prepared) / "training.csv")
    test = load_test(raw)
    test["parent"] = test.SMILES.map(parent_smiles)
    official_ids = set(official.Molecule_Name)
    by_id = table.set_index("Molecule_Name")
    if official_ids - set(by_id.index):
        raise ValueError("Official records lack upstream weight mapping")
    mapped = by_id.loc[official.Molecule_Name, "parent"].to_numpy()
    if not np.array_equal(mapped, official.parent.to_numpy()):
        raise ValueError("Record identity/parent mismatch")
    aux = pd.read_csv(Path(auxiliary) / "assays.csv")
    groups = {
        "single": sorted(t for t in aux.task.unique() if "_log2fc_" in t),
        "tdi": sorted(t for t in aux.task.unique() if "_pIC50_TDI_condition" in t),
        **GROUPS,
    }
    tasks = list(ACTIVITY_ENDPOINTS) + sum(groups.values(), [])
    # Preserve each upstream record and its weight. Never collapse different weights by parent.
    collisions = int((table.groupby("parent").sample_weight.nunique() > 1).sum())
    for t in ACTIVITY_ENDPOINTS:
        table[t] = table.Molecule_Name.map(official.set_index("Molecule_Name")[t])
    wide = aux[aux.task.isin(groups["single"] + groups["tdi"])].pivot_table(
        index="parent", columns="task", values="value", aggfunc="median"
    )
    missing = sorted(set(wide.index) - set(table.parent))
    if missing:
        extra = pd.DataFrame(
            {
                "Molecule_Name": ["official_aux:" + str(i) for i in range(len(missing))],
                "parent": missing,
                "sample_weight": 1.0,
            }
        )
        table = pd.concat([table, extra], ignore_index=True)
    for t in groups["single"] + groups["tdi"]:
        table[t] = table.parent.map(wide[t])
    blind = set(test.parent)
    if set(official.parent) & blind:
        raise ValueError("Official training overlaps test")
    excluded = table.parent.isin(blind)
    table = table.loc[~excluded, ["Molecule_Name", "parent", "sample_weight", *tasks]]
    parents = np.asarray(sorted(official.parent.unique()))
    perm = np.random.default_rng(42).permutation(parents)
    val_parents = set(perm[: int(np.ceil(len(parents) * 0.2))])
    official["partition"] = np.where(official.parent.isin(val_parents), "validation", "train")
    counts = official[official.partition == "validation"][list(ACTIVITY_ENDPOINTS)].count()
    if (counts < 100).any():
        raise ValueError("Insufficient validation endpoint coverage")
    output.mkdir(parents=True)
    official.to_csv(output / "official.csv", index=False)
    table.to_csv(output / "pool.csv", index=False)
    test.to_csv(output / "test.csv", index=False)
    manifest = {
        "reference_revision": revision,
        "groups": groups,
        "tasks": tasks,
        "seed": 42,
        "exclusion_policy": "exact_parent",
        "weight_mapping": "unique upstream record ID; duplicate parents retained as records",
        "parents_with_distinct_weights": collisions,
        "added_neutral_auxiliary_parents": len(missing),
        "excluded_test_records": int(excluded.sum()),
        "validation_label_counts": counts.to_dict(),
        "partition_rows": official.partition.value_counts().to_dict(),
        "upstream_sha256": {p.name: sha256_file(p) for p in paths},
        "official_training_sha256": sha256_file(Path(prepared) / "training.csv"),
        "official_auxiliary_sha256": sha256_file(Path(auxiliary) / "assays.csv"),
        "test_sha256": sha256_file(Path(raw) / TEST_FILE),
        "file_sha256": {
            n: sha256_file(output / n) for n in ["official.csv", "pool.csv", "test.csv"]
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_frames(bundle, recipe="full", final=False):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, digest in manifest["file_sha256"].items():
        if sha256_file(bundle / name) != digest:
            raise ValueError("Changed prepared bundle")
    official = pd.read_csv(bundle / "official.csv")
    if manifest.get("protocol") in {"scaffold-five-fold-v1", "scaffold-single-v1"}:
        from .scaffold_ensemble import load_fold_frames

        return load_fold_frames(bundle, manifest, recipe, final=final)
    if manifest.get("protocol") == "submission-train-plus-validation-v1":
        if final:
            raise ValueError("Submission selection must retain its validation parents")
        expected = official.original_partition.map(
            {"train": "train", "validation": "train", "internal_test": "validation"}
        )
        if (
            set(official.original_partition) != {"train", "validation", "internal_test"}
            or not expected.equals(official.partition)
            or official.groupby("parent").partition.nunique().max() != 1
            or manifest["exclusion_policy"] != "exact_parent"
        ):
            raise ValueError("Invalid submission-selection partition mapping")
        original = pd.read_csv(bundle / "source_assignments.csv")
        actual = official[["Molecule_Name", "parent", "original_partition"]].rename(
            columns={"original_partition": "partition"}
        )
        if (
            not actual.sort_values("Molecule_Name")
            .reset_index(drop=True)
            .equals(original[actual.columns].sort_values("Molecule_Name").reset_index(drop=True))
        ):
            raise ValueError("Submission selection differs from original assignments")
    pool = pd.read_csv(bundle / "pool.csv")
    tasks = select_tasks(manifest, recipe)
    validation = official[official.partition == "validation"].copy()
    blind = set(pd.read_csv(bundle / "test.csv").parent)
    heldout = set() if final else set(official.loc[official.partition != "train", "parent"])
    pool = pool.loc[~pool.parent.isin(blind | heldout)].copy()
    if manifest.get("protocol") == "molecular-60-20-20-v1":
        if set(official.partition) != {"train", "validation", "internal_test"}:
            raise ValueError("Missing frozen partition")
        if official.groupby("parent").partition.nunique().max() != 1:
            raise ValueError("Parent crosses partitions")
        if manifest["exclusion_policy"] != "exact_parent":
            raise ValueError("Training exclusion policy mismatch")
    pool = pool.loc[pool[tasks].notna().any(axis=1)].reset_index(drop=True)
    return pool, validation.reset_index(drop=True), tasks, manifest


def select_tasks(manifest, recipe="full"):
    """Select auxiliary groups/heads without ever removing scored endpoints."""
    tasks = manifest["tasks"][:]
    groups = manifest["groups"]
    if recipe == "full":
        return tasks
    if recipe == "direct":
        return list(ACTIVITY_ENDPOINTS)
    if recipe == "official-only":
        keep = set(ACTIVITY_ENDPOINTS) | set(groups["single"]) | set(groups["tdi"])
        return [t for t in tasks if t in keep]
    if recipe == "without-pubchem":
        remove = set(groups["pubchem_response"]) | set(groups["pubchem_ac50"])
    elif recipe.startswith("without-response-"):
        enzyme = recipe.removeprefix("without-response-")
        task = enzyme + "_max_response"
        if task not in groups["pubchem_response"]:
            raise ValueError("Unknown response head")
        remove = {task}
    elif recipe.startswith("without-") and recipe[8:] in groups:
        remove = set(groups[recipe[8:]])
    else:
        raise ValueError("Unknown ablation")
    return [t for t in tasks if t not in remove]


def freeze_split(base_bundle, split_dir, output):
    """Adopt reviewed assignments without regenerating chemistry or source snapshots."""
    from .molecular_similarity import MatchingProblem, load_assignments, read_molecules

    base_bundle, split_dir, output = map(Path, (base_bundle, split_dir, output))
    if output.exists():
        raise FileExistsError(output)
    _, _, _, original = load_frames(base_bundle, final=True)
    split = json.loads((split_dir / "manifest.json").read_text())
    if split["required_training_exclusion_policy"] != "exact_parent_only":
        raise ValueError("Incompatible split exclusion policy")
    if split["analogue_exclusion_threshold"] is not None:
        raise ValueError("Analogue exclusions are incompatible with this protocol")
    if original["exclusion_policy"] != "exact_parent":
        raise ValueError("Source bundle may have removed analogues")
    if original["official_training_sha256"] != split["input_sha256"]["training"]:
        raise ValueError("Official training snapshot mismatch")
    if original["test_sha256"] != split["input_sha256"]["official_test"]:
        raise ValueError("Official test snapshot mismatch")
    assignment_path = split_dir / split["assignments_file"]
    if sha256_file(assignment_path) != split["assignments_sha256"]:
        raise ValueError("Assignments changed")
    problem = MatchingProblem(
        read_molecules(base_bundle / "official.csv", labeled=True),
        read_molecules(base_bundle / "test.csv"),
    )
    assignment = load_assignments(assignment_path, problem)
    if not problem.coverage(assignment, 100):
        raise ValueError("Insufficient held-out endpoint coverage")
    official = pd.read_csv(base_bundle / "official.csv")
    mapping = problem.assignments_frame(assignment).set_index("Molecule_Name").partition
    official["partition"] = official.Molecule_Name.map(mapping)
    pool = pd.read_csv(base_bundle / "pool.csv")
    # Bounds accompany official record IDs only; supplementary point labels stay unchanged.
    by_id = official.set_index("Molecule_Name")
    for ep in ACTIVITY_ENDPOINTS:
        for suffix in ("_conf_low", "_conf_high"):
            pool[ep + suffix] = pool.Molecule_Name.map(by_id[ep + suffix])
    output.mkdir(parents=True)
    official.to_csv(output / "official.csv", index=False)
    pool.to_csv(output / "pool.csv", index=False)
    (output / "test.csv").write_bytes((base_bundle / "test.csv").read_bytes())
    (output / "assignments.csv").write_bytes(assignment_path.read_bytes())
    manifest = {
        **original,
        "protocol": "molecular-60-20-20-v1",
        "base_bundle_manifest_sha256": sha256_file(base_bundle / "manifest.json"),
        "split_manifest_sha256": sha256_file(split_dir / "manifest.json"),
        "assignments_sha256": split["assignments_sha256"],
        "partition_rows": official.partition.value_counts().to_dict(),
        "validation_label_counts": official[official.partition == "validation"][
            list(ACTIVITY_ENDPOINTS)
        ]
        .count()
        .to_dict(),
        "internal_test_label_counts": official[official.partition == "internal_test"][
            list(ACTIVITY_ENDPOINTS)
        ]
        .count()
        .to_dict(),
        "file_sha256": {
            n: sha256_file(output / n)
            for n in ["official.csv", "pool.csv", "test.csv", "assignments.csv"]
        },
    }
    write_json(output / "manifest.json", manifest)
    load_frames(output)
    return manifest


def submission_selection_bundle(base_bundle, output):
    """Train on old train+validation; select checkpoints on old internal test."""
    base_bundle, output = Path(base_bundle), Path(output)
    if output.exists():
        raise FileExistsError(output)
    _, _, _, original = load_frames(base_bundle)
    if original.get("protocol") != "molecular-60-20-20-v1":
        raise ValueError("Submission selection requires the frozen three-way bundle")
    official = pd.read_csv(base_bundle / "official.csv")
    official["original_partition"] = official.partition
    official["partition"] = official.original_partition.map(
        {"train": "train", "validation": "train", "internal_test": "validation"}
    )
    blind = set(pd.read_csv(base_bundle / "test.csv").parent)
    if set(official.parent) & blind:
        raise ValueError("Official labeled parents overlap blinded test")
    output.mkdir(parents=True)
    official.to_csv(output / "official.csv", index=False)
    for name in ["pool.csv", "test.csv"]:
        (output / name).write_bytes((base_bundle / name).read_bytes())
    (output / "source_assignments.csv").write_bytes((base_bundle / "assignments.csv").read_bytes())
    manifest = {
        **{
            k: v
            for k, v in original.items()
            if k
            not in ["assignments_sha256", "internal_test_label_counts", "split_manifest_sha256"]
        },
        "protocol": "submission-train-plus-validation-v1",
        "source_bundle_manifest_sha256": sha256_file(base_bundle / "manifest.json"),
        "source_assignments_sha256": sha256_file(base_bundle / "assignments.csv"),
        "partition_rows": official.partition.value_counts().to_dict(),
        "validation_label_counts": official.loc[
            official.partition == "validation", list(ACTIVITY_ENDPOINTS)
        ]
        .count()
        .to_dict(),
        "validation_role": (
            "Former internal test, now used for checkpoint selection; not independent evaluation"
        ),
        "training_role": "Original train and validation plus eligible auxiliary data",
        "file_sha256": {
            name: sha256_file(output / name)
            for name in ["official.csv", "pool.csv", "test.csv", "source_assignments.csv"]
        },
    }
    write_json(output / "manifest.json", manifest)
    pool, val, _, _ = load_frames(output)
    if set(pool.parent) & (set(val.parent) | blind):
        raise ValueError("Submission training overlaps held-out parents")
    return manifest
