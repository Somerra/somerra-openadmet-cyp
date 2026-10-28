"""Protect frozen evaluation identities independently of any model implementation."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "evaluation/splits/frozen-20261028"


def read_rows(name):
    with (ROOT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_snapshot_hashes_and_coverage():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    identities = read_rows("identities.csv")
    assignments = read_rows("assignments.csv")
    blind = read_rows("official-blind.csv")
    ids = {r["Molecule_Name"] for r in identities}
    assert len(identities) == len(ids) == manifest["pool_records"]
    assert len(assignments) == len(ids)
    assert {r["Molecule_Name"] for r in assignments} == ids
    assert len(blind) == len({r["Molecule_Name"] for r in blind}) == 750
    assert not ids & {r["Molecule_Name"] for r in blind}
    assert not {r["parent"] for r in identities} & {r["parent"] for r in blind}
    official = {r["Molecule_Name"] for r in identities if r["official_training"] == "True"}
    assert len(official) == manifest["official_training_records"] == 4905
    assert {r["Molecule_Name"] for r in assignments if r["molecular_partition"]} == official
    assert (
        Counter(r["molecular_partition"] for r in assignments if r["Molecule_Name"] in official)
        == manifest["molecular_counts"]
    )
    for kind in ("reference", "balanced"):
        counts = Counter(r[f"{kind}_fold"] for r in assignments if r["Molecule_Name"] in official)
        assert counts == manifest[f"{kind}_official_fold_counts"]
        assert set(r[f"{kind}_fold"] for r in assignments) == set("01234")


def test_group_separation_and_rotation():
    identities = {r["Molecule_Name"]: r for r in read_rows("identities.csv")}
    assignments = read_rows("assignments.csv")
    for column in ("reference_fold", "balanced_fold", "molecular_partition"):
        groups = defaultdict(set)
        for row in assignments:
            if row[column]:
                groups[identities[row["Molecule_Name"]]["parent"]].add(row[column])
        assert all(len(values) == 1 for values in groups.values())
    scaffolds = defaultdict(set)
    for row in assignments:
        scaffolds[identities[row["Molecule_Name"]]["scaffold"]].add(row["balanced_fold"])
    assert all(len(values) == 1 for values in scaffolds.values())
    for column in ("reference_fold", "balanced_fold"):
        folds = [{r["Molecule_Name"] for r in assignments if int(r[column]) == k} for k in range(5)]
        for k in range(5):
            test, valid = folds[k], folds[(k + 1) % 5]
            train = set().union(*(folds[j] for j in range(5) if j not in (k, (k + 1) % 5)))
            assert not (test & valid or test & train or valid & train)
            assert train | valid | test == set(identities)
