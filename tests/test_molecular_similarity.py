import json

import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.molecular_similarity import (
    MatchingProblem,
    acceptable,
    composition_mmd,
    fingerprints,
    kernel,
    load_assignments,
    nearest_distances,
    read_molecules,
    search,
)
from somerra_openadmet.molecular_split_cli import main, supplemental_audit

SMILES = [
    "CC",
    "CCC",
    "CCCC",
    "CCCCC",
    "CCCCCC",
    "CCO",
    "CCCO",
    "CCCCO",
    "CCCCCO",
    "CCN",
    "CCCN",
    "CCCCN",
    "CCOC",
    "CCOCC",
    "CCCOC",
    "CCNC",
    "CCNCC",
    "c1ccccc1",
    "Cc1ccccc1",
    "Oc1ccccc1",
    "Nc1ccccc1",
    "c1ccncc1",
    "CC(=O)O",
    "CCC(=O)O",
    "CCCC(=O)O",
    "C1CCCCC1",
    "CC1CCCCC1",
    "C1CCCC1",
    "C1CCOC1",
    "C1CCNCC1",
]


@pytest.fixture
def inputs(tmp_path):
    train = pd.DataFrame({"Molecule_Name": [f"t{i}" for i in range(len(SMILES))], "SMILES": SMILES})
    for i, ep in enumerate(ACTIVITY_ENDPOINTS):
        train[ep] = np.arange(len(train)) + i
    train.loc[len(train)] = train.iloc[0].copy()
    train.loc[len(train) - 1, "Molecule_Name"] = "duplicate-parent-record"
    blind = pd.DataFrame(
        {"Molecule_Name": ["b0", "b1", "b2"], "SMILES": ["CCCCCCC", "CCCCCCO", "CCc1ccccc1"]}
    )
    paths = tmp_path / "train.csv", tmp_path / "test.csv"
    train.to_csv(paths[0], index=False)
    blind.to_csv(paths[1], index=False)
    return paths


def problem(inputs):
    return MatchingProblem(read_molecules(inputs[0], labeled=True), read_molecules(inputs[1]))


def test_tanimoto_and_self_exclusion():
    fps = fingerprints(["CC", "CCC", "CC"])
    values = kernel(fps, fps)
    assert values[0, 0] == 1 and values[0, 2] == 1
    assert 0 < values[0, 1] < 1
    d = nearest_distances(values, ["a", "b", "a"], ["a", "b", "a"])
    assert d[0] > 0 and d[1] > 0
    with pytest.raises(ValueError, match="No non-self"):
        nearest_distances(np.ones((1, 1)), ["a"], ["a"])
    with pytest.raises(ValueError, match="nonempty"):
        nearest_distances(np.empty((2, 0)))


def test_composition_and_training_distance_are_distinct():
    # Same mean nearest-training similarity can coexist with unrelated test chemistry.
    same = composition_mmd(np.eye(2), 0.5, 0.5)
    different = composition_mmd(np.eye(2), 0.0, 0.5)
    assert same == 0 and different == 1
    assert np.array_equal(
        nearest_distances(np.array([[0.6, 0.2], [0.1, 0.6]])),
        nearest_distances(np.array([[0.6, 0.1], [0.2, 0.6]])),
    )


def test_reproducible_value_blind_parent_grouping(inputs):
    p = problem(inputs)
    assignment, baseline, result = search(p, candidates=8, swaps=16, minimum=1)
    before = p.assignments_frame(assignment)
    frame = pd.read_csv(inputs[0])
    for ep in ACTIVITY_ENDPOINTS:
        frame[ep] = frame[ep] * 1000 - 777
    frame.sample(frac=1, random_state=123).to_csv(inputs[0], index=False)
    blind = pd.read_csv(inputs[1])
    blind.iloc[::-1].to_csv(inputs[1], index=False)
    q = problem(inputs)
    again, _, repeat = search(q, candidates=8, swaps=16, minimum=1)
    pd.testing.assert_frame_equal(before, q.assignments_frame(again))
    assert repeat == result
    assert np.bincount(assignment).tolist() == [18, 6, 6]
    assert before.groupby("parent").partition.nunique().max() == 1
    assert len(before) == 31 and not set(before.Molecule_Name) & set(q.blind.Molecule_Name)
    assert (assignment == search(p, candidates=8, swaps=16, minimum=1)[0]).all()
    assert p.coverage(baseline, 1)


def test_acceptance_gate_and_unapproved_noop(inputs):
    assert acceptable(np.array([0.9, 1.0, 1.0, 1.0]), np.ones(4), np.ones(4))
    assert not acceptable(np.array([0.1, 0.1, 0.1, 1.01]), np.ones(4), np.ones(4))
    p = problem(inputs)
    _, _, result = search(p, candidates=1, swaps=0, minimum=1)
    assert result["status"] == "unapproved"
    with pytest.raises(ValueError, match="coverage"):
        search(p, minimum=100)
    with pytest.raises(ValueError, match="Invalid search"):
        search(p, candidates=0, minimum=1)


def test_assignments_reject_leakage_and_wrong_coverage(inputs, tmp_path):
    p = problem(inputs)
    a = p.random_partition(np.random.default_rng(42))
    f = p.assignments_frame(a)
    path = tmp_path / "assignment.csv"
    f.to_csv(path, index=False)
    assert np.array_equal(a, load_assignments(path, p))
    f.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly"):
        load_assignments(path, p)
    dup = f.index[f.parent.duplicated(keep=False)]
    f.loc[dup[0], "partition"] = "train"
    f.loc[dup[1], "partition"] = "validation"
    f.to_csv(path, index=False)
    with pytest.raises(ValueError, match="crosses"):
        load_assignments(path, p)


def test_input_failures(inputs):
    train = pd.read_csv(inputs[0])
    blind = pd.read_csv(inputs[1])
    blind.loc[0, "SMILES"] = train.iloc[0].SMILES
    blind.to_csv(inputs[1], index=False)
    with pytest.raises(ValueError, match="share standardized"):
        problem(inputs)
    train.loc[0, "SMILES"] = "not-a-smiles"
    train.to_csv(inputs[0], index=False)
    with pytest.raises(ValueError, match="Invalid molecular"):
        read_molecules(inputs[0], labeled=True)
    train.loc[0, "SMILES"] = None
    train.to_csv(inputs[0], index=False)
    with pytest.raises(ValueError, match="Missing molecular"):
        read_molecules(inputs[0], labeled=True)


def test_scaffold_groups_never_cross(inputs):
    p = problem(inputs)
    a = p.scaffold_partition()
    from somerra_openadmet.molecular_similarity import scaffold

    groups = {}
    for parent, part in zip(p.parents, a, strict=True):
        groups.setdefault(scaffold(parent), set()).add(part)
    assert all(len(v) == 1 for v in groups.values())
    assert set(a) == {0, 1, 2}


def test_supplements_do_not_enter_optimization(inputs, tmp_path):
    p = problem(inputs)
    a, _, _ = search(p, candidates=2, swaps=2, minimum=1)
    source = tmp_path / "aux.csv"
    pd.DataFrame({"parent": list(p.parents) + list(p.blind_parents)}).to_csv(source, index=False)
    result, hashes = supplemental_audit([f"aux={source}"], p, a)
    assert result["aux"]["eligible_training_parents"] == 18
    assert result["aux"]["exact_overlap"]["official_test"] == 3
    assert hashes["aux"]
    assert np.array_equal(a, search(p, candidates=2, swaps=2, minimum=1)[0])


def test_cli_generate_failure_and_audit(inputs, tmp_path):
    out = tmp_path / "generated"
    common = ["--training", str(inputs[0]), "--official-test", str(inputs[1]), "--min-labels", "1"]
    assert (
        main(["generate", *common, "--output", str(out), "--candidates", "1", "--swaps", "0"]) == 2
    )
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "unapproved" and not (out / "assignments.csv").exists()
    assert (out / "matching-diagnostics.svg").exists()
    with pytest.raises(FileExistsError):
        main(["generate", *common, "--output", str(out)])
    audit = tmp_path / "audit"
    assert (
        main(
            [
                "audit",
                *common,
                "--assignments",
                str(out / "candidate-assignments.csv"),
                "--output",
                str(audit),
            ]
        )
        == 0
    )
    result = json.loads((audit / "audit.json").read_text())
    assert result["designs"]["selected"]["partitions"]["internal_test"]["parents"] == 6


def test_cli_approved_and_coverage_failure(inputs, tmp_path):
    common = ["--training", str(inputs[0]), "--official-test", str(inputs[1])]
    out = tmp_path / "approved"
    code = main(
        [
            "generate",
            *common,
            "--output",
            str(out),
            "--min-labels",
            "1",
            "--candidates",
            "16",
            "--swaps",
            "128",
        ]
    )
    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "approved"
    assert (out / "assignments.csv").exists()
    with pytest.raises(ValueError, match="coverage"):
        main(["generate", *common, "--output", str(tmp_path / "infeasible")])
    assert not (tmp_path / "infeasible").exists()


def test_missing_columns_duplicate_ids_and_malformed_assignments(inputs, tmp_path):
    p = problem(inputs)
    frame = pd.read_csv(inputs[0])
    frame.loc[1, "Molecule_Name"] = frame.loc[0, "Molecule_Name"]
    frame.to_csv(inputs[0], index=False)
    with pytest.raises(ValueError, match="unique"):
        read_molecules(inputs[0], labeled=True)
    frame.drop(columns=["SMILES"]).to_csv(inputs[0], index=False)
    with pytest.raises(ValueError):
        read_molecules(inputs[0], labeled=True)
    f = p.assignments_frame(p.random_partition(np.random.default_rng(42)))
    f.loc[0, "parent"] = "wrong-parent"
    path = tmp_path / "bad.csv"
    f.to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not match"):
        load_assignments(path, p)
