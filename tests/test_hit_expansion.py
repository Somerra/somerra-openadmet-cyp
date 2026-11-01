"""Scientific isolation and immutable artifact contracts for hit expansion."""

import json

import numpy as np
import pandas as pd
import pytest

from somerra_openadmet import hit_expansion as he
from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.download import sha256_file


@pytest.fixture
def tables():
    smiles = [
        "Cc1ccccc1",
        "CCc1ccccc1",
        "CCCc1ccccc1",
        "Oc1ccccc1",
        "COc1ccccc1",
        "Nc1ccccc1",
        "CNc1ccccc1",
        "Clc1ccccc1",
        "Brc1ccccc1",
        "Fc1ccccc1",
        "CCO",
        "CCCO",
        "CCCCO",
        "CCN",
        "CCCN",
        "CCCCN",
        "CCOC",
        "CCOCC",
        "C1CCCCC1",
        "CC1CCCCC1",
        "OC1CCCCC1",
        "C1CCNCC1",
        "CC1CCNCC1",
        "c1ccncc1",
        "Cc1ccncc1",
        "Oc1ccncc1",
        "c1ccoc1",
        "Cc1ccoc1",
        "CC(=O)O",
        "CCC(=O)O",
        "CC(=O)N",
        "CCC(=O)N",
    ]
    f = pd.DataFrame({"Molecule_Name": [f"m{i:02}" for i in range(len(smiles))], "SMILES": smiles})
    f["parent"] = f.SMILES.map(he.structure_key)
    for i, ep in enumerate(ACTIVITY_ENDPOINTS):
        f[ep] = 3 + np.random.default_rng(i).permutation(len(f)) / 10
        f[ep + "_conf_low"] = f[ep] - 0.05
        f[ep + "_conf_high"] = f[ep] + 0.05
    blind = pd.DataFrame({"Molecule_Name": ["blind"], "SMILES": ["P(=O)(O)(O)O"]})
    return f, blind


def small_split(f, blind, seed=42):
    return he.make_assignments(f, blind, seed, hits=2, neighbors=1, minimum_similarity=0)


def test_order_invariant_and_candidate_label_blind(tables):
    f, blind = tables
    a, seeds, links = small_split(f, blind)
    shuffled = small_split(f.sample(frac=1, random_state=10), blind)
    for expected, actual in zip((a, seeds, links), shuffled, strict=True):
        pd.testing.assert_frame_equal(expected, actual)
    candidate_ids = a.loc[~a.seed_pool, "Molecule_Name"]
    changed = f.copy()
    for ep in ACTIVITY_ENDPOINTS:
        changed.loc[changed.Molecule_Name.isin(candidate_ids), ep] = np.nan
        changed.loc[changed.Molecule_Name.isin(candidate_ids), ep + "_conf_low"] = 999
    for expected, actual in zip((a, seeds, links), small_split(changed, blind), strict=True):
        pd.testing.assert_frame_equal(expected, actual)
    assert a.loc[a.selected_seed, "partition"].eq("train").all()
    assert set(seeds.selection_endpoint) == set(he.SEED_TARGETS)
    assert not any(c in a for c in ACTIVITY_ENDPOINTS)


def test_shared_nominations_merge_families():
    links = pd.DataFrame(
        {"seed_parent": ["s1", "s1", "s2", "s2", "s3"], "analog_parent": ["a", "b", "b", "c", "d"]}
    )
    roles = he.assign_series(links)
    assert roles["a"] == roles["b"] == roles["c"]
    assert roles["d"][1] != roles["a"][1]
    with pytest.raises(ValueError, match="two distinct"):
        he.assign_series(links.iloc[:4])


def test_similarity_ties_and_shortages():
    # Ethane versus these molecules gives equal Tanimoto; lexical ties are stable.
    all_links = he.select_analogs(["CC"], ["N", "O", "F"], neighbors=2, minimum_similarity=0)
    assert all_links.analog_parent.tolist() == ["F", "N"]
    assert he.select_analogs(["CC"], ["N", "O"], minimum_similarity=0.45).empty
    with pytest.raises(ValueError, match="overlap"):
        he.select_analogs(["CC"], ["CC"])


def test_parent_aliases_and_all_source_exclusions(tables):
    f, blind = tables
    a, _, _ = small_split(f, blind)
    held_id = a.loc[a.partition == "test", "Molecule_Name"].iloc[0]
    alias = f.loc[f.Molecule_Name == held_id].copy()
    alias["Molecule_Name"] = "external-alias"
    alias["SMILES"] += ".[Na+]"
    alias["auxiliary"] = 123.0
    blind_alias = f.iloc[[0]].copy()
    blind_alias["Molecule_Name"] = "blind-alias"
    blind_alias["SMILES"] = blind.SMILES.iloc[0]
    blind_alias["parent"] = he.structure_key(blind.SMILES.iloc[0])
    pool = pd.concat([f, alias, blind_alias], ignore_index=True)
    frames, removed = he.build_frames(f, pool, blind, a)
    assert {"external-alias", "blind-alias"} <= set(removed.Molecule_Name)
    assert not {"external-alias", "blind-alias"} & set(frames["train"].Molecule_Name)
    assert "auxiliary" in frames["train"]
    roles = {r: set(t.expansion_parent) for r, t in frames.items()}
    assert not roles["train"] & (roles["test"] | roles["validation"])


def test_duplicate_parents_share_role_and_stereo_key(tables):
    f, blind = tables
    alias = f.iloc[[0]].copy()
    alias["Molecule_Name"] = "z_alias"
    alias["SMILES"] += ".[Na+]"
    a, _, _ = small_split(pd.concat([f, alias]), blind)
    assert a.groupby("expansion_parent").partition.nunique().max() == 1
    assert he.structure_key("C[C@H](O)F") == he.structure_key("C[C@@H](O)F")


def test_malformed_inputs_and_seed_shortfall(tables):
    f, blind = tables
    with pytest.raises(ValueError, match="duplicate"):
        small_split(pd.concat([f, f.iloc[[0]]]), blind)
    bad = f.copy()
    bad.loc[0, "SMILES"] = "not-a-molecule"
    with pytest.raises(ValueError, match="Invalid"):
        small_split(bad, blind)
    with pytest.raises(ValueError, match="Fewer than"):
        he.make_assignments(f, blind, hits=100)
    with pytest.raises(ValueError, match="two distinct"):
        he.make_assignments(f, blind, hits=2, minimum_similarity=1)


def test_pool_mismatch_rejected(tables):
    f, blind = tables
    a, _, _ = small_split(f, blind)
    pool = f.copy()
    pool.loc[0, ACTIVITY_ENDPOINTS[0]] += 1
    with pytest.raises(ValueError, match="label mismatch"):
        he.build_frames(f, pool, blind, a)


def test_prediction_alignment_missingness_and_undefined_metrics(tables):
    f, _ = tables
    pred = f[["Molecule_Name", *ACTIVITY_ENDPOINTS]].copy()
    result = he.score_predictions(f, pred.sample(frac=1))
    assert result["MA-ST-RAE"] == 0
    sparse = f.copy()
    sparse[ACTIVITY_ENDPOINTS[0]] = np.nan
    result = he.score_predictions(sparse, pred)
    assert result["MA-ST-RAE"] is None
    assert result["endpoints"][ACTIVITY_ENDPOINTS[0]]["n"] == 0
    assert result["endpoints"][ACTIVITY_ENDPOINTS[0]]["undefined_reason"]
    pred.loc[0, ACTIVITY_ENDPOINTS[0]] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        he.score_predictions(sparse, pred)
    with pytest.raises(ValueError, match="exactly"):
        he.score_predictions(f, pred.iloc[1:])
    with pytest.raises(ValueError, match="duplicate"):
        he.score_predictions(f, pd.concat([pred, pred.iloc[[0]]]))


def test_single_label_and_zero_denominator(tables):
    f, _ = tables
    one = f.iloc[[0]]
    result = he.score_predictions(one, one)
    assert result["MA-ST-RAE"] is None
    assert all(e["underpowered"] for e in result["endpoints"].values())
    assert all("denominator" in e["undefined_reason"] for e in result["endpoints"].values())


@pytest.fixture(params=["current", "five_rotation_legacy"])
def prepared(tmp_path, tables, monkeypatch, request):
    if request.param == "five_rotation_legacy":
        monkeypatch.setattr(he, "PROTOCOL", he.FIVE_ROTATION_PROTOCOL)
        monkeypatch.setattr(he, "REPEATS", tuple(range(42, 47)))
    f, blind = tables
    source = tmp_path / "source"
    source.mkdir()
    for name, table in [("official", f), ("pool", f), ("test", blind)]:
        table.to_csv(source / f"{name}.csv", index=False)
    he.write_json(source / "manifest.json", {"tasks": list(ACTIVITY_ENDPOINTS)})
    wb = tmp_path / "wb"
    (wb / "data").mkdir(parents=True)
    pd.DataFrame({"molecule_name": f.Molecule_Name, "fold": np.arange(len(f)) % 5}).to_csv(
        wb / "data/mt_aux8.csv", index=False
    )
    he.write_json(wb / "contract.json", {"historical": True})
    hashes = {str(p.relative_to(wb)): sha256_file(p) for p in wb.rglob("*") if p.is_file()}
    wb_hashes = tmp_path / "wb-hashes.json"
    he.write_json(wb_hashes, {"sha256": hashes})
    prior = tmp_path / "prior.json"
    he.write_json(
        prior,
        {"input_sha256": {str(p): sha256_file(p) for p in source.glob("*.csv")}, "file_sha256": {}},
    )
    original = he.make_bank
    monkeypatch.setattr(
        he,
        "make_bank",
        lambda f, b: original(f, b, minimum_potency=0, neighbors=1, minimum_similarity=0),
    )
    root = tmp_path / "benchmark"
    monkeypatch.chdir(tmp_path)
    he.prepare(prior.name, wb.name, wb_hashes.name, root)
    if request.param == "five_rotation_legacy":
        monkeypatch.setattr(he, "PROTOCOL", "hit-expansion-frozen-bank-v4")
        monkeypatch.setattr(he, "REPEATS", tuple(range(42, 45)))
    return root, wb, hashes


def test_relative_preparation_paths_verify_from_another_directory(prepared, tmp_path, monkeypatch):
    root, _, _ = prepared
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    manifest = he.verify_bundle(root)
    assert all(he.Path(p).is_absolute() for p in manifest["protected_sha256"])


def test_prepare_audit_immutable_and_historical_hashes(prepared, tmp_path):
    root, wb, hashes = prepared
    he.verify_hashes(hashes, wb)
    manifest = he.verify_bundle(root)
    assert manifest["training_launched"] is False
    result = he.audit(root, tmp_path / "audit")
    assert result["historical_hashes_verified"]
    assert result["unique_evaluation_parents"] > 0
    assert result["evaluation_parents_used_for_training_in_other_repeats"] == 0
    assert result["pooled_test"]["rows"] == result["unique_evaluation_parents"]
    assert all(r["blind_overlap"] == 0 for r in result["checks"].values())
    with pytest.raises(FileExistsError):
        he.prepare("unused", "unused", "unused", root)
    with pytest.raises(FileExistsError):
        he.audit(root, tmp_path / "audit")
    (root / "repeat-42/test.csv").write_text("changed")
    with pytest.raises(ValueError, match="Changed artifact"):
        he.verify_bundle(root)


def test_evaluate_provenance_and_no_partial_macro(prepared, tmp_path):
    root, _, _ = prepared
    m = he.verify_bundle(root)
    predictions, records = tmp_path / "predictions", tmp_path / "records"
    predictions.mkdir()
    records.mkdir()
    checkpoint = records / "model.bin"
    checkpoint.write_bytes(b"synthetic test fixture")
    for seed in he.read_json(root / "manifest.json")["repeats"]:
        name = f"repeat-{seed}"
        truth = pd.read_csv(root / name / "test.csv")
        truth[["Molecule_Name", *ACTIVITY_ENDPOINTS]].to_csv(
            predictions / f"{name}.csv", index=False
        )
        he.write_json(
            records / f"{name}.json",
            {
                "train_sha256": m["file_sha256"][f"{name}/train.csv"],
                "validation_sha256": m["file_sha256"][f"{name}/validation.csv"],
                "fresh_fit": True,
                "test_used_for_fitting_or_selection": False,
                "calibration": "none",
                "scalers_and_task_weights_fit_on": "train",
                "analog_bank_used_for_gradient_training": False,
                "checkpoint_selection_scope": "this_rotation_validation_only",
                "checkpoint_sha256": {"model.bin": sha256_file(checkpoint)},
            },
        )
    result = he.evaluate(root, predictions, records, tmp_path / "evaluation")
    assert len(result["repeats"]) == len(m["repeats"])
    macro_key = (
        "macro_across_rotations" if m["protocol"] == he.PROTOCOL else "macro_across_all_five"
    )
    assert macro_key in result
    assert result[f"{macro_key}_role"].startswith("diagnostic only")
    assert result["unique_test_parents"] <= result["test_row_appearances"]
    assert result["unique_test_parents"] == result["test_row_appearances"]
    assert result["primary_pooled_score"]["MA-ST-RAE"] == 0
    assert result["family_bootstrap"]["n_resamples"] == 1000
    path = records / "repeat-42.json"
    record = json.loads(path.read_text())
    record["test_used_for_fitting_or_selection"] = True
    he.write_json(path, record)
    with pytest.raises(ValueError, match="test-isolation"):
        he.evaluate(root, predictions, records, tmp_path / "bad-evaluation")


def test_bank_labels_and_input_order_cannot_change_membership(tables):
    f, blind = tables

    def build(table):
        return he.make_bank(table, blind, minimum_potency=0, minimum_similarity=0, neighbors=1)

    bank, seeds, links = build(f)
    changed = f.copy()
    candidate_ids = bank.loc[~bank.seed_pool, "Molecule_Name"]
    changed.loc[changed.Molecule_Name.isin(candidate_ids), list(ACTIVITY_ENDPOINTS)] = np.nan
    changed = changed.sample(frac=1, random_state=17)
    for expected, actual in zip((bank, seeds, links), build(changed), strict=True):
        pd.testing.assert_frame_equal(expected, actual)
    assert bank.loc[bank.selected_seed, "expansion_fold"].eq(-1).all()
    assert not seeds.selection_endpoint.str.startswith("CYP2D6").any()
    assert links.groupby("series_id").expansion_fold.nunique().max() == 1


def test_bank_qualifying_hits_and_shortfalls(tables):
    f, blind = tables
    bank, seeds, _ = he.make_bank(
        f, blind, minimum_potency=4, minimum_similarity=0, neighbors=1, n_folds=2
    )
    assert seeds.seed_pIC50.ge(4).all()
    for ep in he.SEED_TARGETS:
        expected = f.loc[
            f.Molecule_Name.isin(bank.loc[bank.seed_pool, "Molecule_Name"]) & (f[ep] >= 4)
        ]
        assert set(seeds.loc[seeds.selection_endpoint == ep, "seed_id"]) == set(
            expected.Molecule_Name
        )
    with pytest.raises(ValueError, match="No qualifying"):
        he.make_bank(f, blind, minimum_potency=100)
    with pytest.raises(ValueError, match="analog families"):
        he.make_bank(f, blind, minimum_potency=0, minimum_similarity=1)


def test_bank_never_trains_and_each_row_scores_once(prepared):
    root, _, _ = prepared
    bank = pd.read_csv(root / "bank-assignments.csv")
    held = set(bank.loc[bank.expansion_fold >= 0, "expansion_parent"])
    test_ids, training_hashes = [], set()
    validation_counts = {}
    for seed in he.read_json(root / "manifest.json")["repeats"]:
        folder = root / f"repeat-{seed}"
        train, val, test = [
            pd.read_csv(folder / f"{r}.csv") for r in ["train", "validation", "test"]
        ]
        assert not held & set(train.expansion_parent)
        assert not set(val.series_id) & set(test.series_id)
        test_ids.extend(test.Molecule_Name)
        training_hashes.add(sha256_file(folder / "train.csv"))
        for mid in val.Molecule_Name:
            validation_counts[mid] = validation_counts.get(mid, 0) + 1
    assert len(training_hashes) == 1
    assert len(test_ids) == len(set(test_ids)) == len(held)
    assert set(validation_counts.values()) == {
        len(he.read_json(root / "manifest.json")["repeats"]) - 1
    }
    he.verify_bank_roles(root)


def test_bank_semantic_check_rejects_training_contamination(prepared):
    root, _, _ = prepared
    folder = root / "repeat-42"
    train, test = [pd.read_csv(folder / f"{r}.csv") for r in ["train", "test"]]
    pd.concat([train, test.iloc[[0]]], ignore_index=True).to_csv(folder / "train.csv", index=False)
    with pytest.raises(ValueError, match="entered training"):
        he.verify_bank_roles(root)


def test_family_bootstrap_is_deterministic_and_does_not_count_rows_as_families(tables):
    truth, _ = tables
    truth["series_id"] = [f"family-{i // 4}" for i in range(len(truth))]
    pred = truth[["Molecule_Name", *ACTIVITY_ENDPOINTS]].copy()
    a = he.family_bootstrap(truth, pred, n_resamples=100)
    b = he.family_bootstrap(truth, pred, n_resamples=100)
    assert a == b
    assert a["families"] == len(truth) // 4
    assert a["intervals"]["MA-ST-RAE"]["percentile_95"] == [0, 0]
    with pytest.raises(ValueError, match="unique"):
        he.score_predictions(pd.concat([truth, truth.iloc[[0]]]), pred)
