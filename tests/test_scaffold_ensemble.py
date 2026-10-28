import json

import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.data_sources.common import write_json
from somerra_openadmet.download import sha256_file
from somerra_openadmet.molecular_similarity import scaffold
from somerra_openadmet.reference_member_data import load_frames
from somerra_openadmet.scaffold_ensemble import aligned_mean, assign_folds, prepare


@pytest.fixture
def bundle(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    parents = [
        "c1ccccc1",
        "c1ccncc1",
        "C1CCCCC1",
        "C1CCCC1",
        "c1ccoc1",
        "c1ccsc1",
        "c1cncnc1",
        "C1CC1",
        "C1CCC1",
        "C1CCCCCC1",
        "CCC",
        "CCO",
    ]
    official = pd.DataFrame(
        {
            "Molecule_Name": [f"m{i}" for i in range(len(parents))],
            "parent": parents,
            "partition": "train",
        }
    )
    for ep in ACTIVITY_ENDPOINTS:
        official[ep] = np.arange(len(parents)) / 5 + 3
        official[ep + "_conf_low"] = official[ep] - 0.1
        official[ep + "_conf_high"] = official[ep] + 0.1
    official.to_csv(base / "official.csv", index=False)
    pool = official.drop(columns="partition").copy()
    pool["sample_weight"] = 1.0
    # Distinct auxiliary parent shares the benzene scaffold with an official parent.
    extra = pool.iloc[[0]].copy()
    extra["Molecule_Name"] = "aux"
    extra["parent"] = "Cc1ccccc1"
    extra["sample_weight"] = 2.0
    pool = pd.concat([pool, extra], ignore_index=True)
    pool.to_csv(base / "pool.csv", index=False)
    pd.DataFrame({"Molecule_Name": ["blind"], "parent": ["C1COCCN1"]}).to_csv(
        base / "test.csv", index=False
    )
    write_json(
        base / "manifest.json",
        {
            "tasks": list(ACTIVITY_ENDPOINTS),
            "groups": {},
            "exclusion_policy": "exact_parent",
            "file_sha256": {
                n: sha256_file(base / n) for n in ["official.csv", "pool.csv", "test.csv"]
            },
        },
    )
    model = tmp_path / "model.json"
    write_json(model, {})
    return base, model


def test_folds_keep_auxiliary_scaffolds_together_and_cover_oof_once(bundle, tmp_path):
    base, model = bundle
    output = tmp_path / "new"
    prepare(base, output, model, min_labels=1)
    assignments = pd.read_csv(output / "folds.csv", keep_default_na=False)
    assert assignments.groupby("scaffold").fold.nunique().max() == 1
    # Empty scaffolds are a single group, too.
    assert assignments.loc[assignments.parent.isin(["CCC", "CCO"]), "fold"].nunique() == 1
    all_held = []
    for k in range(5):
        fold = output / f"fold-{k}/data"
        train, val, _, _ = load_frames(fold)
        official = pd.read_csv(fold / "official.csv")
        held = official.loc[official.partition == "internal_test"]
        all_held.extend(held.Molecule_Name)
        assert not set(train.parent.map(scaffold)) & set(
            pd.concat([val, held]).parent.map(scaffold)
        )
        assert not set(val.parent.map(scaffold)) & set(held.parent.map(scaffold))
        if "aux" in set(train.Molecule_Name):
            assert train.set_index("Molecule_Name").loc["aux", "sample_weight"] == 2
        with pytest.raises(ValueError, match="retain validation"):
            load_frames(fold, final=True)
    assert sorted(all_held) == sorted(pd.read_csv(base / "official.csv").Molecule_Name)


def test_assignments_do_not_depend_on_row_order_or_target_values(bundle):
    base, _ = bundle
    pool = pd.read_csv(base / "pool.csv")
    official = pd.read_csv(base / "official.csv")
    original = assign_folds(pool, official)
    changed = official.sample(frac=1, random_state=4).copy()
    changed[list(ACTIVITY_ENDPOINTS)] *= 10
    pd.testing.assert_frame_equal(original, assign_folds(pool.sample(frac=1), changed))


def test_loader_rejects_forged_scaffold_identity_even_with_new_file_hash(bundle, tmp_path):
    base, model = bundle
    output = tmp_path / "new"
    prepare(base, output, model, min_labels=1)
    fold = output / "fold-0/data"
    path = fold / "folds.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    frame.loc[0, "scaffold"] = "invented"
    frame.to_csv(path, index=False)
    manifest = json.loads((fold / "manifest.json").read_text())
    manifest["file_sha256"]["folds.csv"] = sha256_file(path)
    write_json(fold / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Scaffold identity"):
        load_frames(fold)


def test_average_requires_every_fold_and_exact_molecule_alignment():
    frames = []
    for k in range(5):
        frame = pd.DataFrame({"Molecule_Name": ["a", "b"], "SMILES": ["C", "CC"]})
        for ep in ACTIVITY_ENDPOINTS:
            frame[ep] = [k, k + 10]
        frames.append(frame)
    assert aligned_mean(frames)[ACTIVITY_ENDPOINTS[0]].tolist() == [2.0, 12.0]
    with pytest.raises(ValueError, match="complete five-fold"):
        aligned_mean(frames[:4])
    frames[4] = frames[4].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="identity/order"):
        aligned_mean(frames)
    frames[4] = frames[4].iloc[::-1].reset_index(drop=True)
    frames[4][ACTIVITY_ENDPOINTS[0]] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        aligned_mean(frames)


def test_single_protocol_restores_all_eligible_records_only_for_final_fit(bundle, tmp_path):
    base, model = bundle
    output = tmp_path / "folds"
    prepare(base, output, model, min_labels=1)
    single = output / "fold-0/data"
    manifest = json.loads((single / "manifest.json").read_text())
    assignments = pd.read_csv(single / "folds.csv", keep_default_na=False)
    official = pd.read_csv(single / "official.csv")
    official["partition"] = np.where(
        official.parent.map(assignments.set_index("parent").fold) == 0, "validation", "train"
    )
    official.to_csv(single / "official.csv", index=False)
    manifest.update(protocol="scaffold-single-v1", fold=0)
    manifest["file_sha256"]["official.csv"] = sha256_file(single / "official.csv")
    write_json(single / "manifest.json", manifest)
    train, valid, _, _ = load_frames(single)
    assert not set(train.parent.map(scaffold)) & set(valid.parent.map(scaffold))
    full, _, _, _ = load_frames(single, final=True)
    pd.testing.assert_frame_equal(full, pd.read_csv(base / "pool.csv"))
    assert set(valid.parent) <= set(full.parent)
    assert not set(full.parent) & {"C1COCCN1"}
    # Even final fitting checks that the frozen development partition is genuine.
    official.loc[official.partition == "validation", "partition"] = "train"
    official.to_csv(single / "official.csv", index=False)
    manifest["file_sha256"]["official.csv"] = sha256_file(single / "official.csv")
    write_json(single / "manifest.json", manifest)
    with pytest.raises(ValueError, match="partition differs"):
        load_frames(single, final=True)
