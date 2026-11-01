import json

import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.download import sha256_file
from somerra_openadmet.expansion_evaluation import evaluate, make_assignments, structure_key


def molecules():
    n = 60
    f = pd.DataFrame(
        {
            "Molecule_Name": [f"m{i:03}" for i in range(n)],
            "SMILES": [
                ["c1ccccc1", "c1ccncc1", "C1CCCCC1"][i // 20] + "C" * (i % 20 + 1) + "O"
                for i in range(n)
            ],
        }
    )
    for j, ep in enumerate(ACTIVITY_ENDPOINTS):
        f[ep] = -np.abs(np.arange(n) - [4, 29, 40, 54][j])
    return f


def test_candidate_labels_cannot_change_split_and_seeds_stay_training():
    f = molecules()
    blind = pd.DataFrame({"SMILES": ["N#N"]})
    a, seeds, _ = make_assignments(f, blind, hits=1, neighbors=6, min_similarity=0)
    changed = f.copy()
    changed.loc[~a.seed_pool.to_numpy(), list(ACTIVITY_ENDPOINTS)] = 99999
    b, _, _ = make_assignments(changed, blind, hits=1, neighbors=6, min_similarity=0)
    assert a.partition.equals(b.partition)
    assert a.loc[a.selected_seed, "partition"].eq("train").all()
    assert len(seeds) == 3
    assert not seeds.selection_endpoint.str.startswith("CYP2D6").any()
    c, _, _ = make_assignments(
        f.sample(frac=1, random_state=3), blind, hits=1, neighbors=6, min_similarity=0
    )
    assert a.partition.equals(c.partition)
    assert a.groupby("expansion_parent").partition.nunique().max() == 1


def test_stereoisomers_share_parent_and_blind_parent_is_excluded():
    assert structure_key("C[C@H](O)CC") == structure_key("C[C@@H](O)CC")
    f = molecules()
    a, _, _ = make_assignments(
        f, pd.DataFrame({"SMILES": [f.SMILES.iloc[0]]}), hits=1, neighbors=6, min_similarity=0
    )
    assert a.loc[a.Molecule_Name == "m000", "partition"].item() == "excluded_blind"


def test_duplicate_ids_and_empty_analog_selection_fail(monkeypatch):
    f = molecules()
    with pytest.raises(ValueError, match="unique"):
        make_assignments(pd.concat([f, f.iloc[:1]]), pd.DataFrame({"SMILES": ["N#N"]}))
    monkeypatch.setattr(
        "somerra_openadmet.expansion_evaluation.DataStructs.BulkTanimotoSimilarity",
        lambda seed, candidates: [0.0] * len(candidates),
    )
    with pytest.raises(ValueError, match="No analogs"):
        make_assignments(
            f, pd.DataFrame({"SMILES": ["N#N"]}), hits=1, neighbors=1, min_similarity=1
        )


def evaluation_fixture(tmp_path):
    root = tmp_path / "split"
    root.mkdir()
    x = np.linspace(-1, 1, 30)
    z = (x - x.mean()) / x.std()
    noise = np.cos(np.arange(30))
    noise -= noise.mean()
    noise -= (noise @ z) / (z @ z) * z
    noise /= noise.std()
    y = 4 + 0.5 * z + np.sqrt(0.75) * noise  # rho exactly .5: valid for all historical factors
    for role in ["train", "validation", "calibration", "test"]:
        f = pd.DataFrame({"Molecule_Name": [f"{role}{i}" for i in range(30)], "SMILES": "CCO"})
        pred = f.copy()
        for ep in ACTIVITY_ENDPOINTS:
            f[ep] = y
            f[ep + "_conf_low"] = y - 0.05
            f[ep + "_conf_high"] = y + 0.05
            pred[ep] = 4 + x
        f.to_csv(root / f"{role}.csv", index=False)
        pred.to_csv(tmp_path / f"{role}-pred.csv", index=False)
    hashes = {
        f"{r}.csv": sha256_file(root / f"{r}.csv")
        for r in ["train", "validation", "calibration", "test"]
    }
    (root / "manifest.json").write_text(
        json.dumps({"status": "ready_for_fresh_fits", "file_sha256": hashes})
    )
    rec = {
        "train_sha256": hashes["train.csv"],
        "validation_sha256": hashes["validation.csv"],
        "calibration_or_test_used_for_training_or_checkpoint_selection": False,
        "checkpoint_sha256": {ep.split("_")[0]: "a" * 64 for ep in ACTIVITY_ENDPOINTS},
    }
    (tmp_path / "fit.json").write_text(json.dumps(rec))
    return root


def test_test_labels_do_not_fit_calibration(tmp_path):
    root = evaluation_fixture(tmp_path)

    def run(name):
        return evaluate(
            root,
            tmp_path / "calibration-pred.csv",
            tmp_path / "test-pred.csv",
            tmp_path / "fit.json",
            tmp_path / name,
        )

    a = run("a")
    f = pd.read_csv(root / "test.csv")
    for col in f.columns[2:]:
        f[col] += 2
    f.to_csv(root / "test.csv", index=False)
    m = json.loads((root / "manifest.json").read_text())
    m["file_sha256"]["test.csv"] = sha256_file(root / "test.csv")
    (root / "manifest.json").write_text(json.dumps(m))
    b = run("b")
    for ep in ACTIVITY_ENDPOINTS:
        assert a["endpoints"][ep]["parameters"] == b["endpoints"][ep]["parameters"]
        assert (
            a["endpoints"][ep]["historical_parameters"]
            == b["endpoints"][ep]["historical_parameters"]
        )
    assert a["macro_ST_RAE"] != b["macro_ST_RAE"]
    rec = json.loads((tmp_path / "fit.json").read_text())
    rec["train_sha256"] = "old-fold-model"
    (tmp_path / "fit.json").write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="this split"):
        run("bad")


def test_missing_predictions_fail_instead_of_dropping_rows(tmp_path):
    root = evaluation_fixture(tmp_path)
    f = pd.read_csv(tmp_path / "test-pred.csv").iloc[:-1]
    f.to_csv(tmp_path / "test-pred.csv", index=False)
    with pytest.raises(ValueError, match="exact partition"):
        evaluate(
            root,
            tmp_path / "calibration-pred.csv",
            tmp_path / "test-pred.csv",
            tmp_path / "fit.json",
            tmp_path / "out",
        )
