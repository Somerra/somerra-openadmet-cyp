import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.cyp_research import (
    AUXILIARY,
    CYPS,
    accept_result,
    leaderboard,
    regression_metrics,
    split_data,
    tasks_for,
)
from somerra_openadmet.research_tracking import Tracker, ensure_private_project


def frames():
    official = pd.DataFrame(
        {
            "Molecule_Name": ["v", "alias", "t", "blind"],
            "parent": ["CC", "CC", "CCC", "CCCC"],
            "SMILES": ["CC", "CC", "CCC", "CCCC"],
        }
    )
    for ep in ACTIVITY_ENDPOINTS:
        official[ep] = [1.0, 1.0, 3.0, 5.0]
        official[ep + "_conf_low"] = official[ep] - 0.1
        official[ep + "_conf_high"] = official[ep] + 0.1
    pool = pd.concat(
        [
            official,
            official.iloc[[0]].assign(Molecule_Name="external-alias"),
            official.iloc[[0]].assign(Molecule_Name="external", parent="CCO"),
        ]
    )
    wb = pd.DataFrame({"molecule_name": official.Molecule_Name, "fold": [0, 1, 1, 2]})
    return official, pool, wb, pd.DataFrame({"parent": ["CCCC"]})


def test_split_uses_workbench_and_excludes_aliases_from_every_source():
    official, pool, wb, blind = frames()
    pool["fold"] = 0  # Expanded bundle fold numbers have no authority here.
    train, val, assignments = split_data(official, pool, wb, blind, 0)
    assert set(train.Molecule_Name) == {"t", "external"}
    assert list(val.Molecule_Name) == ["v"]
    assert (
        assignments.loc[assignments.Molecule_Name == "alias", "partition"].item()
        == "excluded_validation_alias"
    )
    assert train.sample_weight.eq(1).all()
    assert not set(train.parent) & set(val.parent)


def test_split_rejects_identity_and_label_mismatch():
    official, pool, wb, blind = frames()
    pool.loc[pool.Molecule_Name == "t", ACTIVITY_ENDPOINTS[0]] = 999
    with pytest.raises(ValueError, match="labels mismatch"):
        split_data(official, pool, wb, blind, 0)


def test_each_specialist_has_one_primary_and_all_22_auxiliaries():
    assert len(AUXILIARY) == len(set(AUXILIARY)) == 22
    for cyp in CYPS:
        tasks = tasks_for(cyp)
        assert len(tasks) == 23
        assert tasks[0].startswith(cyp)
        assert set(tasks) & set(ACTIVITY_ENDPOINTS) == {tasks[0]}


def test_metrics_interval_semantics_and_constant_prediction():
    ep = ACTIVITY_ENDPOINTS[0]
    truth = pd.DataFrame(
        {
            ep: [1.0, 3.0, np.nan],
            ep + "_conf_low": [0.5, 2.5, np.nan],
            ep + "_conf_high": [1.5, 3.5, np.nan],
        }
    )
    scores = regression_metrics(truth, [1.3, 2.9, 0], ep)
    assert scores["ST-RAE"] == 0
    assert scores["MAE"] == pytest.approx(0.2)
    assert scores["Spearman_R"] == pytest.approx(1)
    assert scores["Kendall_Tau"] == pytest.approx(1)
    assert scores["n"] == 2
    constant = regression_metrics(truth, [2, 2, 2], ep)
    assert constant["ST-RAE"] == pytest.approx(1)
    assert constant["R2"] == pytest.approx(0)
    assert constant["Spearman_R"] is None
    with pytest.raises(ValueError, match="Invalid primary predictions"):
        regression_metrics(truth, [np.nan, 3, 0], ep)


def test_promotion_separates_cyps_and_excludes_smoke_and_failed_trials():
    registry = {"best": {}}
    trial = {"cyp": "CYP1A2", "trial_id": "one", "output": "/tmp/one", "smoke": True}
    result = {"status": "complete", "metrics": {"ST-RAE": 0.5}, "best_epoch": 4}
    assert not accept_result(registry, trial, result)
    trial["smoke"] = False
    assert accept_result(registry, trial, result)
    assert not accept_result(registry, trial, result)
    assert leaderboard(registry)["MA-ST-RAE"] is None
    for cyp in CYPS[1:]:
        assert accept_result(registry, {**trial, "cyp": cyp}, result)
    assert leaderboard(registry)["MA-ST-RAE"] == 0.5
    assert not accept_result(registry, trial, {**result, "status": "failed"})


def test_local_tracking_does_not_require_wandb(tmp_path):
    tracker = Tracker(
        tmp_path, {}, run_id="test", group="group", cyp="CYP1A2", config={}, mode="disabled"
    )
    tracker.log({"epoch": 1, "val/ST-RAE": 0.7, "val/Spearman_R": None})
    assert json.loads((tmp_path / "events.jsonl").read_text())["val/ST-RAE"] == 0.7


def test_private_project_required(monkeypatch):
    pytest.importorskip("wandb")  # Initializes the SDK's vendored GraphQL imports.

    class Client:
        def execute(self, *args, **kwargs):
            return {"project": {"access": "PUBLIC"}}

    class API:
        client = Client()

    class Wandb:
        def Api(self, **kwargs):
            return API()

    with pytest.raises(ValueError, match="must be private"):
        ensure_private_project(Wandb(), "entity", "project")


def test_masked_loss_scaling_and_batch_coverage():
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from somerra_openadmet.cyp_research_worker import batch_indices, loss_parts, normalize

    prediction = torch.tensor([[1.0, 1.0], [1.0, 9.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0], [float("nan"), float("nan")]])
    loss, _, count = loss_parts(prediction, target, torch.tensor([3.0, 0.3]))
    loss.backward()
    assert loss.item() == pytest.approx(1.65)
    assert prediction.grad[0, 0] / prediction.grad[0, 1] == pytest.approx(10.0)
    assert prediction.grad[1].eq(0).all()
    assert count.tolist() == [1, 1]
    for n in [3, 32, 33, 65]:
        batches = batch_indices(n, 32)
        assert len(np.concatenate(batches)) == n
        assert all(len(b) > 1 for b in batches)
    mean, scale = normalize(np.array([[1.0, 2.0], [3.0, 6.0], [np.nan, 4.0]]))
    assert mean.tolist() == [2.0, 4.0]
    assert scale[0] == 1.0
    with pytest.raises(ValueError, match="constant"):
        normalize(np.array([[1.0, 2.0], [1.0, 3.0]]))


def test_official_st_rae_parity():
    import importlib.util

    path = Path("runs/competition-recheck-20261026/custom_scoring_functions.py")
    if not path.exists():
        pytest.skip("Private pinned evaluator not present")
    spec = importlib.util.spec_from_file_location("official_metric", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ep = ACTIVITY_ENDPOINTS[0]
    rng = np.random.default_rng(42)
    y = rng.normal(4, 2, 100)
    pred = y + rng.normal(size=100)
    lo, hi = y - 0.2, y + 0.4
    truth = pd.DataFrame({ep: y, ep + "_conf_low": lo, ep + "_conf_high": hi})
    expected = module.rae_soft_threshold_absolute_error(y, pred, y_true_upper=hi, y_true_lower=lo)
    assert regression_metrics(truth, pred, ep)["ST-RAE"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("failure", ["privacy", "network", "init"])
def test_wandb_initialization_falls_back_without_bypassing_privacy(tmp_path, monkeypatch, failure):
    import sys
    from types import SimpleNamespace

    from somerra_openadmet import research_tracking

    initialized = []

    def privacy(*args):
        if failure == "privacy":
            raise ValueError("Public project")
        if failure == "network":
            raise ConnectionError("Cannot verify privacy")

    def init(**kwargs):
        initialized.append(True)
        raise ConnectionError("Unavailable")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **kw: kw))
    monkeypatch.setattr(research_tracking, "ensure_private_project", privacy)
    tracker = Tracker(
        tmp_path,
        {"entity": "private", "project": "test"},
        run_id="t",
        group="g",
        cyp="CYP1A2",
        config={},
        mode="online",
    )
    tracker.log({"epoch": 1, "val/ST-RAE": 0.5})
    tracker.finish({"status": "complete"})
    assert initialized == ([True] if failure == "init" else [])
    assert json.loads((tmp_path / "events.jsonl").read_text())["val/ST-RAE"] == 0.5
    assert (
        json.loads((tmp_path / "tracking-error.json").read_text())["effective_mode"] == "disabled"
    )


@pytest.fixture
def fivefold_frames():
    official = pd.DataFrame(
        {
            "Molecule_Name": [f"m{i}" for i in range(10)],
            "SMILES": ["C" * (i + 1) for i in range(10)],
            "parent": ["C" * (i + 1) for i in range(10)],
        }
    )
    for ep in ACTIVITY_ENDPOINTS:
        official[ep] = np.arange(10, dtype=float)
        official[ep + "_conf_low"] = official[ep] - 0.1
        official[ep + "_conf_high"] = official[ep] + 0.1
    pool = official.copy()
    for task in AUXILIARY:
        pool[task] = np.arange(10, dtype=float)
    wb = pd.DataFrame({"molecule_name": official.Molecule_Name, "fold": np.repeat(range(5), 2)})
    blind = pd.DataFrame({"Molecule_Name": ["blind"], "SMILES": ["N"], "parent": ["N"]})
    return official, pool, wb, blind


def test_three_training_folds_exclude_both_heldouts_and_external_aliases(fivefold_frames):
    from somerra_openadmet.cyp_research import evaluation_rows

    official, pool, wb, blind = fivefold_frames
    # An official training-fold alias and external aliases must all be excluded.
    official.loc[4, "parent"] = official.loc[2, "parent"]
    pool.loc[4, "parent"] = official.loc[2, "parent"]
    pool = pd.concat(
        [
            pool,
            pool.iloc[[0]].assign(Molecule_Name="aux-val"),
            pool.iloc[[2]].assign(Molecule_Name="aux-test"),
            pool.iloc[[2]].assign(Molecule_Name="aux-blind", parent="N"),
        ]
    )
    train, val, assignments = split_data(official, pool, wb, blind, 0, 1)
    test = evaluation_rows(official, pool, assignments, "test")
    assert set(train.Molecule_Name) == {f"m{i}" for i in range(5, 10)}
    assert list(val.Molecule_Name) == ["m0", "m1"]
    assert list(test.Molecule_Name) == ["m2", "m3"]
    assert assignments.set_index("Molecule_Name").loc["m4", "partition"] == "excluded_test_alias"
    assert set(assignments.loc[assignments.partition == "train", "workbench_fold"]) == {2, 3, 4}
    assert not set(train.parent) & (set(test.parent) | set(val.parent) | set(blind.parent))


@pytest.mark.parametrize("test_fold", [0, 5, -1, True, 1.5])
def test_invalid_test_fold_rejected(fivefold_frames, test_fold):
    with pytest.raises(ValueError, match="distinct saved folds"):
        split_data(*fivefold_frames, 0, test_fold)


def test_validation_test_parent_overlap_rejected(fivefold_frames):
    official, pool, wb, blind = fivefold_frames
    official.loc[2, "parent"] = official.loc[0, "parent"]
    pool.loc[2, "parent"] = official.loc[0, "parent"]
    with pytest.raises(ValueError, match="parents overlap"):
        split_data(official, pool, wb, blind, 0, 1)


def test_prepare_records_three_way_split_and_refuses_inplace_revision(fivefold_frames, tmp_path):
    from somerra_openadmet.cyp_research import prepare, read, verify, write
    from somerra_openadmet.download import sha256_file

    official, pool, wb, blind = fivefold_frames
    source, workbench = tmp_path / "source", tmp_path / "wb"
    source.mkdir()
    workbench.mkdir()
    for name, frame in {"official": official, "pool": pool, "test": blind}.items():
        frame.to_csv(source / f"{name}.csv", index=False)
    wb.to_csv(workbench / "mt_aux8.csv", index=False)
    write(
        source / "manifest.json",
        {
            "tasks": [*ACTIVITY_ENDPOINTS, *AUXILIARY],
            "file_sha256": {p.name: sha256_file(p) for p in source.glob("*.csv")},
        },
    )
    write(
        workbench / "manifest.json",
        {"file_sha256": {"mt_aux8.csv": sha256_file(workbench / "mt_aux8.csv")}},
    )
    config = {
        "source_dir": str(source),
        "workbench_data": str(workbench),
        "output_dir": str(tmp_path / "research"),
        "fold": 0,
        "test_fold": 1,
        "seed": 42,
        "wandb": {"mode": "disabled"},
        "campaigns": {c: {"budget_gpu_hours": 0, "trial_seconds": 1800} for c in CYPS},
    }
    root = prepare(config)
    manifest = read(root / "manifest.json")
    assert manifest["fold_roles"] == {"train": [2, 3, 4], "validation": 0, "test": 1}
    assert manifest["rows"] == {"train": 6, "validation": 2, "internal-test": 2}
    assert manifest["protocol"] == "cyp-recipes-v3"
    assert prepare(config) == root
    with pytest.raises(ValueError, match="fresh output"):
        prepare({**config, "test_fold": 2})
    with (root / "internal-test.csv").open("a") as handle:
        handle.write("changed")
    with pytest.raises(ValueError, match="Prepared data changed"):
        verify(root)
