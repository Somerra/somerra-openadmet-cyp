import json
import sys
from pathlib import Path

import pytest

from somerra_openadmet import regression_task as task
from somerra_openadmet.model_recipe import resolve_model


def fitted(path, score, settings):
    path.mkdir()
    record = {
        "complete": True,
        "smoke": False,
        "final": False,
        "seed": 42,
        "batch_size": 128,
        "scheduler_horizon": 200,
        "completed_epochs": 100,
        "epoch_limit": 100,
        "model_config": resolve_model(settings),
        "loss_kind": "mae",
        "auxiliary_weight": 1.0,
        "best_epoch": 69,
        "internal_test_metrics": {"MA-ST-RAE": score, "endpoints": {"ep": score}},
    }
    (path / "run_manifest.json").write_text(json.dumps(record))
    (path / "best.ckpt").write_text("fake checkpoint")
    (path / "internal-test.csv").write_text("fake predictions")
    return path


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    reference = fitted(tmp_path / "reference", 0.7, {})
    task_path = tmp_path / "task.json"
    task_path.write_text("{}")
    contract = {
        "task_id": "frozen",
        "bundle": "bundle",
        "score": "MA-ST-RAE",
        "promotion": {},
        "campaign_ledger": str(tmp_path / "budget.json"),
    }
    (tmp_path / "registry.json").write_text(
        json.dumps({"task_id": "frozen", "incumbent": task.member(reference, {}), "trials": []})
    )
    monkeypatch.setattr(task, "read_task", lambda *a, **kw: contract)
    calls = []

    def train(task_path, task_doc, output, settings, hours, **kw):
        calls.append(settings)
        return fitted(output / "fit", 0.6, settings)

    monkeypatch.setattr(task, "train_supervised", train)
    monkeypatch.setattr(task, "compare_runs", lambda *a: {"delta_95ci": [-0.15, -0.05]})
    return task_path, contract, calls


def test_one_candidate_promotes_without_refitting_reference_or_final(frozen, tmp_path):
    task_path, contract, calls = frozen
    result = task.run_trial(task_path, tmp_path / "candidate", {"loss_normalization": "tasks"}, 1)
    assert len(calls) == 1 and calls[0]["loss_normalization"] == "tasks"
    assert result["decision"] == "keep" and result["score"] == 0.6
    state = task.read_registry(task_path, contract)
    assert state["incumbent"]["run"] == result["run"]
    assert len(state["trials"]) == 1
    assert not (tmp_path / "candidate/predictions").exists()
    assert (tmp_path / "candidate/model.json").exists()


def test_reproduction_ignores_train_and_never_promotes(frozen, tmp_path, monkeypatch):
    task_path, contract, calls = frozen
    first = tmp_path / "first"
    task.run_trial(task_path, first, {}, 1, promote=False)
    monkeypatch.setitem(sys.modules, "train", None)
    result = task.main(
        [
            "reproduce",
            "--task",
            str(task_path),
            "--trial",
            str(first),
            "--output-dir",
            str(tmp_path / "repeat"),
        ]
    )
    assert result["decision"] == "reproduced" and len(calls) == 2
    assert task.read_registry(task_path, contract)["incumbent"]["score"] == 0.7
    (first / "model.json").write_text("{}")
    with pytest.raises(ValueError, match="Saved recipe changed"):
        task.main(
            [
                "reproduce",
                "--preview",
                "--task",
                str(task_path),
                "--trial",
                str(first),
                "--output-dir",
                str(tmp_path / "not-created"),
            ]
        )


def test_plan_is_read_only_and_reproduction_preview_does_not_train(frozen, tmp_path, monkeypatch):
    task_path, contract, calls = frozen
    state_path = tmp_path / "registry.json"
    before = state_path.stat().st_mtime_ns
    result = task.plan(task_path, {})
    assert result["incumbent"]["score"] == 0.7
    assert calls == [] and state_path.stat().st_mtime_ns == before
    assert not (tmp_path / "registry.lock").exists()


def test_crash_keeps_incumbent_and_records_evidence(frozen, tmp_path, monkeypatch):
    task_path, contract, _ = frozen

    def fail(*a, **kw):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(task, "train_supervised", fail)
    with pytest.raises(RuntimeError, match="worker failed"):
        task.run_trial(task_path, tmp_path / "crash", {}, 1)
    state = task.read_registry(task_path, contract)
    assert state["incumbent"]["score"] == 0.7
    assert state["trials"][0]["decision"] == "crash"
    assert json.loads((tmp_path / "crash/trial.json").read_text())["status"] == "failed"


def test_completion_compares_with_latest_incumbent(frozen, tmp_path, monkeypatch):
    task_path, contract, _ = frozen

    def train(task_path, task_doc, output, settings, hours, **kw):
        better = fitted(tmp_path / "concurrent-winner", 0.5, {})
        with task.registry(task_path) as state:
            state["incumbent"] = task.member(better, {})
        return fitted(output / "fit", 0.6, settings)

    references = []

    def compare(bundle, reference, candidate):
        references.append(reference)
        return {"delta_95ci": [0.05, 0.15]}

    monkeypatch.setattr(task, "train_supervised", train)
    monkeypatch.setattr(task, "compare_runs", compare)
    result = task.run_trial(task_path, tmp_path / "candidate", {}, 1)
    assert references == [str(tmp_path / "concurrent-winner")]
    assert result["decision"] == "discard"
    assert task.read_registry(task_path, contract)["incumbent"]["score"] == 0.5


def test_recipe_cannot_override_evaluation():
    for values in [{"seed": 9}, {"epochs": 1}, {"partition": "random"}, {"data_sources": []}]:
        with pytest.raises(ValueError, match="evaluation is fixed"):
            resolve_model(values)
    for values in [{"depth": 0}, {"dropout": float("nan")}, {"loss_normalization": "unknown"}]:
        with pytest.raises(ValueError):
            resolve_model(values)


def test_task_changes_fail_and_model_source_is_separate(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    fixed = {"metric.py": "fixed"}
    contract = {
        "bundle": str(bundle),
        "bundle_sha256": task.sha256_file(bundle / "manifest.json"),
        "fixed_source_sha256": fixed,
    }
    contract["task_id"] = task.digest(contract)
    path = tmp_path / "task.json"
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(task, "load_frames", lambda *a: None)
    monkeypatch.setattr(
        task,
        "file_hashes",
        lambda paths: fixed if paths == task.FIXED_FILES else {"model.py": "changed"},
    )
    assert task.read_task(path)["task_id"] == contract["task_id"]
    monkeypatch.setattr(task, "file_hashes", lambda paths: {"metric.py": "changed"})
    with pytest.raises(ValueError, match="Frozen evaluation changed"):
        task.read_task(path)


def test_primary_routes_to_new_task_without_loading_legacy(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "primary", Path(__file__).parents[1] / "autoresearch.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setattr(task, "main", lambda argv: argv)
    monkeypatch.setitem(sys.modules, "legacy_train", None)
    assert entry.main(["plan", "--task", "frozen.json"]) == ["plan", "--task", "frozen.json"]


def test_supervisor_failed_launch_releases_reservation(tmp_path, monkeypatch):
    from somerra_openadmet.campaign_budget import CampaignLedger

    study = tmp_path / "study.json"
    study.write_text(
        json.dumps(
            {
                "status": "complete_predictions_prepared_not_submitted",
                "used_hours": 7.0,
                "gpu_limit_hours": 12,
                "runs": [],
            }
        )
    )
    ledger = CampaignLedger(tmp_path / "ledger.json")
    ledger.import_study(study)
    contract = {"campaign_ledger": str(ledger.path)}
    output = tmp_path / "trial"
    output.mkdir()
    monkeypatch.setattr(task, "read_task", lambda *a, **kw: contract)

    def fail(*a, **kw):
        raise OSError("launch failed")

    monkeypatch.setattr(task.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="launch failed"):
        task.train_supervised(tmp_path / "task.json", contract, output, resolve_model({}), 1)
    assert ledger.snapshot() == {"used_hours": 7.0, "reserved_hours": 0, "committed_hours": 7.0}


@pytest.mark.parametrize("weight", [0, -1, 17, True, float("nan"), float("inf")])
def test_invalid_scored_endpoint_weight(weight):
    with pytest.raises(ValueError):
        resolve_model({"scored_endpoint_weight": weight})


def test_legacy_member_defaults_scored_endpoint_weight(tmp_path):
    path = fitted(tmp_path / "legacy", 0.7, {})
    manifest = path / "run_manifest.json"
    record = json.loads(manifest.read_text())
    del record["model_config"]["scored_endpoint_weight"]
    manifest.write_text(json.dumps(record))
    assert task.member(path, {})["model_config"]["scored_endpoint_weight"] == 1.0
    with pytest.raises(ValueError, match="Recipe does not match"):
        task.member(path, {"scored_endpoint_weight": 2.0})
