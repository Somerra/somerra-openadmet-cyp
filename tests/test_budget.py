import json

import pytest

from somerra_openadmet.budget import GPUBudget


def test_reservations_enforce_global_cap_and_release_unused_time(tmp_path):
    path = tmp_path / "budget.json"
    first = GPUBudget(path, tmp_path / "one", 50)
    first.reserve()
    with pytest.raises(ValueError, match="exceeds"):
        GPUBudget(path, tmp_path / "two", 30).reserve()
    assert first.finish() < 1
    second = GPUBudget(path, tmp_path / "two", 30)
    second.reserve()
    second.finish()
    assert len(json.loads(path.read_text())) == 2


def test_budget_counts_preledger_embedding_time(tmp_path):
    old = tmp_path / "regression-embeddings-v1"
    old.mkdir()
    (old / "manifest.json").write_text(json.dumps({"gpu_hours": 71}))
    with pytest.raises(ValueError, match="exceeds"):
        GPUBudget(tmp_path / "budget.json", tmp_path / "new", 2).reserve()


def test_interrupted_allocation_keeps_full_budget_charge(tmp_path):
    path = tmp_path / "budget.json"
    allocation = GPUBudget(path, tmp_path / "old", 71)
    allocation.reserve()
    record = json.loads(path.read_text())[str((tmp_path / "old").resolve())]
    assert record["status"] == "running"
    assert "used_hours" not in record and record["reserved_hours"] == 71
    with pytest.raises(ValueError, match="exceeds"):
        GPUBudget(path, tmp_path / "new", 2).reserve()


def test_ledger_location_is_independent_of_working_directory(tmp_path, monkeypatch):
    from somerra_openadmet.budget import budget_ledger_path

    monkeypatch.delenv("SOMERRA_GPU_LEDGER", raising=False)
    default = budget_ledger_path()
    monkeypatch.chdir(tmp_path)
    assert budget_ledger_path() == default
    shared = tmp_path / "shared.json"
    monkeypatch.setenv("SOMERRA_GPU_LEDGER", str(shared))
    assert budget_ledger_path() == shared
    monkeypatch.setenv("SOMERRA_GPU_LEDGER", "relative.json")
    with pytest.raises(ValueError, match="absolute"):
        budget_ledger_path()
