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
    allocation.record_interruption("Verified original process exited; elapsed time unavailable")
    record = json.loads(path.read_text())[str((tmp_path / "old").resolve())]
    assert record["status"] == "interrupted_unknown_usage"
    assert "used_hours" not in record and record["reserved_hours"] == 71
    with pytest.raises(ValueError, match="exceeds"):
        GPUBudget(path, tmp_path / "new", 2).reserve()
