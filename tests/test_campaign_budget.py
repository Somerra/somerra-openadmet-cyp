import json
import multiprocessing

import pytest

from somerra_openadmet.campaign_budget import CampaignLedger, WorkerBudget


def initialized(tmp_path, used=7.364):
    study = tmp_path / "study.json"
    study.write_text(
        json.dumps(
            {
                "status": "complete_predictions_prepared_not_submitted",
                "used_hours": used,
                "gpu_limit_hours": 12,
                "runs": [],
            }
        )
    )
    ledger = CampaignLedger(tmp_path / "ledger.json")
    ledger.import_study(study)
    return ledger, study


def reserve_competing(path, run, barrier, queue):
    barrier.wait()
    try:
        CampaignLedger(path).reserve_many([(run, 2)])
        queue.put("reserved")
    except ValueError:
        queue.put("rejected")


def test_restart_and_worker_exit_do_not_reset_allocation(tmp_path):
    ledger, study = initialized(tmp_path)
    run = tmp_path / "first"
    ledger.reserve_many([(run, 2)])
    worker = WorkerBudget(ledger.path, run, 2)
    worker.reserve()
    worker.finish()
    restarted = CampaignLedger(ledger.path)
    assert restarted.snapshot()["committed_hours"] == pytest.approx(9.364)
    with pytest.raises(ValueError, match="exceeds"):
        restarted.reserve_many([(tmp_path / "second", 3)])
    with pytest.raises(FileExistsError):
        restarted.import_study(study)
    with pytest.raises(ValueError, match="unclaimed"):
        WorkerBudget(ledger.path, run, 2).reserve()
    ledger.settle(run, 0.3, "finished")
    assert restarted.snapshot()["committed_hours"] == pytest.approx(7.664)
    restarted.reserve_many([(tmp_path / "second", 3)])


def test_concurrent_coordinators_reserve_atomically(tmp_path):
    ledger, _ = initialized(tmp_path, used=9)
    ctx = multiprocessing.get_context("spawn")
    barrier, queue = ctx.Barrier(2), ctx.Queue()
    processes = [
        ctx.Process(target=reserve_competing, args=(ledger.path, tmp_path / str(i), barrier, queue))
        for i in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert sorted([queue.get(timeout=1), queue.get(timeout=1)]) == ["rejected", "reserved"]
    assert ledger.snapshot()["committed_hours"] == 11


def test_atomic_group_reservation_and_abandoned_worker_charge(tmp_path):
    ledger, _ = initialized(tmp_path, used=9)
    with pytest.raises(ValueError, match="exceeds"):
        ledger.reserve_many([(tmp_path / "one", 2), (tmp_path / "two", 2)])
    assert ledger.snapshot()["committed_hours"] == 9
    ledger.reserve_many([(tmp_path / "one", 2)])
    ledger.claim_worker(tmp_path / "one", 2)
    assert CampaignLedger(ledger.path).snapshot()["committed_hours"] == 11
    with pytest.raises(ValueError, match="unclaimed"):
        WorkerBudget(ledger.path, tmp_path / "unreserved", 1).reserve()
    with pytest.raises(ValueError, match="Invalid"):
        ledger.settle(tmp_path / "one", float("nan"), "finished")


def test_failed_launch_releases_unstarted_reservation(tmp_path, monkeypatch):
    from somerra_openadmet.hillclimb import run_campaign

    ledger, _ = initialized(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")

    def fail(*args, **kwargs):
        raise OSError("launch failed")

    monkeypatch.setattr("somerra_openadmet.hillclimb.subprocess.Popen", fail)
    with pytest.raises(OSError, match="launch failed"):
        run_campaign(bundle, tmp_path / "campaign", tmp_path, ledger.path)
    assert ledger.snapshot()["committed_hours"] == pytest.approx(7.364)


def test_explicit_limit_amendment_preserves_spending(tmp_path):
    ledger, _ = initialized(tmp_path, used=11)
    with pytest.raises(ValueError, match="exceeds"):
        ledger.reserve_many([(tmp_path / "blocked", 2)])
    ledger.amend_limit(20, "User approved 20 hours for Mol-JEPA recovery")
    assert ledger.snapshot()["committed_hours"] == 11
    ledger.reserve_many([(tmp_path / "allowed", 2)])
    assert CampaignLedger(ledger.path).snapshot()["committed_hours"] == 13
    with pytest.raises(ValueError, match="below committed"):
        ledger.amend_limit(12, "Cannot discard charges")
    with pytest.raises(ValueError, match="authorization"):
        ledger.amend_limit(24, "")
