import json
from pathlib import Path

import pytest


def test_pilot_charges_failures_and_diagnostics_without_resetting_history(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_frozen_representation_pilot as driver

    root = tmp_path / "pilot"
    extraction = tmp_path / "extraction"
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                str(root / "fit"): {
                    "reserved_hours": 0.1,
                    "used_hours": 0.02,
                    "status": "finished",
                },
                str(extraction): {"reserved_hours": 1.5, "used_hours": 0.03, "status": "finished"},
                str(extraction) + "-diagnostics": {"reserved_hours": 0.25, "status": "interrupted"},
                str(tmp_path / "unrelated"): {"reserved_hours": 10, "status": "running"},
            }
        )
    )
    monkeypatch.setattr(driver, "ROOT", root)
    monkeypatch.setattr(driver, "EXTRACTION_PREFIX", extraction)
    monkeypatch.setattr(driver, "pilot_ledger_path", lambda config: ledger)
    assert driver.charged_hours() == pytest.approx(0.30)
    assert len(list(driver.matrix())) == 21


def test_partial_pilot_cannot_export_blind_predictions(tmp_path):
    pytest.importorskip("torch")
    from somerra_openadmet.representation_training import predict_representations

    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "complete": False,
                "pilot_complete": True,
                "requested_folds_complete": True,
                "completed_folds": [0],
            }
        )
    )
    config = tmp_path / "predict.json"
    config.write_text(
        json.dumps(
            {
                "run_dir": str(run),
                "blind_csv": "unused",
                "output_dir": "unused",
                "max_gpu_hours": 0.1,
            }
        )
    )
    with pytest.raises(ValueError, match="completed five-fold"):
        predict_representations(config)


def test_reservation_atomically_enforces_original_pilot_cap(tmp_path, monkeypatch):
    from somerra_openadmet.frozen_pilot_budget import PilotReservation, bind_pilot_history

    campaign = initialize_campaign(tmp_path, monkeypatch)
    ledger = campaign.parent / "regression-gpu-budget.json"
    prefix = tmp_path / "regression-frozen-representation-pilot-v1"
    old = "/previous-checkout/runs/regression-frozen-representation-pilot-v1/historical"
    ledger.write_text(json.dumps({old: {"reserved_hours": 3.8}}))
    bind_pilot_history({}, ledger)
    config = {"ledger": str(ledger), "output_dir": str(prefix / "a"), "max_gpu_hours": 0.15}
    first = PilotReservation(config)
    first.reserve()
    with pytest.raises(ValueError, match="four-GPU-hour"):
        PilotReservation({**config, "output_dir": str(prefix / "b")}).reserve()
    first.finish()
    assert json.loads(ledger.read_text())[old] == {"reserved_hours": 3.8}
    with pytest.raises(ValueError, match="namespace"):
        PilotReservation({**config, "output_dir": str(tmp_path / "unaccounted")})


@pytest.mark.parametrize("exit_code", [0, 1])
def test_supervisor_charges_both_ledgers_after_child_exit(tmp_path, monkeypatch, exit_code):
    pytest.importorskip("torch")
    from somerra_openadmet import representation_budget as supervisor
    from somerra_openadmet.campaign_budget import CampaignLedger

    for name in ("SOMERRA_CAMPAIGN_LEDGER", "SOMERRA_GPU_LEDGER", "SOMERRA_REPRESENTATION_WORKER"):
        monkeypatch.delenv(name, raising=False)
    campaign, historical = tmp_path / "campaign.json", tmp_path / "regression-gpu-budget.json"
    historical.write_text(
        json.dumps(
            {"/old/runs/regression-frozen-representation-pilot-v1/fit": {"reserved_hours": 0.38}}
        )
    )
    campaign.write_text(
        json.dumps({"version": 1, "limit_hours": 12.0, "historical_hours": 0.0, "reservations": {}})
    )
    from somerra_openadmet.frozen_pilot_budget import bind_pilot_history

    bind_pilot_history({"campaign_ledger": str(campaign)}, historical)
    output = tmp_path / "regression-frozen-representation-pilot-v1" / "test"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "device": "cuda",
                "max_gpu_hours": 0.1,
                "output_dir": str(output),
                "ledger": str(historical),
                "campaign_ledger": str(campaign),
            }
        )
    )

    class Worker:
        def __init__(self, *args, **kwargs):
            CampaignLedger(campaign).claim_worker(output, 0.1)
            output.mkdir()
            (output / "worker-result.json").write_text(json.dumps({"complete": True}))
            self.returncode = exit_code

        def wait(self, **kwargs):
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(supervisor.subprocess, "Popen", Worker)
    if exit_code:
        with pytest.raises(RuntimeError, match="worker failed"):
            supervisor.maybe_supervise("frozen-pilot", config)
    else:
        assert supervisor.maybe_supervise("frozen-pilot", config)["complete"]
    primary = json.loads(campaign.read_text())["reservations"][str(output)]
    legacy = json.loads(historical.read_text())[str(output)]
    assert primary["status"] == "exited" and primary["used_hours"] >= 0
    assert legacy["status"] == "finished" and legacy["used_hours"] >= 0
    assert primary["reserved_hours"] == legacy["reserved_hours"] == 0.1


def initialize_campaign(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign.json"
    campaign.write_text(
        json.dumps({"version": 1, "limit_hours": 12.0, "historical_hours": 0.0, "reservations": {}})
    )
    monkeypatch.setenv("SOMERRA_CAMPAIGN_LEDGER", str(campaign))
    return campaign


def test_shared_pilot_history_survives_checkout_and_cwd_changes(tmp_path, monkeypatch):
    from somerra_openadmet.frozen_pilot_budget import (
        PilotReservation,
        bind_pilot_history,
        pilot_ledger_path,
    )

    campaign = initialize_campaign(tmp_path, monkeypatch)
    historical = campaign.parent / "regression-gpu-budget.json"
    old = "/old-checkout/runs/regression-frozen-representation-pilot-v1/fit"
    historical.write_text(json.dumps({old: {"reserved_hours": 3.8}}))
    bind_pilot_history({}, historical)
    for checkout in ("checkout-a", "checkout-b"):
        local = tmp_path / checkout
        (local / "runs").mkdir(parents=True)
        # This stale/empty local history must never override the shared history.
        (local / "runs/regression-gpu-budget.json").write_text("{}")
        monkeypatch.chdir(local)
        assert pilot_ledger_path({}) == historical
        config = {
            "output_dir": "runs/regression-frozen-representation-pilot-v1/next",
            "max_gpu_hours": 0.15,
        }
        if checkout == "checkout-a":
            PilotReservation(config).reserve()
        else:
            with pytest.raises(ValueError, match="four-GPU-hour"):
                PilotReservation(config).reserve()
        for override in (
            "runs/regression-gpu-budget.json",
            str(local / "runs/regression-gpu-budget.json"),
        ):
            with pytest.raises(ValueError, match="absolute history"):
                PilotReservation({**config, "ledger": override})
    assert json.loads(historical.read_text())[old] == {"reserved_hours": 3.8}


def test_pilot_refuses_missing_or_empty_shared_history(tmp_path, monkeypatch):
    from somerra_openadmet.frozen_pilot_budget import PilotReservation, bind_pilot_history

    campaign = initialize_campaign(tmp_path, monkeypatch)
    historical = campaign.parent / "regression-gpu-budget.json"
    config = {
        "output_dir": str(tmp_path / "regression-frozen-representation-pilot-v1/fit"),
        "max_gpu_hours": 0.1,
    }
    with pytest.raises(ValueError, match="Bind the existing"):
        PilotReservation(config)
    assert not historical.exists()
    with pytest.raises(FileNotFoundError):
        bind_pilot_history({}, historical)
    historical.write_text("{}")
    with pytest.raises(ValueError, match="missing or invalid"):
        bind_pilot_history({}, historical)
    old = "/old/runs/regression-frozen-representation-pilot-v1/fit"
    historical.write_text(json.dumps({old: {"reserved_hours": 0.38}}))
    bind_pilot_history({}, historical)
    historical.unlink()
    with pytest.raises(ValueError, match="Restore the existing"):
        PilotReservation(config)
    historical.write_text("{}")
    with pytest.raises(ValueError, match="history is missing"):
        PilotReservation(config).reserve()
    assert historical.read_text() == "{}"
    allocation = PilotReservation(config)
    historical.unlink()
    with pytest.raises(FileNotFoundError):
        allocation.reserve()
    assert not historical.exists()


def test_binding_preserves_both_ledgers_and_cannot_be_redirected(tmp_path, monkeypatch):
    from somerra_openadmet.frozen_pilot_budget import bind_pilot_history, pilot_ledger_path

    campaign = initialize_campaign(tmp_path, monkeypatch)
    original = json.loads(campaign.read_text())
    history = tmp_path / "original-checkout" / "legacy.json"
    history.parent.mkdir()
    history.write_text(
        json.dumps(
            {"/old/runs/regression-frozen-representation-pilot-v1/fit": {"reserved_hours": 0.38}}
        )
    )
    before = history.read_bytes()
    bind_pilot_history({}, history)
    bound = campaign.read_bytes()
    state = json.loads(bound)
    binding = state.pop("frozen_pilot_history")
    assert state == original
    assert binding["bound_committed_hours"] == pytest.approx(0.38)
    assert pilot_ledger_path({}) == history
    assert history.read_bytes() == before
    bind_pilot_history({}, history)
    assert campaign.read_bytes() == bound
    alternate = tmp_path / "other.json"
    alternate.write_bytes(before)
    with pytest.raises(ValueError, match="already bound"):
        bind_pilot_history({}, alternate)
    assert campaign.read_bytes() == bound
