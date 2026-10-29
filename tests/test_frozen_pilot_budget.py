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
    monkeypatch.setattr(driver, "LEDGER", ledger)
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


def test_reservation_atomically_enforces_original_pilot_cap(tmp_path):
    from somerra_openadmet.frozen_pilot_budget import PilotReservation

    ledger = tmp_path / "ledger.json"
    prefix = tmp_path / "regression-frozen-representation-pilot-v1"
    old = "/previous-checkout/runs/regression-frozen-representation-pilot-v1/historical"
    ledger.write_text(json.dumps({old: {"reserved_hours": 3.8}}))
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
    campaign, historical = tmp_path / "campaign.json", tmp_path / "legacy.json"
    campaign.write_text(
        json.dumps({"version": 1, "limit_hours": 12.0, "historical_hours": 0.0, "reservations": {}})
    )
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
