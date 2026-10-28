import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from somerra_openadmet import regression_block as block
from somerra_openadmet.model_recipe import resolve_model


def test_initial_schedule_changes_one_axis_and_is_unique():
    base = resolve_model({})
    proposals = block.proposals(base)
    assert len(proposals) == 25
    assert len({json.dumps(p["model"], sort_keys=True) for p in proposals}) == 25
    for proposal in proposals:
        assert sum(value != base[key] for key, value in proposal["model"].items()) == 1
        assert "seed" not in proposal["model"]


def test_follow_ups_use_validation_not_internal_test():
    base = resolve_model({})
    rows = [
        {**p, "validation": 0.6 + i / 100, "score": 0.1 * i}
        for i, p in enumerate(block.proposals(base))
    ]
    before = block.refinements(base, rows, 0.661)
    changed = deepcopy(rows)
    for row in changed:
        row["score"] = 10 - row["score"]
    assert block.refinements(base, changed, 0.661) == before
    assert all(p["model"] == resolve_model(p["model"]) for p in before)


def test_allowance_preserves_deadline_and_report_margin():
    assert block.allowance(700, []) is None
    assert block.allowance(800, []) * 3600 <= 680
    assert block.allowance(2000, [1000]) == 0.4
    assert block.allowance(1500, [1000]) is None


@pytest.mark.parametrize("code", [0, 1])
def test_block_only_invokes_candidates_and_stops(tmp_path, monkeypatch, code):
    base = resolve_model({})
    incumbent = {"run": "reference", "score": 0.66, "model_config": base}
    monkeypatch.setattr(block, "read_task", lambda *a, **kw: {"campaign_ledger": "ledger"})
    monkeypatch.setattr(block, "read_registry", lambda *a: {"incumbent": incumbent})
    monkeypatch.setattr(block, "checked_run", lambda *a: {"best_score": 0.661, "best_epoch": 69})
    monkeypatch.setattr(
        block, "CampaignLedger", lambda *a: SimpleNamespace(snapshot=lambda: {"committed_hours": 8})
    )
    monkeypatch.setattr(block.shutil, "disk_usage", lambda *a: SimpleNamespace(free=20 * 1024**3))
    monkeypatch.setattr(block.subprocess, "check_output", lambda *a, **kw: "commit")
    calls = []

    def launch(command, **kwargs):
        from pathlib import Path

        calls.append(command)
        assert command[2] == "run"
        path = Path(command[command.index("--output-dir") + 1])
        path.mkdir()
        (path / "trial.json").write_text(
            json.dumps(
                {
                    "status": "complete" if code == 0 else "failed",
                    "score": 0.67,
                    "decision": "discard" if code == 0 else "crash",
                    "run": str(path / "fit"),
                }
            )
        )
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(block.subprocess, "run", launch)
    result = block.run(
        tmp_path / "task.json", tmp_path / "block", hours=4, max_trials=2 if code == 0 else 40
    )
    assert len(calls) == 2
    assert result["status"] == (
        "completed_trial_limit" if code == 0 else "stopped_repeated_failures"
    )
    assert (tmp_path / "block/report.html").exists()
    assert json.loads((tmp_path / "block/recommendation.json").read_text()) == incumbent


def test_block_rejects_expansion_beyond_authorized_limit(tmp_path):
    with pytest.raises(ValueError):
        block.run(tmp_path, tmp_path, hours=5)
