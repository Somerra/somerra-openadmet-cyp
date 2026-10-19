import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "research_train", Path(__file__).parents[1] / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_trial_snapshots_inputs(tmp_path, monkeypatch):
    from somerra_openadmet import campaign

    runner = load_runner()
    recipe, profile = tmp_path / "recipe.json", tmp_path / "profile.json"
    recipe.write_text('{"models": ["ridge"], "data_sources": ["official_direct"]}')
    profile.write_text('{"prepared_dir": "unchanged-data"}')
    runner.MODEL_RECIPE = recipe
    runner.MODEL_OVERRIDES = {"n_jobs": 1}
    output = tmp_path / "regression-trial"
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        assert json.loads(kwargs["model_config"].read_text())["n_jobs"] == 1
        assert kwargs["data_config"].read_text() == profile.read_text()
        assert kwargs["output_dir"] == output
        return {"macro": {"nested_ensemble": 0.8}}

    monkeypatch.setattr(campaign, "evaluate", fake_run)
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--data-config", str(profile), "--output-dir", str(output)]
    )
    runner.main()
    assert len(calls) == 1
    assert (tmp_path / "regression-trial-inputs/train.py").exists()
    with pytest.raises(FileExistsError):
        runner.main()


def test_partial_folds_not_a_research_trial(tmp_path, monkeypatch):
    runner = load_runner()
    recipe = tmp_path / "recipe.json"
    recipe.write_text('{"selected_folds": [4]}')
    runner.MODEL_RECIPE = recipe
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--data-config", "unused.json", "--output-dir", str(tmp_path / "run")],
    )
    with pytest.raises(ValueError, match="all folds"):
        runner.main()
