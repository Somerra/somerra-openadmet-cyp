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
    status = json.loads((tmp_path / "regression-trial-inputs/trial.json").read_text())
    assert status["status"] == "completed"
    assert status["score"] == 0.8
    assert status["score_kind"] == "nested_ensemble"
    assert status["score_metric"] == "MA-ST-RAE"
    assert status["score_direction"] == "minimize"
    assert status["decision"] == "pending review"
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


def trial_args(tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text('{"models": ["ridge"], "data_sources": ["official_direct"]}')
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"prepared_dir": "missing-on-purpose"}')
    return dict(backend="cpu", model_recipe=recipe, overrides={}, recipe_source=recipe), catalog


def test_plan_does_not_train_or_create_artifacts(tmp_path, monkeypatch):
    from somerra_openadmet import campaign, research

    kwargs, catalog = trial_args(tmp_path)
    monkeypatch.setattr(campaign, "evaluate", lambda **kw: pytest.fail("preview trained"))
    output = tmp_path / "trial"
    plan = research.main(
        **kwargs, argv=["plan", "--data-config", str(catalog), "--output-dir", str(output)]
    )
    assert plan["selected_data_sources"] == ["official_direct"]
    assert not output.exists()
    assert not (tmp_path / "trial-inputs").exists()


@pytest.mark.parametrize("error", [RuntimeError("test failure"), KeyboardInterrupt()])
def test_failed_trial_preserves_evidence(tmp_path, monkeypatch, error):
    from somerra_openadmet import campaign, research

    kwargs, catalog = trial_args(tmp_path)

    def fail(**kw):
        raise error

    monkeypatch.setattr(campaign, "evaluate", fail)
    with pytest.raises(type(error)):
        research.main(
            **kwargs,
            argv=["run", "--data-config", str(catalog), "--output-dir", str(tmp_path / "trial")],
        )
    status = json.loads((tmp_path / "trial-inputs/trial.json").read_text())
    assert status["status"] == "failed"
    assert status["elapsed_seconds"] >= 0
    assert "score" not in status


def test_nonfinite_trial_is_failure(tmp_path, monkeypatch):
    from somerra_openadmet import campaign, research

    kwargs, catalog = trial_args(tmp_path)
    monkeypatch.setattr(
        campaign, "evaluate", lambda **kw: {"macro": {"nested_ensemble": float("nan")}}
    )
    with pytest.raises(ValueError, match="Non-finite"):
        research.main(
            **kwargs,
            argv=["run", "--data-config", str(catalog), "--output-dir", str(tmp_path / "trial")],
        )
    status = json.loads((tmp_path / "trial-inputs/trial.json").read_text())
    assert status["status"] == "failed"


def test_reproduce_ignores_research_settings_and_uses_same_evaluator(tmp_path, monkeypatch):
    from somerra_openadmet import campaign, research

    kwargs, catalog = trial_args(tmp_path)
    saved = kwargs["model_recipe"]
    kwargs.update(
        backend="neural",
        model_recipe=tmp_path / "missing.json",
        overrides={"models": ["invalid"]},
        recipe_source=tmp_path / "broken.py",
    )
    calls = []

    def evaluate(**kw):
        calls.append(kw)
        assert json.loads(kw["model_config"].read_text())["models"] == ["ridge"]
        return {"macro": {"nested_ensemble": 0.75}}

    monkeypatch.setattr(campaign, "evaluate", evaluate)
    record = research.main(
        **kwargs,
        argv=[
            "reproduce",
            "--model-config",
            str(saved),
            "--backend",
            "cpu",
            "--data-config",
            str(catalog),
            "--output-dir",
            str(tmp_path / "repeat"),
        ],
    )
    assert len(calls) == 1
    assert record["workflow"] == "reproduction"
    assert record["score"] == 0.75
    assert (tmp_path / "repeat-inputs/recipe.json").read_text() == saved.read_text()
    assert not (tmp_path / "repeat-inputs/train.py").exists()


def test_primary_reproduce_preview_does_not_import_train(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "autoresearch.py"
    spec = importlib.util.spec_from_file_location("research_entry", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    kwargs, catalog = trial_args(tmp_path)
    monkeypatch.setitem(sys.modules, "train", None)
    result = entry.main(
        [
            "reproduce",
            "--preview",
            "--model-config",
            str(kwargs["model_recipe"]),
            "--backend",
            "cpu",
            "--data-config",
            str(catalog),
            "--output-dir",
            str(tmp_path / "repeat"),
        ]
    )
    assert result["workflow"] == "reproduction"
    assert not (tmp_path / "repeat-inputs").exists()


def test_reproduce_requires_explicit_recipe_and_backend(tmp_path):
    from somerra_openadmet import research

    kwargs, _ = trial_args(tmp_path)
    with pytest.raises(SystemExit):
        research.main(**kwargs, argv=["reproduce", "--output-dir", str(tmp_path / "repeat")])


@pytest.mark.parametrize("complete", [False, True])
def test_shared_neural_evaluator_enforces_completion(tmp_path, monkeypatch, complete):
    from types import SimpleNamespace

    from somerra_openadmet.evaluation import evaluate_recipe

    monkeypatch.setitem(
        sys.modules,
        "somerra_openadmet.neural",
        SimpleNamespace(screen=lambda **kw: {"complete": complete, "MA-ST-RAE": 0.8}),
    )
    kwargs = dict(
        backend="neural", model_config=tmp_path, data_config=tmp_path, output_dir=tmp_path
    )
    if complete:
        _, score = evaluate_recipe(**kwargs)
        assert score == 0.8
    else:
        with pytest.raises(ValueError, match="Incomplete"):
            evaluate_recipe(**kwargs)
