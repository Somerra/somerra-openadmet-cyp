"""Recipe freedom, campaign isolation, and the scientific handoff to a private build."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from somerra_openadmet import cyp_submission
from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.cyp_research import (
    CYPS,
    RUNTIME_FILES,
    fixed_hashes,
    read,
    run_trial,
    status,
    tasks_for,
    write,
)
from somerra_openadmet.download import sha256_file


def runtime_hashes():
    base = Path(cyp_submission.__file__).parent
    return {n: sha256_file(base / n) for n in RUNTIME_FILES}


@pytest.fixture
def data_roots(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    official = pd.DataFrame(
        {
            "Molecule_Name": [f"m{i}" for i in range(10)],
            "SMILES": ["C" * (i + 1) for i in range(10)],
            "parent": ["C" * (i + 1) for i in range(10)],
        }
    )
    for i, ep in enumerate(ACTIVITY_ENDPOINTS):
        official[ep] = np.arange(10, dtype=float) + i
        official[ep + "_conf_low"] = official[ep] - 0.1
        official[ep + "_conf_high"] = official[ep] + 0.1
    official.to_csv(source / "official.csv", index=False)
    test = pd.DataFrame({"Molecule_Name": [f"blind{i}" for i in range(750)], "SMILES": "N"})
    test.to_csv(source / "test.csv", index=False)
    sources = {str(p): sha256_file(p) for p in source.iterdir()}
    roots = []
    for fold in range(5):
        root = tmp_path / f"fold{fold}"
        root.mkdir()
        val = official.iloc[fold * 2 : fold * 2 + 2]
        train = official.drop(val.index)
        for name, frame in [("train", train), ("validation", val)]:
            frame.to_csv(root / f"{name}.csv", index=False)
        config = {
            "seed": 42 + fold,
            "fold": fold,
            "source_dir": str(source),
            "wandb": {"mode": "disabled"},
            "campaigns": {c: {"budget_gpu_hours": 0, "trial_seconds": 180} for c in CYPS},
        }
        manifest = {
            "config": config,
            "fixed_source_sha256": fixed_hashes(),
            "source_sha256": sources,
            "files": {n: sha256_file(root / n) for n in ["train.csv", "validation.csv"]},
            "validation_role": "development",
            "campaign_id": f"test{fold}",
        }
        write(root / "manifest.json", manifest)
        for cyp in CYPS:
            campaign = root / "campaigns" / cyp
            campaign.mkdir(parents=True)
            write(
                campaign / "campaign.json",
                {"campaign_id": f"test{fold}-{cyp}", **config["campaigns"][cyp]},
            )
            write(campaign / "registry.json", {"trials": [], "best": {}})
        roots.append(root)
    return roots


def artifact(root, cyp):
    output = root / "campaigns" / cyp / "trial"
    output.mkdir()
    (output / "recipe.py").write_text("# synthetic frozen recipe fixture\n")
    (output / "best.pt").write_bytes(b"synthetic checkpoint fixture")
    write(output / "strategy.json", {"description": "synthetic fixture", "assets_sha256": {}})
    ep = tasks_for(cyp)[0]
    pd.read_csv(root / "validation.csv")[["Molecule_Name", "parent", ep]].to_csv(
        output / "validation-predictions.csv", index=False
    )
    job = {
        "root": str(root),
        "output": str(output),
        "cyp": cyp,
        "smoke": False,
        "trial_id": cyp,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "recipe_sha256": sha256_file(output / "recipe.py"),
        "runtime_sha256": runtime_hashes(),
    }
    result = {
        "status": "complete",
        "smoke": False,
        "reload_max_abs": 0,
        "metrics": {"ST-RAE": 0.0},
        "best_epoch": 1,
        "checkpoint_sha256": sha256_file(output / "best.pt"),
        "strategy_sha256": sha256_file(output / "strategy.json"),
        "validation_predictions_sha256": sha256_file(output / "validation-predictions.csv"),
    }
    write(output / "job.json", job)
    write(output / "result.json", result)
    return output


def test_disabled_campaigns_are_independent_and_cannot_launch(data_roots):
    root = data_roots[0]
    for cyp in CYPS:
        with pytest.raises(ValueError, match="launch disabled"):
            run_trial(root, cyp, Path("missing.py"), "Must not execute")
    path = root / "campaigns" / "CYP1A2" / "registry.json"
    registry = read(path)
    registry["trials"].append({"smoke": False, "reserved_seconds": 100, "used_seconds": 20})
    registry["best"]["CYP1A2"] = {"metrics": {"ST-RAE": 0.4}}
    write(path, registry)
    board = status(root)
    assert board["campaigns"]["CYP1A2"]["committed_seconds"] == 20
    assert board["campaigns"]["CYP2D6"]["committed_seconds"] == 0
    assert board["best"]["CYP2D6"] is None
    assert board["MA-ST-RAE"] is None


def test_recipe_tasks_can_drop_auxiliaries_but_not_replace_primary():
    pytest.importorskip("torch")
    from somerra_openadmet.cyp_research_worker import recipe_tasks

    recipe = SimpleNamespace(select_tasks=lambda cyp, available: available[:2])
    assert recipe_tasks(recipe, "CYP2D6") == tasks_for("CYP2D6")[:2]
    recipe.select_tasks = lambda cyp, available: available[1:]
    with pytest.raises(ValueError, match="focal CYP"):
        recipe_tasks(recipe, "CYP2D6")


def test_campaign_telemetry_includes_macro_without_combining_selection(tmp_path, monkeypatch):
    from somerra_openadmet import research_tracking

    recorded = []

    class Tracker:
        def __init__(self, *args, **kwargs):
            pass

        def log(self, values):
            recorded.append(values)

        def finish(self, values):
            pass

    monkeypatch.setattr(research_tracking, "Tracker", Tracker)
    manifest = {"config": {"wandb": {}, "fold": 0}, "campaign_id": "one-cyp"}
    own = {"trials": [], "best": {CYPS[0]: {"metrics": {"ST-RAE": 0.2}}}}
    combined = {c: {"metrics": {"ST-RAE": i / 10}} for i, c in enumerate(CYPS, 2)}
    research_tracking.log_campaign(tmp_path, manifest, own, "disabled", all_best=combined)
    assert recorded[0]["campaign/MA-ST-RAE"] == pytest.approx(0.35)
    assert set(own["best"]) == {CYPS[0]}


def tensor_trial(root):
    output = artifact(root, "CYP1A2")
    (output / "recipe.py").write_text("""
import numpy as np
import torch
STRATEGY = {"name": "Tensor model with MSE", "loss": "masked squared error"}
SETTINGS = {"batch_size": 4, "max_epochs": 3, "patience": 2,
            "primary_weight": 1., "auxiliary_weight": 0.}
def select_tasks(cyp, available):
    return available[:1]
def build_dataset(frame, targets, settings):
    return torch.utils.data.TensorDataset(
        torch.tensor([[len(s) / 10] for s in frame.SMILES]),
        torch.tensor(targets, dtype=torch.float32))
def batch(ds, indices, device):
    return tuple(torch.stack([ds[int(i)][j] for i in indices]).to(device) for j in [0, 1])
def build_model(n_tasks, settings):
    return torch.nn.Linear(1, n_tasks)
def build_optimizer(model, settings):
    return torch.optim.SGD(model.parameters(), lr=.1)
def build_scheduler(optimizer, settings, steps_per_epoch):
    return None
def compute_loss(prediction, target, weights, context):
    assert len(context["primary_low"]) == len(target)
    mask = torch.isfinite(target)
    return ((prediction - torch.nan_to_num(target)).square() * mask * weights).sum() / mask.sum()
""")
    job = read(output / "job.json")
    job.update(
        smoke=True,
        reserved_seconds=180,
        wandb_mode="disabled",
        campaign_id="tensor-test",
        hypothesis="Exercise a non-graph model and a squared-error objective",
        recipe_sha256=sha256_file(output / "recipe.py"),
    )
    write(output / "job.json", job)
    return job


def test_custom_tensor_model_and_mse_loss_are_recorded_and_replayable(data_roots):
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from somerra_openadmet.cyp_research_worker import train

    root = data_roots[0]
    job = tensor_trial(root)
    output = Path(job["output"])
    train(job)
    result, strategy = read(output / "result.json"), read(output / "strategy.json")
    assert result["status"] == "complete"
    assert result["strategy_sha256"] == sha256_file(output / "strategy.json")
    assert strategy["description"]["loss"] == "masked squared error"
    assert strategy["tasks"] == [ACTIVITY_ENDPOINTS[0]]
    assert strategy["train_rows"] == 8
    saved = torch.load(output / "best.pt", weights_only=True)
    pred = cyp_submission.predict_trial(output, pd.read_csv(root / "validation.csv"))
    target = pd.read_csv(root / "validation.csv")[ACTIVITY_ENDPOINTS[0]].to_numpy()
    events = [
        __import__("json").loads(line)
        for line in (output / "events.jsonl").read_text().splitlines()
    ]
    selected = events[saved["epoch"] - 1]
    assert selected["val/loss"] == pytest.approx(
        np.mean(((pred - target) / saved["scale"][0]) ** 2), rel=1e-5
    )
    assert result["metrics"]["ST-RAE"] == min(event["val/ST-RAE"] for event in events)
    with pytest.raises(ValueError, match="smoke"):
        cyp_submission.checked_trial(output, "CYP1A2")


def test_build_plan_freezes_explicit_recipes_and_stays_disabled(data_roots, tmp_path):
    selections = {c: artifact(data_roots[0], c) for c in CYPS}
    root = tmp_path / "build"
    plan = cyp_submission.freeze(selections, root)
    assert plan["budget_gpu_hours"] == 0
    assert plan["folds"] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="launch disabled"):
        cyp_submission.fit(root)
    with pytest.raises(ValueError, match="20"):
        cyp_submission.build(root)
    # An editable research recipe no longer controls the frozen build.
    (selections["CYP1A2"] / "recipe.py").write_text("# changed after selection")
    cyp_submission.verify_plan(root)
    (root / "recipes" / "CYP1A2" / "recipe.py").write_text("# changed frozen recipe")
    with pytest.raises(ValueError, match="recipe record changed"):
        cyp_submission.verify_plan(root)


def test_complete_build_has_five_fold_mean_complete_oof_and_private_metadata(
    data_roots, tmp_path, monkeypatch
):
    fits = {f"{c}/fold{f}": artifact(r, c) for f, r in enumerate(data_roots) for c in CYPS}
    root = tmp_path / "build"
    cyp_submission.freeze({c: fits[f"{c}/fold0"] for c in CYPS}, root)
    state = read(root / "build-state.json")
    state["fits"] = {k: str(v) for k, v in fits.items()}
    write(root / "build-state.json", state)

    def predict(path, frame, device="auto"):
        job = read(Path(path) / "job.json")
        ep = tasks_for(job["cyp"])[0]
        if ep in frame:
            return frame[ep].to_numpy()
        fold = read(Path(job["root"]) / "manifest.json")["config"]["fold"]
        return np.full(len(frame), fold + CYPS.index(job["cyp"]))

    monkeypatch.setattr(cyp_submission, "predict_trial", predict)
    result = cyp_submission.build(root)
    prediction = pd.read_csv(root / "predictions.csv")
    assert len(prediction) == 750
    for i, ep in enumerate(ACTIVITY_ENDPOINTS):
        assert prediction[ep].eq(2 + i).all()
        assert len(pd.read_csv(root / f"{CYPS[i]}-oof.csv")) == 10
    assert result["MA-ST-RAE"] == 0
    assert result["open_source_code"] is False
    assert result["model_report_link"] == ""
    assert result["submitted"] is False
    assert len(result["fits"]) == 20
    (fits["CYP1A2/fold2"] / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed trial artifact"):
        cyp_submission.build(root)


def test_build_pins_assets_before_launch_and_rejects_replaced_fit_assets(
    data_roots, tmp_path, monkeypatch
):
    asset = tmp_path / "pretrained.pt"
    asset.write_bytes(b"original weights")
    originals = {str(asset): sha256_file(asset)}
    selections = {c: artifact(data_roots[0], c) for c in CYPS}

    def set_assets(path, assets):
        strategy = read(path / "strategy.json")
        strategy["assets_sha256"] = assets
        write(path / "strategy.json", strategy)
        result = read(path / "result.json")
        result["strategy_sha256"] = sha256_file(path / "strategy.json")
        write(path / "result.json", result)

    for path in selections.values():
        set_assets(path, originals)
    root = tmp_path / "build"
    plan = cyp_submission.freeze(selections, root, budget_gpu_hours=1)
    monkeypatch.setattr(cyp_submission, "prepare", lambda *args: pytest.fail("launched with drift"))
    asset.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="Frozen recipe asset changed"):
        cyp_submission.fit(root)
    asset.write_bytes(b"original weights")
    cyp_submission.verify_plan(root)
    # A fit can be internally consistent but use a different asset than its selection.
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"alternative")
    fitted = artifact(data_roots[1], "CYP1A2")
    set_assets(fitted, {str(replacement): sha256_file(replacement)})
    with pytest.raises(ValueError, match="Fit assets differ"):
        cyp_submission.checked_fit(root, plan, fitted, "CYP1A2", 1)


@pytest.mark.parametrize("expire_after", [1, 3])
def test_time_cap_finalizes_only_a_previously_validated_checkpoint(
    data_roots, monkeypatch, expire_after
):
    pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from somerra_openadmet import cyp_research_worker as worker

    job = tensor_trial(data_roots[0])
    output = Path(job["output"])
    clock, calls = [0.0], [0]
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original = worker.objective

    def objective(*args):
        if args[-1]["phase"] == "train":
            calls[0] += 1
            if calls[0] == expire_after:
                clock[0] = 200.0
        return original(*args)

    monkeypatch.setattr(worker, "objective", objective)
    if expire_after == 1:
        with pytest.raises(TimeoutError, match="before any validated checkpoint"):
            worker.train(job)
    else:
        worker.train(job)
        result = read(output / "result.json")
        assert result["status"] == "complete"
        assert result["stop_reason"] == "time_cap"
        assert result["epochs"] == result["best_epoch"] == 1
        assert result["optimizer_steps"] == 3
        assert result["partial_epoch_steps"] == 1
        assert result["reload_max_abs"] < 1e-4


def test_training_metrics_allow_missing_intervals_and_do_not_run_train_inference(
    data_roots, monkeypatch
):
    pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from somerra_openadmet import cyp_research_worker as worker

    root = data_roots[0]
    train_frame = pd.read_csv(root / "train.csv")
    for suffix in ["_conf_low", "_conf_high"]:
        train_frame[ACTIVITY_ENDPOINTS[0] + suffix] = np.nan
    train_frame.to_csv(root / "train.csv", index=False)
    manifest = read(root / "manifest.json")
    manifest["files"]["train.csv"] = sha256_file(root / "train.csv")
    write(root / "manifest.json", manifest)
    job = tensor_trial(root)
    sizes = []
    original = worker.RecipeData.predict

    def predict(self, model, ds, device):
        sizes.append(len(ds))
        return original(self, model, ds, device)

    monkeypatch.setattr(worker.RecipeData, "predict", predict)
    worker.train(job)
    output = Path(job["output"])
    event = read(output / "progress.json")
    result = read(output / "result.json")
    assert event["train/ST-RAE"] is None
    assert event["train/missing_intervals"] == event["train/n"] == 8
    assert event["train/MAE"] is not None
    assert event["val/ST-RAE"] is not None
    assert sizes == [2] * (result["epochs"] + 1)  # Validation each epoch, then selected replay.


def test_cached_loss_context_keeps_shuffled_bounds_aligned(data_roots):
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from somerra_openadmet.cyp_research_worker import LossContext

    frame = pd.read_csv(data_roots[0] / "train.csv")
    ep = ACTIVITY_ENDPOINTS[0]
    context = LossContext(frame, [ep], [4.0], [2.0], torch.device("cpu"), {}, "train")
    one = context.batch(1, np.array([4, 0, 2]))
    two = context.batch(2, np.array([7, 1]))
    assert one["mean"].data_ptr() == two["mean"].data_ptr()
    assert one["primary_low"].tolist() == pytest.approx(frame.iloc[[4, 0, 2]][ep + "_conf_low"])
    assert two["primary_high"].tolist() == pytest.approx(frame.iloc[[7, 1]][ep + "_conf_high"])


def test_worker_rejects_asset_drift_from_selected_recipe(data_roots):
    pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from somerra_openadmet.cyp_research_worker import train

    job = tensor_trial(data_roots[0])
    job["expected_assets_sha256"] = {"required-checkpoint": "missing"}
    with pytest.raises(ValueError, match="assets differ from frozen"):
        train(job)


def test_inference_defaults_to_recorded_device_without_silent_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cyp_submission.inference_device({"inference_device": "cpu"}).type == "cpu"
    with pytest.raises(RuntimeError, match="Recorded inference requires CUDA"):
        cyp_submission.inference_device({"inference_device": "cuda"})
    assert cyp_submission.inference_device({"inference_device": "cuda"}, "cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert cyp_submission.inference_device({"inference_device": "cuda"}).type == "cuda"


def test_retry_allocation_keeps_plan_and_prior_charges(data_roots, tmp_path, monkeypatch):
    from somerra_openadmet import cyp_research

    selections = {c: artifact(data_roots[0], c) for c in CYPS}
    root = tmp_path / "build"
    cyp_submission.freeze(selections, root)
    plan_hash = sha256_file(root / "plan.json")
    record = cyp_submission.allocate(root, "CYP1A2", 0, 1 / 60, "Authorized retry after failure")
    assert sha256_file(root / "plan.json") == plan_hash
    assert record["additional_gpu_hours"] == 1 / 60
    registry_path = data_roots[0] / "campaigns" / "CYP1A2" / "registry.json"
    write(
        registry_path,
        {
            "best": {},
            "trials": [
                {"smoke": False, "reserved_seconds": 30, "used_seconds": 20, "status": "failed"},
                {"smoke": False, "reserved_seconds": 20, "status": "interrupted"},
            ],
        },
    )
    before = read(registry_path)
    with pytest.raises(ValueError, match="budget exhausted"):
        run_trial(data_roots[0], "CYP1A2", selections["CYP1A2"] / "recipe.py", "retry")

    class Worker:
        def __init__(self, args, **kwargs):
            job = read(Path(args[-1]))
            assert job["reserved_seconds"] == pytest.approx(20)  # 60 new - 40 prior.
            write(Path(job["output"]) / "result.json", {"status": "failed", "error": "fixture"})

        def wait(self, **kwargs):
            return 1

    monkeypatch.setattr(cyp_research.subprocess, "Popen", Worker)
    trial = run_trial(
        data_roots[0],
        "CYP1A2",
        selections["CYP1A2"] / "recipe.py",
        "retry",
        additional_budget_hours=record["additional_gpu_hours"],
    )
    assert trial["status"] == "failed"
    assert read(registry_path)["trials"][:2] == before["trials"]
    assert read(root / "plan.json")["budget_gpu_hours"] == 0


def add_internal_test(root):
    manifest = read(root / "manifest.json")
    manifest["config"]["test_fold"] = 1
    frame = pd.read_csv(root / "train.csv")
    frame.iloc[:2].to_csv(root / "internal-test.csv", index=False)
    frame.iloc[2:].to_csv(root / "train.csv", index=False)
    for name in ["train.csv", "internal-test.csv"]:
        manifest["files"][name] = sha256_file(root / name)
    write(root / "manifest.json", manifest)


def test_training_never_reads_internal_test_or_logs_test_metrics(data_roots, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from somerra_openadmet import cyp_research_worker as worker

    root = data_roots[0]
    add_internal_test(root)
    job = tensor_trial(root)
    original = pd.read_csv

    def read_csv(path, *args, **kwargs):
        assert Path(path).name != "internal-test.csv", "Training accessed test labels"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(worker.pd, "read_csv", read_csv)
    worker.train(job)
    output = Path(job["output"])
    assert "test/" not in (output / "events.jsonl").read_text()
    result = read(output / "result.json")
    assert "test_metrics" not in result
    assert read(output / "strategy.json")["train_rows"] == 6


def test_internal_test_scores_any_completed_trial_and_ranks_strategies(data_roots, monkeypatch):
    import shutil

    from somerra_openadmet import cyp_test
    from somerra_openadmet.cyp_research import accept_result

    root = data_roots[0]
    add_internal_test(root)
    incumbent = artifact(root, "CYP1A2")
    trial = incumbent.parent / "trial-two"
    shutil.copytree(incumbent, trial)
    job = read(trial / "job.json")
    job.update(trial_id="trial-two", output=str(trial))
    write(trial / "job.json", job)
    val = pd.read_csv(root / "validation.csv")
    ep = tasks_for("CYP1A2")[0]
    validation_prediction = val[ep].to_numpy() + 0.5
    saved_validation = pd.read_csv(trial / "validation-predictions.csv")
    saved_validation[ep] = validation_prediction
    saved_validation.to_csv(trial / "validation-predictions.csv", index=False)
    result = read(trial / "result.json")
    from somerra_openadmet.cyp_research import regression_metrics

    result["metrics"]["ST-RAE"] = regression_metrics(val, validation_prediction, ep)["ST-RAE"]
    result["validation_predictions_sha256"] = sha256_file(trial / "validation-predictions.csv")
    write(trial / "result.json", result)
    campaign = root / "campaigns" / "CYP1A2"
    best_record = {
        **read(incumbent / "job.json"),
        "status": "complete",
        "reserved_seconds": 180,
        "used_seconds": 1,
    }
    trial_record = {
        **job,
        "status": "complete",
        "reserved_seconds": 180,
        "used_seconds": 1,
    }
    registry = {"trials": [best_record, trial_record], "best": {}}
    accept_result(registry, best_record, read(incumbent / "result.json"))
    write(campaign / "registry.json", registry)
    calls = []

    def predict(path, frame, device="auto"):
        assert set(frame.columns) == {"Molecule_Name", "SMILES", "parent"}
        names = list(frame.Molecule_Name)
        calls.append((Path(path).name, names))
        prediction = np.array([float(name[1:]) for name in names])
        if Path(path).name == "trial" and names[0] == "m2":
            prediction = prediction + 1
        if Path(path).name == "trial-two" and names[0] == "m0":
            prediction = prediction + 0.5
        return prediction

    monkeypatch.setattr(cyp_test, "predict_trial", predict)
    before = (campaign / "registry.json").read_bytes()
    incumbent_report = cyp_test.evaluate_test(incumbent, device="cpu", mode="disabled")
    report = cyp_test.evaluate_test(trial, device="cpu", mode="disabled")
    assert report["metrics"]["ST-RAE"] == pytest.approx(0)
    assert report["selection_eligible"] is True
    assert (
        report["role"] == "adaptive internal-test development; not an independent or official score"
    )
    assert incumbent_report["metrics"]["ST-RAE"] > report["metrics"]["ST-RAE"]
    assert (campaign / "registry.json").read_bytes() == before
    assert calls == [
        ("trial", ["m0", "m1"]),
        ("trial", ["m2", "m3"]),
        ("trial-two", ["m0", "m1"]),
        ("trial-two", ["m2", "m3"]),
    ]
    assert cyp_test.evaluate_test(trial, device="cpu", mode="disabled") == report
    assert len(calls) == 4
    assert (trial / "internal-test" / "selection.json").exists()
    assert (trial / "internal-test" / "report.json").exists()
    assert not (campaign / "test-selection.json").exists()
    strategy_registry = read(campaign / "strategy-registry.json")
    assert strategy_registry["selection_metric"] == "internal-test/ST-RAE"
    assert strategy_registry["best"]["CYP1A2"]["trial_id"] == "trial-two"
    assert len(strategy_registry["trials"]) == 2
    (trial / "internal-test" / "predictions.csv").write_text("corrupted\n")
    with pytest.raises(ValueError, match="changed|corrupt|hash|artifact"):
        cyp_test.evaluate_test(trial, device="cpu", mode="disabled")


def test_internal_test_preserves_historical_closed_campaign(data_roots):
    from somerra_openadmet.cyp_test import evaluate_test

    root = data_roots[0]
    add_internal_test(root)
    trial = artifact(root, "CYP1A2")
    campaign = root / "campaigns" / "CYP1A2"
    write(campaign / "test-selection.json", {"historical": True})
    with pytest.raises(ValueError, match="Historical campaign is closed"):
        evaluate_test(trial, device="cpu", mode="disabled")


def test_internal_test_rejects_unregistered_and_smoke_trials(data_roots):
    from somerra_openadmet.cyp_test import evaluate_test

    root = data_roots[0]
    add_internal_test(root)
    trial = artifact(root, "CYP1A2")
    with pytest.raises(ValueError, match="registered|registry"):
        evaluate_test(trial, device="cpu", mode="disabled")
    job = read(trial / "job.json")
    job["smoke"] = True
    write(trial / "job.json", job)
    with pytest.raises(ValueError, match="smoke"):
        evaluate_test(trial, device="cpu", mode="disabled")


def test_submission_fit_clears_research_test_fold(data_roots, tmp_path, monkeypatch):
    root = data_roots[0]
    add_internal_test(root)
    selections = {c: artifact(root, c) for c in CYPS}
    build_root = tmp_path / "build"
    cyp_submission.freeze(selections, build_root, budget_gpu_hours=1)

    class Prepared(Exception):
        pass

    def prepare(config):
        assert config["test_fold"] is None
        assert config["workflow"] == "submission-fit"
        raise Prepared

    monkeypatch.setattr(cyp_submission, "prepare", prepare)
    with pytest.raises(Prepared):
        cyp_submission.fit(build_root)
