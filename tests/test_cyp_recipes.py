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


def test_custom_tensor_model_and_mse_loss_are_recorded_and_replayable(data_roots):
    torch = pytest.importorskip("torch")
    from somerra_openadmet.cyp_research_worker import train

    root = data_roots[0]
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

    def predict(path, frame):
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
