"""Score fixed validation checkpoints for adaptive internal-test strategy selection."""

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cyp_research import lock, read, regression_metrics, tasks_for, verify, write
from .cyp_submission import checked_trial, predict_trial
from .download import sha256_file
from .research_tracking import Tracker


def evaluate_test(trial, device="auto", mode=None):
    trial = Path(trial).resolve()
    job = read(trial / "job.json")
    root, cyp = Path(job["root"]), job["cyp"]
    campaign = root / "campaigns" / cyp
    with lock(campaign):
        job, result = checked_trial(trial, cyp)
        manifest = verify(root)
        if sha256_file(root / "manifest.json") != job["manifest_sha256"]:
            raise ValueError("Research manifest changed")
        if manifest["config"].get("test_fold") is None:
            raise ValueError("No internal test fold in this campaign; prepare a fresh revision")
        registry = read(campaign / "registry.json")
        if (campaign / "test-selection.json").exists():
            raise ValueError("Historical campaign is closed; preserve its original test selection")
        if not any(
            t["trial_id"] == job["trial_id"]
            and Path(t["output"]).resolve() == trial
            and t["status"] == "complete"
            for t in registry["trials"]
        ):
            raise ValueError("Internal test requires a registered completed trial")
        selection = {
            "trial": str(trial),
            "cyp": cyp,
            "manifest_sha256": job["manifest_sha256"],
            "result_sha256": sha256_file(trial / "result.json"),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "selection_metric": "validation/ST-RAE",
            "selection_score": result["metrics"]["ST-RAE"],
            "test_fold": manifest["config"]["test_fold"],
            "test_sha256": manifest["files"]["internal-test.csv"],
            "runtime_sha256": {
                name: sha256_file(Path(__file__).parent / name)
                for name in ["cyp_test.py", "cyp_submission.py"]
            },
        }
        output = trial / "internal-test"
        output.mkdir(exist_ok=True)
        selection_path = output / "selection.json"
        if selection_path.exists():
            if read(selection_path) != selection:
                raise ValueError("Internal-test selection is frozen; cannot replace the model")
        else:
            # Freeze before reading test labels or producing any test feedback.
            # Freeze this checkpoint, not the campaign; future strategies remain eligible.
            write(selection_path, selection)
        for name in selection["runtime_sha256"]:
            shutil.copyfile(Path(__file__).parent / name, output / name)
        report_path = output / "report.json"
        if report_path.exists():
            report = read(report_path)
            if report["selection_sha256"] != sha256_file(selection_path) or report[
                "predictions_sha256"
            ] != sha256_file(output / "predictions.csv"):
                raise ValueError("Internal-test report artifacts changed")
            record_strategy(campaign, trial, report)
            return report
        started = time.monotonic()
        ep = tasks_for(cyp)[0]
        val = pd.read_csv(root / "validation.csv")
        # Never expose held-out labels/bounds to recipe input adapters.
        columns = ["Molecule_Name", "SMILES", "parent"]
        replay = predict_trial(trial, val[columns], device=device)
        saved = pd.read_csv(trial / "validation-predictions.csv")
        if list(saved.Molecule_Name) != list(val.Molecule_Name) or not np.allclose(
            replay, saved[ep], atol=1e-4, rtol=0
        ):
            raise ValueError("Validation checkpoint replay differs before internal test")
        if not np.isclose(
            regression_metrics(val, replay, ep)["ST-RAE"],
            result["metrics"]["ST-RAE"],
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("Validation replay score differs before internal test")
        test = pd.read_csv(root / "internal-test.csv")
        prediction = predict_trial(trial, test[columns], device=device)
        metrics = regression_metrics(test, prediction, ep)
        checked_trial(trial, cyp)
        verify(root)
        exported = test[["Molecule_Name", "parent"]].copy()
        exported[ep] = prediction
        exported.to_csv(output / "predictions.csv", index=False)
        report = {
            "cyp": cyp,
            "trial_id": job["trial_id"],
            "selection_sha256": sha256_file(selection_path),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "metrics": metrics,
            "role": "adaptive internal-test development; not an independent or official score",
            "selection_eligible": True,
            "predictions_sha256": sha256_file(output / "predictions.csv"),
            "elapsed_seconds": time.monotonic() - started,
            "inference_device": result.get("inference_device", "cuda")
            if device == "auto"
            else device,
        }
        write(report_path, report)
        record_strategy(campaign, trial, report)
        tracker = Tracker(
            output,
            manifest["config"]["wandb"],
            run_id=job["trial_id"] + "-internal-test",
            group=read(campaign / "campaign.json")["campaign_id"],
            cyp=cyp,
            config={**selection, "workflow": "adaptive-strategy-score"},
            mode=mode or manifest["config"]["wandb"]["mode"],
        )
        values = {f"test/{key}": value for key, value in metrics.items()}
        tracker.log(values)
        tracker.finish({**values, "selection_eligible": True})
        return report


def record_strategy(campaign, trial, report):
    """Update the separate strategy ledger while holding the campaign lock.

    The original registry remains an unchanged training/validation ledger for
    replay compatibility. It is not the strategy-selection authority.
    """
    score = report["metrics"]["ST-RAE"]
    if not np.isfinite(score):
        raise ValueError("Nonfinite internal-test score")
    path = campaign / "strategy-registry.json"
    registry = (
        read(path)
        if path.exists()
        else {"selection_metric": "internal-test/ST-RAE", "trials": [], "best": {}}
    )
    digest = sha256_file(trial / "internal-test" / "report.json")
    for existing in registry["trials"]:
        if existing["trial_id"] == report["trial_id"]:
            if existing["report_sha256"] != digest:
                raise ValueError("Recorded strategy report changed")
            return
    cyp = report["cyp"]
    old = registry["best"].get(cyp)
    promoted = old is None or score < old["metrics"]["ST-RAE"] - 1e-12
    registry["trials"].append(
        {
            "trial_id": report["trial_id"],
            "output": str(trial),
            "metrics": report["metrics"],
            "report_sha256": digest,
            "promoted": promoted,
        }
    )
    if promoted:
        registry["best"][cyp] = {
            "trial_id": report["trial_id"],
            "metrics": report["metrics"],
            "checkpoint": str(trial / "best.pt"),
            "strategy_record": str(trial / "strategy.json"),
            "report": str(trial / "internal-test" / "report.json"),
        }
    write(path, registry)
