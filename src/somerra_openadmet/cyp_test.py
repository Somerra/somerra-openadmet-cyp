"""Report internal-test performance only after freezing validation-based selection."""

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
        incumbent = registry["best"].get(cyp, {})
        if incumbent.get("trial_id") != job["trial_id"] or not any(
            t["trial_id"] == job["trial_id"]
            and Path(t["output"]).resolve() == trial
            and t["status"] == "complete"
            for t in registry["trials"]
        ):
            raise ValueError("Internal test requires the recorded validation incumbent")
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
        selection_path = campaign / "test-selection.json"
        if selection_path.exists():
            if read(selection_path) != selection:
                raise ValueError("Internal-test selection is frozen; cannot replace the model")
        else:
            # Freeze before reading test labels or producing any test feedback.
            # run_trial shares this lock and refuses new trials once this exists.
            write(selection_path, selection)
        output = campaign / "internal-test"
        output.mkdir(exist_ok=True)
        for name in selection["runtime_sha256"]:
            shutil.copyfile(Path(__file__).parent / name, output / name)
        report_path = output / "report.json"
        if report_path.exists():
            report = read(report_path)
            if report["selection_sha256"] != sha256_file(selection_path) or report[
                "predictions_sha256"
            ] != sha256_file(output / "predictions.csv"):
                raise ValueError("Internal-test report artifacts changed")
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
            "role": "internal test after validation-only model selection; not an official score",
            "selection_eligible": False,
            "predictions_sha256": sha256_file(output / "predictions.csv"),
            "elapsed_seconds": time.monotonic() - started,
            "inference_device": result.get("inference_device", "cuda")
            if device == "auto"
            else device,
        }
        write(report_path, report)
        tracker = Tracker(
            output,
            manifest["config"]["wandb"],
            run_id=job["trial_id"] + "-internal-test",
            group=read(campaign / "campaign.json")["campaign_id"],
            cyp=cyp,
            config={**selection, "workflow": "internal-test"},
            mode=mode or manifest["config"]["wandb"]["mode"],
        )
        values = {f"test/{key}": value for key, value in metrics.items()}
        tracker.log(values)
        tracker.finish({**values, "selection_eligible": False})
        return report
