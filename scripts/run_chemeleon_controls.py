"""Predeclared CPU sanity checks and regularized outer-only CheMeleon follow-up."""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from somerra_openadmet.campaign_budget import CampaignLedger
from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.download import sha256_file
from somerra_openadmet.metrics import soft_threshold_rae
from somerra_openadmet.regression_report import compare
from somerra_openadmet.representation_budget import ledger_path
from somerra_openadmet.representation_training import oof_rows, screen_representations
from somerra_openadmet.representations import atomic_json, load_cache, prepared_data, provenance


def cpu_control(name, frame, features, output):
    output.mkdir()
    start = time.monotonic()
    result = np.full((len(frame), len(ACTIVITY_ENDPOINTS)), np.nan)
    models = []
    for fold in range(5):
        held = np.flatnonzero(frame.fold.to_numpy() == fold)
        for column, endpoint in enumerate(ACTIVITY_ENDPOINTS):
            fit = np.flatnonzero((frame.fold != fold) & frame[endpoint].notna())
            target = frame.iloc[fit][endpoint].to_numpy()
            if name == "constant":
                result[held, column] = target.mean()
                continue
            model = (
                make_pipeline(StandardScaler(), Ridge(alpha=10, solver="lsqr", tol=1e-4))
                if name == "ridge"
                else LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.03,
                    num_leaves=15,
                    min_child_samples=40,
                    reg_lambda=5,
                    colsample_bytree=0.8,
                    subsample=0.8,
                    subsample_freq=1,
                    random_state=42 + fold,
                    n_jobs=8,
                    verbosity=-1,
                )
            )
            model.fit(features[fit], target)
            result[held, column] = model.predict(features[held])
            path = output / f"fold-{fold}-{endpoint}.joblib"
            joblib.dump(model, path)
            replay = joblib.load(path).predict(features[held])
            if not np.allclose(replay, result[held, column], atol=1e-8, rtol=0):
                raise ValueError("CPU control checkpoint replay mismatch")
            models.append({"path": path.name, "sha256": sha256_file(path), "fit_rows": len(fit)})
    if not np.isfinite(result).all():
        raise ValueError("Incomplete CPU OOF predictions")
    oof = oof_rows(frame, np.arange(len(frame)), result)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    endpoints = {}
    for endpoint in ACTIVITY_ENDPOINTS:
        part = oof.loc[oof.endpoint == endpoint]
        endpoints[endpoint] = {
            "ST-RAE": soft_threshold_rae(part.y_true, part.y_pred, part.conf_low, part.conf_high),
            "MAE": float(np.mean(np.abs(part.y_true - part.y_pred))),
            "observed_labels": len(part),
        }
    metrics = {
        "endpoints": endpoints,
        "MA-ST-RAE": float(np.mean([x["ST-RAE"] for x in endpoints.values()])),
    }
    atomic_json(output / "metrics.json", metrics)
    atomic_json(
        output / "run_manifest.json",
        {
            "complete": True,
            "model": name,
            "features": "CheMeleon only; no fingerprints",
            "standardization": "Ridge features fit only on endpoint training rows",
            "models": models,
            "wall_seconds": time.monotonic() - start,
            "gpu_hours": 0,
            "oof_sha256": sha256_file(output / "oof_predictions.csv"),
            **provenance(),
        },
    )


def main(output, cache, prepared, checkpoint):
    ledger = CampaignLedger(ledger_path({}))
    # Refuse a partial matrix before launching any GPU work; never increase this allocation.
    if ledger.snapshot()["committed_hours"] + 0.12 > 12:
        raise ValueError("The fixed follow-up requires 0.12 hours of remaining campaign capacity")
    output.mkdir(parents=True, exist_ok=False)
    frame, manifest = prepared_data(prepared)
    features, cache_manifest = load_cache(cache, frame)
    common = {
        "prepared_dir": str(prepared),
        "cache_dir": str(cache),
        "checkpoint": str(checkpoint),
        "device": "cuda",
        "threads": 8,
        "seed": 42,
        "h100_select": False,
        "selected_folds": [0, 1, 2, 3, 4],
        "batch_size": 256,
    }
    recipes = {
        "frozen-reference": {
            "adaptation": "frozen",
            "epochs": 300,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "precision": "float32",
            "max_gpu_hours": 0.03,
        },
        "frozen-regularized": {
            "adaptation": "frozen",
            "epochs": 100,
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "precision": "float32",
            "max_gpu_hours": 0.03,
            "head_dropout": 0.2,
            "standardize_targets": True,
        },
        "full-regularized": {
            "adaptation": "full",
            "epochs": 60,
            "learning_rate": 5e-5,
            "weight_decay": 1e-3,
            "precision": "bf16",
            "max_gpu_hours": 0.06,
            "head_dropout": 0.2,
            "standardize_targets": True,
        },
    }
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "inner_selection": False,
        "outer_early_stopping": False,
        "folds": [0, 1, 2, 3, 4],
        "training_sha256": manifest["training_sha256"],
        "cache_manifest": cache_manifest,
        "cpu_controls": [
            "constant training mean",
            "StandardScaler + Ridge alpha=10",
            "LightGBM 300 trees, LR .03, 15 leaves, min child 40, L2 5, row/col .8",
        ],
        "neural_recipes": recipes,
        "max_total_gpu_hours": 0.12,
        "campaign_ledger": str(ledger.path),
        "budget_before": ledger.snapshot(),
        "interpretation": "Predeclared follow-up to a failed pilot; exploratory, no promotion",
        **provenance(),
    }
    atomic_json(output / "protocol.json", protocol)
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(
        source / "src/somerra_openadmet",
        output / "source",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(__file__, output / Path(__file__).name)
    for name, recipe in recipes.items():
        atomic_json(output / f"{name}.json", {**common, **recipe, "output_dir": str(output / name)})
    status = {"complete": False, "completed": [], "failures": {}}

    def update(stage):
        status["stage"] = stage
        atomic_json(output / "status.json", status)
        print(json.dumps(status), flush=True)

    with threadpool_limits(limits=8):
        for name in ("constant", "ridge", "lightgbm"):
            update(name)
            cpu_control(name, frame, features, output / name)
            status["completed"].append(name)
    for name in recipes:
        update(name)
        try:
            screen_representations(output / f"{name}.json")
            status["completed"].append(name)
        except Exception as exc:
            status["failures"][name] = f"{type(exc).__name__}: {exc}"
    update("report")
    summary = {"models": {}, "comparisons": {}, "budget_after": ledger.snapshot()}
    for name in status["completed"]:
        summary["models"][name] = json.loads((output / name / "metrics.json").read_text())
        summary["comparisons"][name] = compare(
            output / name,
            "nested_ensemble",
            Path("runs/regression-nested-incumbent-v1"),
            output / f"{name}-versus-incumbent.json",
        )
    for name in ("frozen-regularized", "full-regularized"):
        if name in status["completed"] and "frozen-reference" in status["completed"]:
            compare(
                output / name,
                "y_pred",
                output / "frozen-reference",
                output / f"{name}-versus-frozen-reference.json",
            )
    summary["charged_gpu_hours"] = (
        ledger.snapshot()["used_hours"] - protocol["budget_before"]["used_hours"]
    )
    atomic_json(output / "summary.json", summary)
    status["complete"] = not status["failures"]
    update("complete" if status["complete"] else "incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, default=Path("runs/regression-corpus-v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/chemeleon_mp.pt"))
    args = parser.parse_args()
    main(args.output_dir, args.cache_dir, args.prepared_dir, args.checkpoint)
