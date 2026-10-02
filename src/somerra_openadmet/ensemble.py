"""Cross-fitted, metric-aligned ensembling for activity regression runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constants import ACTIVITY_ENDPOINTS
from .data import load_test
from .download import sha256_file
from .metrics import soft_threshold_rae
from .validation import validate_submission

OOF_KEYS = ("Molecule_Name", "endpoint", "fold", "y_true", "conf_low", "conf_high")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _metric_weights(
    predictions: np.ndarray,
    labels: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    seed: int,
    search_samples: int,
) -> tuple[np.ndarray, float]:
    """Find non-negative simplex weights minimizing ST-RAE."""
    matrix = np.asarray(predictions, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("Ensembling requires predictions from at least two runs")
    if search_samples < 1:
        raise ValueError("search_samples must be positive")

    n_models = matrix.shape[1]
    rng = np.random.default_rng(seed)
    candidates = np.vstack(
        [
            np.full((1, n_models), 1.0 / n_models),
            np.eye(n_models),
            rng.dirichlet(np.ones(n_models), size=search_samples),
        ]
    )
    scores = np.asarray(
        [soft_threshold_rae(labels, matrix @ weights, lower, upper) for weights in candidates]
    )
    start = candidates[int(np.argmin(scores))]

    def objective(weights: np.ndarray) -> float:
        return soft_threshold_rae(labels, matrix @ weights, lower, upper)

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"ftol": 1e-10, "maxiter": 1_000},
    )
    weights = np.asarray(result.x if result.success else start, dtype=float)
    weights = np.clip(weights, 0.0, 1.0)
    weights /= weights.sum()
    score = objective(weights)
    if score > float(np.min(scores)):
        weights = start
        score = float(np.min(scores))
    return weights, float(score)


def _load_run(run_dir: Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    required = {
        "manifest": run_dir / "run_manifest.json",
        "metrics": run_dir / "metrics.json",
        "oof": run_dir / "oof_predictions.csv",
        "submission": run_dir / "activity_submission.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete baseline run {run_dir}: missing {missing}")
    manifest = json.loads(required["manifest"].read_text())
    config = manifest.get("config", {})
    if config.get("track") != "activity":
        raise ValueError(f"Run is not an activity baseline: {run_dir}")
    return {
        "name": run_dir.name,
        "path": run_dir,
        "manifest": manifest,
        "oof": pd.read_csv(required["oof"]),
        "submission": pd.read_csv(required["submission"]),
        "hashes": {label: sha256_file(path) for label, path in required.items()},
    }


def _aligned_oof(runs: list[dict[str, object]], endpoint: str) -> tuple[pd.DataFrame, np.ndarray]:
    reference = runs[0]["oof"]
    assert isinstance(reference, pd.DataFrame)
    base = reference.loc[reference["endpoint"] == endpoint, list(OOF_KEYS)].copy()
    base = base.sort_values("Molecule_Name").reset_index(drop=True)
    columns: list[np.ndarray] = []
    for run in runs:
        frame = run["oof"]
        assert isinstance(frame, pd.DataFrame)
        selected = frame.loc[frame["endpoint"] == endpoint, [*OOF_KEYS, "y_pred"]]
        selected = selected.sort_values("Molecule_Name").reset_index(drop=True)
        if not selected[list(OOF_KEYS)].equals(base):
            raise ValueError(f"OOF rows or fold assignments differ for run {run['name']}")
        columns.append(selected["y_pred"].to_numpy(dtype=float))
    return base, np.column_stack(columns)


def _aligned_test_predictions(
    runs: list[dict[str, object]], endpoint: str, test: pd.DataFrame
) -> np.ndarray:
    columns: list[np.ndarray] = []
    expected = test[["Molecule_Name", "SMILES"]].reset_index(drop=True)
    for run in runs:
        frame = run["submission"]
        assert isinstance(frame, pd.DataFrame)
        aligned = expected.merge(
            frame[["Molecule_Name", "SMILES", endpoint]],
            on=["Molecule_Name", "SMILES"],
            how="left",
            validate="one_to_one",
        )
        if aligned[endpoint].isna().any():
            raise ValueError(f"Submission IDs differ for run {run['name']}")
        columns.append(aligned[endpoint].to_numpy(dtype=float))
    return np.column_stack(columns)


def run_activity_ensemble(
    run_dirs: list[Path],
    data_dir: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    search_samples: int = 4096,
) -> dict[str, object]:
    """Cross-fit blend weights and emit a validated activity submission."""
    if len(run_dirs) < 2:
        raise ValueError("At least two --run-dir values are required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    runs = [_load_run(path) for path in run_dirs]

    input_hashes = [run["manifest"].get("input_sha256") for run in runs]
    if any(value != input_hashes[0] for value in input_hashes[1:]):
        raise ValueError("Baseline runs were built from different input files")

    test = load_test(data_dir)
    submission = test[["Molecule_Name", "SMILES"]].copy()
    oof_rows: list[pd.DataFrame] = []
    endpoint_metrics: dict[str, object] = {}
    final_weights: dict[str, dict[str, float]] = {}
    cross_fit_weights: dict[str, dict[str, dict[str, float]]] = {}
    run_names = [str(run["name"]) for run in runs]

    for endpoint_index, endpoint in enumerate(ACTIVITY_ENDPOINTS):
        frame, matrix = _aligned_oof(runs, endpoint)
        labels = frame["y_true"].to_numpy(dtype=float)
        lower = frame["conf_low"].to_numpy(dtype=float)
        upper = frame["conf_high"].to_numpy(dtype=float)
        folds = frame["fold"].to_numpy(dtype=int)
        cross_fitted = np.full(len(frame), np.nan)
        endpoint_fold_weights: dict[str, dict[str, float]] = {}

        for fold in sorted(np.unique(folds)):
            fit = folds != fold
            holdout = folds == fold
            weights, _ = _metric_weights(
                matrix[fit],
                labels[fit],
                lower[fit],
                upper[fit],
                seed=seed + endpoint_index * 100 + int(fold),
                search_samples=search_samples,
            )
            cross_fitted[holdout] = matrix[holdout] @ weights
            endpoint_fold_weights[str(int(fold))] = dict(
                zip(run_names, weights.tolist(), strict=True)
            )

        weights, fitted_score = _metric_weights(
            matrix,
            labels,
            lower,
            upper,
            seed=seed + endpoint_index * 100 + 99,
            search_samples=search_samples,
        )
        cross_fitted_score = soft_threshold_rae(labels, cross_fitted, lower, upper)
        base_scores = {
            name: soft_threshold_rae(labels, matrix[:, index], lower, upper)
            for index, name in enumerate(run_names)
        }
        endpoint_metrics[endpoint] = {
            "labels": len(frame),
            "base_oof_st_rae": base_scores,
            "cross_fitted_ensemble_st_rae": cross_fitted_score,
            "all_oof_fitted_st_rae_model_selection_only": fitted_score,
        }
        final_weights[endpoint] = dict(zip(run_names, weights.tolist(), strict=True))
        cross_fit_weights[endpoint] = endpoint_fold_weights
        oof_rows.append(frame.assign(y_pred=cross_fitted))

        test_matrix = _aligned_test_predictions(runs, endpoint, test)
        submission[endpoint] = test_matrix @ weights

    metrics = {
        "endpoints": endpoint_metrics,
        "MA-ST-RAE_cross_fitted": float(
            np.mean([value["cross_fitted_ensemble_st_rae"] for value in endpoint_metrics.values()])
        ),
        "MA-ST-RAE_all_oof_fitted_model_selection_only": float(
            np.mean(
                [
                    value["all_oof_fitted_st_rae_model_selection_only"]
                    for value in endpoint_metrics.values()
                ]
            )
        ),
    }
    pd.concat(oof_rows, ignore_index=True).to_csv(output_dir / "oof_predictions.csv", index=False)
    submission_path = output_dir / "activity_submission.csv"
    submission.to_csv(submission_path, index=False)
    validation = validate_submission(submission_path, test, "activity")
    if not validation.valid:
        raise RuntimeError(f"Generated invalid ensemble submission: {validation.errors}")

    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "weights.json",
        {"final_all_oof_weights": final_weights, "cross_fit_weights": cross_fit_weights},
    )
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "cross_fitted_metric_aligned_simplex_ensemble",
        "seed": seed,
        "search_samples": search_samples,
        "base_runs": [
            {"name": run["name"], "path": str(run["path"]), "hashes": run["hashes"]} for run in runs
        ],
        "input_sha256": input_hashes[0],
        "submission": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "validation": validation.to_dict(),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return {"metrics": metrics, "manifest": manifest, "submission": str(submission_path)}
