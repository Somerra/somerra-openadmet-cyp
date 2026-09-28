"""End-to-end cross-validation, fitting, and submission generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from .constants import (
    ACTIVITY_ENDPOINTS,
    ACTIVITY_TRAIN_FILE,
    TDI_ENDPOINTS,
    TDI_TRAIN_FILE,
    TEST_FILE,
    lower_bound_column,
    upper_bound_column,
)
from .data import load_activity, load_tdi, load_test
from .download import sha256_file
from .features import featurize_smiles
from .metrics import soft_threshold_rae
from .models import make_classifier, make_regressor
from .splits import (
    assert_group_disjoint,
    classification_splits,
    regression_splits,
)
from .validation import validate_submission


@dataclass(frozen=True)
class BaselineConfig:
    track: str
    model: str
    n_splits: int = 5
    seed: int = 42
    radius: int = 2
    n_bits: int = 2048


def _best_mcc_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.05, 0.95, 181)
    scores = np.asarray(
        [matthews_corrcoef(labels, probabilities >= threshold) for threshold in candidates]
    )
    best = int(np.nanargmax(scores))
    return float(candidates[best]), float(scores[best])


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _activity_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_features: np.ndarray,
    test_features: np.ndarray,
    output_dir: Path,
    config: BaselineConfig,
) -> tuple[dict[str, object], Path]:
    submission = test[["Molecule_Name", "SMILES"]].copy()
    metrics: dict[str, object] = {"endpoints": {}}
    oof_rows: list[pd.DataFrame] = []

    for endpoint in ACTIVITY_ENDPOINTS:
        mask = train[endpoint].notna().to_numpy()
        frame = train.loc[mask].reset_index(drop=True)
        features = train_features[mask]
        labels = frame[endpoint].to_numpy(dtype=float)
        lower = frame[lower_bound_column(endpoint)].to_numpy(dtype=float)
        upper = frame[upper_bound_column(endpoint)].to_numpy(dtype=float)
        oof = np.full(len(frame), np.nan)
        folds = np.full(len(frame), -1, dtype=int)
        fold_scores: list[float] = []
        split_iterator = regression_splits(frame["SMILES"].tolist(), config.n_splits)
        for fold, (fit_idx, val_idx) in enumerate(split_iterator):
            assert_group_disjoint(frame["SMILES"].tolist(), fit_idx, val_idx)
            model = make_regressor(config.model, config.seed + fold)
            model.fit(features[fit_idx], labels[fit_idx])
            oof[val_idx] = model.predict(features[val_idx])
            folds[val_idx] = fold
            fold_scores.append(
                soft_threshold_rae(labels[val_idx], oof[val_idx], lower[val_idx], upper[val_idx])
            )

        endpoint_score = soft_threshold_rae(labels, oof, lower, upper)
        metrics["endpoints"][endpoint] = {
            "labels": len(frame),
            "oof_st_rae": endpoint_score,
            "fold_st_rae": fold_scores,
        }
        oof_rows.append(
            pd.DataFrame(
                {
                    "Molecule_Name": frame["Molecule_Name"],
                    "endpoint": endpoint,
                    "fold": folds,
                    "y_true": labels,
                    "y_pred": oof,
                    "conf_low": lower,
                    "conf_high": upper,
                }
            )
        )

        final_model = make_regressor(config.model, config.seed)
        final_model.fit(features, labels)
        submission[endpoint] = final_model.predict(test_features)
        joblib.dump(final_model, output_dir / f"{endpoint}.joblib")

    metrics["MA-ST-RAE"] = float(
        np.mean([entry["oof_st_rae"] for entry in metrics["endpoints"].values()])
    )
    pd.concat(oof_rows, ignore_index=True).to_csv(output_dir / "oof_predictions.csv", index=False)
    path = output_dir / "activity_submission.csv"
    submission.to_csv(path, index=False)
    return metrics, path


def _tdi_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_features: np.ndarray,
    test_features: np.ndarray,
    output_dir: Path,
    config: BaselineConfig,
) -> tuple[dict[str, object], Path]:
    submission = test[["Molecule_Name", "SMILES"]].copy()
    metrics: dict[str, object] = {"endpoints": {}}
    oof_rows: list[pd.DataFrame] = []

    for endpoint in TDI_ENDPOINTS:
        mask = train[endpoint].notna().to_numpy()
        frame = train.loc[mask].reset_index(drop=True)
        features = train_features[mask]
        labels = frame[endpoint].astype(int).to_numpy()
        oof = np.full(len(frame), np.nan)
        folds = np.full(len(frame), -1, dtype=int)
        split_iterator = classification_splits(
            frame["SMILES"].tolist(), labels, config.n_splits, config.seed
        )
        fold_scores: list[float] = []
        for fold, (fit_idx, val_idx) in enumerate(split_iterator):
            assert_group_disjoint(frame["SMILES"].tolist(), fit_idx, val_idx)
            model = make_classifier(config.model, config.seed + fold)
            model.fit(features[fit_idx], labels[fit_idx])
            oof[val_idx] = model.predict_proba(features[val_idx])[:, 1]
            folds[val_idx] = fold
            fold_scores.append(float(matthews_corrcoef(labels[val_idx], oof[val_idx] >= 0.5)))

        threshold, tuned_mcc = _best_mcc_threshold(labels, oof)
        default_mcc = float(matthews_corrcoef(labels, oof >= 0.5))
        metrics["endpoints"][endpoint] = {
            "labels": len(frame),
            "positives": int(labels.sum()),
            "oof_mcc_at_0.5": default_mcc,
            "oof_selected_threshold": threshold,
            "oof_mcc_at_selected_threshold": tuned_mcc,
            "fold_mcc_at_0.5": fold_scores,
            "note": (
                "Threshold-selected OOF MCC is a model-selection estimate, "
                "not a final unbiased score."
            ),
        }
        oof_rows.append(
            pd.DataFrame(
                {
                    "Molecule_Name": frame["Molecule_Name"],
                    "endpoint": endpoint,
                    "fold": folds,
                    "y_true": labels,
                    "probability": oof,
                }
            )
        )

        final_model = make_classifier(config.model, config.seed)
        final_model.fit(features, labels)
        probabilities = final_model.predict_proba(test_features)[:, 1]
        submission[endpoint] = probabilities >= threshold
        joblib.dump(
            {"model": final_model, "threshold": threshold},
            output_dir / f"{endpoint}.joblib",
        )

    metrics["MA-MCC@0.5"] = float(
        np.mean([entry["oof_mcc_at_0.5"] for entry in metrics["endpoints"].values()])
    )
    metrics["MA-MCC@selected-threshold"] = float(
        np.mean([entry["oof_mcc_at_selected_threshold"] for entry in metrics["endpoints"].values()])
    )
    pd.concat(oof_rows, ignore_index=True).to_csv(output_dir / "oof_predictions.csv", index=False)
    path = output_dir / "tdi_submission.csv"
    submission.to_csv(path, index=False)
    return metrics, path


def run_baseline(config: BaselineConfig, data_dir: Path, output_dir: Path) -> dict[str, object]:
    if config.track not in {"activity", "tdi"}:
        raise ValueError("track must be 'activity' or 'tdi'")
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    train = load_activity(data_dir) if config.track == "activity" else load_tdi(data_dir)
    test = load_test(data_dir)
    train_features = featurize_smiles(train["SMILES"], radius=config.radius, n_bits=config.n_bits)
    test_features = featurize_smiles(test["SMILES"], radius=config.radius, n_bits=config.n_bits)

    if config.track == "activity":
        metrics, submission_path = _activity_baseline(
            train, test, train_features, test_features, output_dir, config
        )
    else:
        metrics, submission_path = _tdi_baseline(
            train, test, train_features, test_features, output_dir, config
        )

    report = validate_submission(submission_path, test, config.track)
    if not report.valid:
        raise RuntimeError(f"Generated invalid submission: {report.errors}")
    _write_json(output_dir / "metrics.json", metrics)
    train_file = ACTIVITY_TRAIN_FILE if config.track == "activity" else TDI_TRAIN_FILE
    input_files = {
        train_file: sha256_file(data_dir / train_file),
        TEST_FILE: sha256_file(data_dir / TEST_FILE),
    }
    dataset_manifest_path = data_dir / "manifest.json"
    dataset_manifest = (
        json.loads(dataset_manifest_path.read_text()) if dataset_manifest_path.is_file() else None
    )
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "input_sha256": input_files,
        "dataset_manifest": dataset_manifest,
        "submission": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "validation": report.to_dict(),
        "feature_shape": {
            "train": list(train_features.shape),
            "test": list(test_features.shape),
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return {"metrics": metrics, "manifest": manifest, "submission": str(submission_path)}
