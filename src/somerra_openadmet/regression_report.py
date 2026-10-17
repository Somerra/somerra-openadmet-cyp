"""Training-only diagnostics and paired cluster resampling for regression runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .campaign import write_json
from .constants import ACTIVITY_ENDPOINTS
from .data import load_activity, load_test
from .metrics import soft_threshold_rae


def paired_bootstrap(oof: pd.DataFrame, candidate: str, reference: str, *, samples=1000, seed=42):
    """Resample molecular clusters jointly across endpoints; negative delta wins."""
    rng = np.random.default_rng(seed)
    groups = np.unique(oof.cluster)
    lookup = {group: i for i, group in enumerate(groups)}
    endpoint_data = []
    for endpoint in ACTIVITY_ENDPOINTS:
        frame = oof.loc[oof.endpoint == endpoint]
        y, lo, hi = (frame[c].to_numpy() for c in ("y_true", "conf_low", "conf_high"))
        errors = [
            np.maximum(frame[c].to_numpy() - hi, 0) + np.maximum(lo - frame[c].to_numpy(), 0)
            for c in (candidate, reference)
        ]
        endpoint_data.append((y, lo, hi, errors, np.array([lookup[g] for g in frame.cluster])))
    deltas = []
    for _ in range(samples):
        counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
        differences = []
        for y, lo, hi, errors, indices in endpoint_data:
            weight = counts[indices]
            if weight.sum() == 0:
                break
            mean = np.average(y, weights=weight)
            denominator = np.sum(weight * (np.maximum(mean - hi, 0) + np.maximum(lo - mean, 0)))
            if denominator <= 0:
                break
            differences.append(float(np.sum(weight * (errors[0] - errors[1])) / denominator))
        if len(differences) == len(ACTIVITY_ENDPOINTS):
            deltas.append(np.mean(differences))
    if len(deltas) < samples * 0.95:
        raise ValueError("Too many undefined bootstrap scores")
    return {
        "candidate": candidate,
        "reference": reference,
        "samples": len(deltas),
        "delta_95ci": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "probability_improvement": float(np.mean(np.asarray(deltas) < 0)),
    }


def compare(
    run_dir: Path,
    reference: str = "lightgbm",
    reference_run: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    if output_path is not None and output_path.exists():
        raise FileExistsError(f"Comparison output already exists: {output_path}")
    oof = pd.read_csv(run_dir / "oof_predictions.csv")
    candidate = "nested_ensemble" if "nested_ensemble" in oof else "y_pred"
    if reference_run is not None:
        other = pd.read_csv(reference_run / "oof_predictions.csv")
        keys = ["Molecule_Name", "endpoint", "cluster", "fold", "y_true", "conf_low", "conf_high"]
        left = oof.sort_values(["Molecule_Name", "endpoint"]).reset_index(drop=True)
        right = other.sort_values(["Molecule_Name", "endpoint"]).reset_index(drop=True)
        if not left[keys].equals(right[keys]):
            raise ValueError("Comparison runs have different rows, labels, or split assignments")
        oof = left
        oof["reference_run"] = right[reference]
        reference = "reference_run"
    if oof.duplicated(["Molecule_Name", "endpoint"]).any():
        raise ValueError("Duplicate molecule/endpoint predictions")
    if not np.isfinite(oof[[candidate, reference, "y_true", "conf_low", "conf_high"]]).all(
        axis=None
    ):
        raise ValueError("Non-finite comparison predictions or labels")
    scores = {
        ep: {
            name: soft_threshold_rae(frame.y_true, frame[name], frame.conf_low, frame.conf_high)
            for name in (candidate, reference)
        }
        for ep in ACTIVITY_ENDPOINTS
        for frame in [oof.loc[oof.endpoint == ep]]
    }
    result = paired_bootstrap(oof, candidate, reference)
    base, new = (
        float(np.mean([s[name] for s in scores.values()])) for name in (reference, candidate)
    )
    deltas = {ep: m[candidate] - m[reference] for ep, m in scores.items()}
    result.update(
        {
            "relative_macro_improvement": (base - new) / base,
            "candidate_run": str(run_dir),
            "candidate_macro": new,
            "reference_macro": base,
            "endpoint_deltas": deltas,
            "passes_numerical_gate": bool(
                new <= 0.97 * base and result["delta_95ci"][1] < 0 and max(deltas.values()) <= 0.02
            ),
            "note": "Numerical comparison only; promotion also requires the campaign gates.",
            "reference_run": str(reference_run) if reference_run else None,
        }
    )
    write_json(output_path if output_path is not None else run_dir / "comparison.json", result)
    return result


def diagnose(data_dir: Path, run_dir: Path, output_dir: Path) -> dict:
    train, test = load_activity(data_dir), load_test(data_dir)
    oof = pd.read_csv(run_dir / "oof_predictions.csv")
    submission = pd.read_csv(run_dir / "activity_submission.csv")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in train.SMILES]
    test_fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in test.SMILES]
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {}
    for endpoint in ACTIVITY_ENDPOINTS:
        frame = oof.loc[oof.endpoint == endpoint].copy()
        labeled_fps = [
            fp for fp, valid in zip(train_fps, train[endpoint].notna(), strict=True) if valid
        ]
        nearest = np.asarray(
            [max(DataStructs.BulkTanimotoSimilarity(fp, labeled_fps)) for fp in test_fps]
        )
        prediction = submission.set_index("Molecule_Name").loc[test.Molecule_Name, endpoint]
        truth = train[endpoint].dropna()
        result[endpoint] = {
            "training_labels": len(truth),
            "training_quantiles": truth.quantile([0, 0.1, 0.5, 0.9, 1]).tolist(),
            "test_prediction_quantiles": prediction.quantile([0, 0.1, 0.5, 0.9, 1]).tolist(),
            "test_nearest_similarity_quantiles": np.quantile(
                nearest, [0, 0.1, 0.5, 0.9, 1]
            ).tolist(),
            "oof_mean_signed_error": float((frame.y_pred - frame.y_true).mean()),
        }
        frame["potency_bin"] = pd.cut(frame.y_true, [-np.inf, 4, 5, 6, np.inf])
        bins = []
        for label, part in frame.groupby("potency_bin", observed=True):
            try:
                score = soft_threshold_rae(part.y_true, part.y_pred, part.conf_low, part.conf_high)
            except ValueError:
                score = None
            bins.append(
                {
                    "bin": str(label),
                    "rows": len(part),
                    "st_rae": score,
                    "mean_signed_error": float((part.y_pred - part.y_true).mean()),
                    "mean_interval_width": float((part.conf_high - part.conf_low).mean()),
                }
            )
        result[endpoint]["potency_bins"] = bins
        pd.DataFrame(
            {
                "Molecule_Name": test.Molecule_Name,
                "nearest_similarity": nearest,
                "prediction": prediction.to_numpy(),
            }
        ).to_csv(output_dir / f"{endpoint}_test_diagnostics.csv", index=False)
    write_json(output_dir / "diagnostics.json", result)
    return result
