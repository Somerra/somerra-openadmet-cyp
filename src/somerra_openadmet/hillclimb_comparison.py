"""Paired scaffold bootstrap for frozen internal-test predictions; no model fitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ACTIVITY_ENDPOINTS
from .download import sha256_file
from .reference_member import score_predictions
from .reference_member_data import load_frames
from .regression_report import paired_bootstrap
from .splits import scaffold_groups


def paired_comparison(truth, reference, candidate, *, samples=1000):
    keys = ["Molecule_Name", "parent"]
    for predictions in (reference, candidate):
        if predictions.Molecule_Name.duplicated().any() or not truth[keys].equals(
            predictions[keys]
        ):
            raise ValueError("Paired prediction identities/order do not match internal test")
        if not np.isfinite(predictions[list(ACTIVITY_ENDPOINTS)].to_numpy()).all():
            raise ValueError("Nonfinite paired predictions")
    groups = scaffold_groups(truth.parent.tolist())
    if len(set(groups)) < 2:
        raise ValueError("Paired bootstrap requires at least two molecular groups")
    frames = []
    for ep in ACTIVITY_ENDPOINTS:
        observed = truth[ep].notna().to_numpy()
        frame = pd.DataFrame(
            {
                "endpoint": ep,
                "cluster": groups[observed],
                "y_true": truth.loc[observed, ep].to_numpy(),
                "conf_low": truth.loc[observed, ep + "_conf_low"].to_numpy(),
                "conf_high": truth.loc[observed, ep + "_conf_high"].to_numpy(),
                "reference": reference.loc[observed, ep].to_numpy(),
                "candidate": candidate.loc[observed, ep].to_numpy(),
            }
        )
        if not np.isfinite(frame.drop(columns=["endpoint", "cluster"]).to_numpy()).all():
            raise ValueError("Invalid paired labels/bounds")
        if (frame.conf_low > frame.conf_high).any():
            raise ValueError("Reversed paired bounds")
        frames.append(frame)
    result = paired_bootstrap(pd.concat(frames), "candidate", "reference", samples=samples)
    scores = [score_predictions(truth, p[list(ACTIVITY_ENDPOINTS)]) for p in [reference, candidate]]
    result.update(
        point_delta=scores[1]["MA-ST-RAE"] - scores[0]["MA-ST-RAE"],
        endpoint_deltas={
            ep: scores[1]["endpoints"][ep] - scores[0]["endpoints"][ep] for ep in ACTIVITY_ENDPOINTS
        },
        grouping="Generic Bemis-Murcko scaffold; acyclic molecules grouped by canonical identity",
        groups=len(set(groups)),
        seed=42,
        interpretation=(
            "Conditional on these fitted models; does not estimate training-seed variability"
        ),
    )
    return result


def compare_runs(bundle, reference_run, candidate_run):
    bundle, reference_run, candidate_run = map(Path, (bundle, reference_run, candidate_run))
    records = [
        json.loads((run / "run_manifest.json").read_text())
        for run in [reference_run, candidate_run]
    ]
    digest = sha256_file(bundle / "manifest.json")
    for record in records:
        if (
            not record.get("complete")
            or record.get("final")
            or record["data_manifest_sha256"] != digest
        ):
            raise ValueError("Paired comparison requires complete runs on the same frozen bundle")
    for key in ["seed", "epoch_limit", "scheduler_horizon", "batch_size", "tasks"]:
        if records[0][key] != records[1][key]:
            raise ValueError(f"Paired run contract mismatch: {key}")
    load_frames(bundle)  # Validate bundle content hashes and partition/exclusion contract.
    truth = pd.read_csv(bundle / "official.csv")
    truth = truth.loc[truth.partition == "internal_test"].reset_index(drop=True)
    paths = [run / "internal-test.csv" for run in [reference_run, candidate_run]]
    predictions = [pd.read_csv(path) for path in paths]
    result = paired_comparison(truth, *predictions)
    for record, prediction in zip(records, predictions, strict=True):
        score = score_predictions(truth, prediction[list(ACTIVITY_ENDPOINTS)])
        # Legacy CSVs serialize float32 predictions with limited decimal precision.
        if not np.isclose(
            score["MA-ST-RAE"], record["internal_test_metrics"]["MA-ST-RAE"], rtol=0, atol=1e-7
        ):
            raise ValueError("Saved internal predictions do not reproduce recorded score")
    result.update(
        reference_run=str(reference_run),
        candidate_run=str(candidate_run),
        bundle_sha256=digest,
        prediction_sha256=[sha256_file(path) for path in paths],
    )
    return result
