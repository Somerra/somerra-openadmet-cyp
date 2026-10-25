"""Summarize completed ablations; never infer scores for unfinished trials."""

import argparse
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("campaign", type=Path)
args = parser.parse_args()
record = json.loads((args.campaign / "results.json").read_text())
rows = []
for trial in record["results"]:
    if trial["status"] != "completed":
        continue
    manifest = json.loads(
        (
            Path(trial.get("run_dir", args.campaign / ("trial-" + trial["name"])))
            / "run_manifest.json"
        ).read_text()
    )
    if not manifest["complete"]:
        raise ValueError("Completed result has an incomplete manifest")
    metrics = trial["metrics"]
    row = {
        "configuration": trial["name"],
        "evaluation_split": "validation_cv_official_training",
        "macro": metrics["MA-ST-RAE"],
        **metrics["endpoints"],
        "elapsed_minutes": trial["elapsed_seconds"] / 60,
    }
    folds = manifest["fold_training"].values()
    row["mean_training_parents"] = sum(f["parents"] for f in folds) / len(manifest["fold_training"])
    row["optimizer_steps"] = sum(f["optimizer_steps"] for f in folds)
    for key in ("versus_full", "versus_direct"):
        if key in trial:
            comparison = trial[key]
            row[key + "_delta"] = comparison["candidate_macro"] - comparison["reference_macro"]
            row[key + "_ci_low"], row[key + "_ci_high"] = comparison["delta_95ci"]
    rows.append(row)
frame = pd.DataFrame(rows)
frame.to_csv(args.campaign / "summary.csv", index=False)
print("Local validation only: five folds within official training data; not official test scores.")
print(frame.to_string(index=False))
print("Exploratory one-seed, 15-epoch comparisons; intervals are not multiplicity-adjusted.")
