"""Private, explicitly single-fold matched representation results."""

import json
from pathlib import Path

import pandas as pd

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.download import sha256_file
from somerra_openadmet.frozen_benchmark import paired_seed_bootstrap
from somerra_openadmet.representations import atomic_json


def report(root):
    root = Path(root)
    design = json.loads((root / "design.json").read_text())
    records, predictions, manifests = [], [], []
    for encoder, view, _cache, _checkpoint, scaling, seed in design["matrix"]:
        name = f"{encoder}-{view}-{scaling}-seed{seed}"
        folder = root / name
        manifest = json.loads((folder / "run_manifest.json").read_text())
        if not manifest.get("pilot_complete") or not manifest.get("requested_folds_complete"):
            raise ValueError(f"Incomplete pilot fit: {name}")
        if sha256_file(folder / "oof_predictions.csv") != manifest["oof_sha256"]:
            raise ValueError(f"Changed predictions: {name}")
        arm = f"{encoder}-{view}-{scaling}"
        row = {
            "arm": arm,
            "seed": seed,
            "dimension": manifest["input_dimension"],
            "MA-ST-RAE": manifest["metrics"]["MA-ST-RAE"],
            "macro_MAE": manifest["metrics"]["macro_MAE"],
            "gpu_hours": manifest["gpu_hours"],
            "head_parameters": manifest["training"]["head_parameters"],
            "cpu_replay_max_abs": manifest["cpu_replay_max_abs"],
        }
        for ep, value in manifest["metrics"]["endpoints"].items():
            row[ep + "_ST-RAE"] = value["ST-RAE"]
            row[ep + "_MAE"] = value["MAE"]
            row[ep + "_n"] = value["n"]
        records.append(row)
        pred = pd.read_csv(folder / "oof_predictions.csv")
        pred["arm"], pred["seed"] = arm, seed
        predictions.append(pred)
        manifests.append(manifest)
    if len(records) != 21:
        raise ValueError("The predeclared matrix requires 21 fits")
    # All arms must have exactly the same fit/holdout rows, batches, and updates per seed.
    for seed in (42, 43, 44):
        selected = [m for m in manifests if m["config"]["seed"] == seed]
        for key in ("batch_order_sha256", "successful_updates"):
            if len({m["training"][key] for m in selected}) != 1:
                raise ValueError(f"Unmatched training protocol: {key}")
        if len({m["prepared_manifest"]["training_sha256"] for m in selected}) != 1:
            raise ValueError("Unmatched training data")
    scores = pd.DataFrame(records)
    all_predictions = pd.concat(predictions, ignore_index=True)
    scores.to_csv(root / "scores.csv", index=False)
    all_predictions.to_csv(root / "paired_predictions.csv", index=False)
    metrics = ["MA-ST-RAE", "macro_MAE"] + [
        ep + suffix for ep in ACTIVITY_ENDPOINTS for suffix in ("_ST-RAE", "_MAE")
    ]
    summary = []
    for arm, group in scores.groupby("arm", sort=False):
        row = {
            "arm": arm,
            "dimension": int(group.dimension.iloc[0]),
            "head_parameters": int(group.head_parameters.iloc[0]),
            "gpu_hours": float(group.gpu_hours.sum()),
        }
        for metric in metrics:
            row[metric + "_mean"] = float(group[metric].mean())
            row[metric + "_sd"] = float(group[metric].std(ddof=1))
        summary.append(row)
    pd.DataFrame(summary).to_csv(root / "summary.csv", index=False)
    comparisons = []
    for scaling in ("none", "standard"):
        candidate = f"atom_jepa-invariant-{scaling}"
        for baseline in (f"chemeleon-mean-{scaling}", f"mol_jepa-cls-{scaling}"):
            comparisons.append(paired_seed_bootstrap(all_predictions, candidate, baseline))
        comparisons.append(
            paired_seed_bootstrap(
                all_predictions, f"mol_jepa-cls-{scaling}", f"chemeleon-mean-{scaling}"
            )
        )
    comparisons.append(
        paired_seed_bootstrap(all_predictions, "atom_jepa-invariant-none", "atom_jepa-scalar-none")
    )
    for encoder, view in (("chemeleon", "mean"), ("mol_jepa", "cls"), ("atom_jepa", "invariant")):
        comparisons.append(
            paired_seed_bootstrap(
                all_predictions, f"{encoder}-{view}-standard", f"{encoder}-{view}-none"
            )
        )
    atomic_json(root / "paired_comparisons.json", comparisons)
    primary = ["chemeleon-mean-none", "mol_jepa-cls-none", "atom_jepa-invariant-none"]
    residuals = {}
    for endpoint in ACTIVITY_ENDPOINTS:
        part = all_predictions[
            (all_predictions.endpoint == endpoint) & all_predictions.arm.isin(primary)
        ].copy()
        part["residual"] = part.y_pred - part.y_true
        wide = part.pivot(index=["Molecule_Name", "seed"], columns="arm", values="residual")
        residuals[endpoint] = wide.corr().to_dict()
    atomic_json(root / "residual_correlations.json", residuals)
    from run_frozen_representation_pilot import EXTRACTION, charged_hours

    extraction = json.loads((EXTRACTION / "run_manifest.json").read_text())

    budget = {
        "cap_gpu_hours": 4,
        "charged_gpu_hours": charged_hours(),
        "head_fit_gpu_hours": float(scores.gpu_hours.sum()),
        "extraction_gpu_hours": extraction["gpu_hours"],
        "projection_five_fold_head_hours": float(scores.gpu_hours.sum() * 5),
        "note": "Projection only; five-fold campaign not authorized or launched.",
    }
    if budget["charged_gpu_hours"] > 4:
        raise ValueError("Incremental GPU cap exceeded")
    atomic_json(root / "budget_summary.json", budget)
    historical_path = root / "historical_head_replay.json"
    historical_note = "Historical baseline replay evidence was not supplied for this report."
    if historical_path.exists():
        historical = json.loads(historical_path.read_text())
        maximum = max(v["max_prediction_difference"] for v in historical.values())
        historical_note = (
            "Fresh seed-42 baseline fits versus historical frozen-reference fold-0 predictions: "
            f"maximum difference {maximum:.9g} pIC50; see historical_head_replay.json."
        )
    validation_note = "Test evidence was not supplied for this report."
    validation_path = Path("runs/atom-jepa-validation.xml")
    if validation_path.exists():
        import xml.etree.ElementTree as ET

        suites = list(ET.parse(validation_path).getroot().iter("testsuite"))
        totals = {
            key: sum(int(s.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        validation_note = f"Targeted validation: {totals}; see runs/atom-jepa-validation.xml."
        atomic_json(
            root / "test_evidence.json",
            {
                "junit_sha256": sha256_file(validation_path),
                "counts": totals,
            },
        )
    lines = [
        "# Matched frozen-representation pilot",
        "",
        "Outer fold 0; seeds 42/43/44. Lower MA-ST-RAE is better. All encoders frozen.",
        "The primary comparison uses raw features and a single fixed Atom-JEPA conformer.",
        "Values are means and sample standard deviations of per-seed scores, "
        "not seed-ensemble scores.",
        "",
        "| Arm | Dimension | MLP parameters | MA-ST-RAE (mean ± SD) | Macro MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['arm']} | {row['dimension']} | {row['head_parameters']:,} | "
            f"{row['MA-ST-RAE_mean']:.6f} ± {row['MA-ST-RAE_sd']:.6f} | "
            f"{row['macro_MAE_mean']:.6f} |"
        )
    lines += [
        "",
        "## Paired comparisons",
        "",
        "Negative differences favor the candidate.",
        "",
        "| Candidate | Reference | 95% cluster-bootstrap interval for macro difference |",
        "|---|---|---|",
    ]
    for result in comparisons:
        lo, hi = result["delta_95ci"]
        lines.append(f"| {result['candidate']} | {result['reference']} | [{lo:.6f}, {hi:.6f}] |")
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "These are exploratory single-fold results. The paired bootstrap resamples molecular "
        "clusters jointly across endpoints and seeds; it does not create independent test folds.",
        "Input dimensions and first-layer parameter counts differ. This measures representation "
        "utility under a common MLP protocol, not an isolated causal effect "
        "of pretraining architecture.",
        "Feature standardization is a separate sensitivity analysis fitted only on outer-training "
        "rows. No outer labels choose epochs, scaling, representation, or checkpoints.",
        "The encoders have different pretraining corpora and modalities; checkpoint provenance "
        "does not establish absence of pretraining overlap with the challenge molecules.",
        "No fine-tuning, conformer averaging, promotion, blind export, "
        "or challenge submission occurred.",
        "The upstream Atom-JEPA CPU and GPU encoder exhibited an approximately 0.015% "
        "relative L2 rotation residual for invariant features in the pre-score audit. "
        "Geometry tolerance is 0.1% relative L2 and 0.005 absolute; cache parity and "
        "prediction replay retain their original tolerances. Diagnostics and failed "
        "extraction attempts are retained and charged to this pilot.",
        "",
        "## Runtime and verification",
        "",
        f"Incremental charged GPU time: {budget['charged_gpu_hours']:.6f} / 4 hours.",
        f"Largest CPU checkpoint-replay error: {scores.cpu_replay_max_abs.max():.9g} pIC50 "
        "(limit 1e-4).",
        f"Atom-JEPA extraction: {extraction['molecules_per_second']:.3f} molecules/s; "
        f"peak allocated memory {extraction['peak_allocated_bytes'] / 2**30:.3f} GiB.",
        "All 21 fits have identical molecule batch-order hashes and optimizer-update counts "
        "for each paired seed.",
        historical_note,
        validation_note,
        "Per-endpoint scores and coverage: scores.csv and summary.csv. Full paired predictions "
        "and residual correlations are retained alongside this report.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    import sys

    report(Path(sys.argv[1]))
