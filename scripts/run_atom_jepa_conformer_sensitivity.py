"""Frozen 1/2/10-conformer extension, charged to the original four-hour pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_frozen_representation_pilot import CAP, EXTRACTION, LEDGER, charged_hours
from run_frozen_representation_pilot import ROOT as PILOT

from somerra_openadmet.conformer_sensitivity import (
    CONFORMER_SEEDS,
    COUNTS,
    cache_stage,
    generate_extra,
    load_conformer_features,
    predict_conformers,
    verify_files,
)
from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.download import sha256_file
from somerra_openadmet.frozen_benchmark import (
    PROTOCOL,
    RECIPE,
    paired_seed_bootstrap,
    restore_head,
    scores,
    screen,
)
from somerra_openadmet.frozen_resume import completed_head, validate_evaluation
from somerra_openadmet.representation_training import oof_rows
from somerra_openadmet.representations import atomic_json, prepared_data

ROOT = PILOT / "conformer-sensitivity-v1"
PREPARED = Path("runs/regression-corpus-v1")
HEAD_SEEDS = (42, 43, 44)
VIEWS = ("scalar", "invariant")


def allocation(maximum):
    amount = min(maximum, CAP - charged_hours())
    if amount <= 0:
        raise TimeoutError("Original four-GPU-hour budget exhausted")
    return amount


def config_for(output, maximum=0.1):
    return {
        "output_dir": str(output),
        "device": "cuda",
        "precision": "float32",
        "threads": 8,
        "max_gpu_hours": allocation(maximum),
        "ledger": str(LEDGER),
        "prepared_dir": str(PREPARED),
        "checkpoint": "runs/atom-jepa-model",
    }


def head_path(view, count, seed):
    if view == "invariant" and count == 1:
        return PILOT / f"atom_jepa-invariant-standard-seed{seed}"
    return ROOT / "heads" / f"{view}-train{count}-seed{seed}"


def fit_multi(view, count, seed):
    output = head_path(view, count, seed)
    config = {
        **config_for(output),
        "protocol": PROTOCOL,
        "encoder": "atom_jepa",
        "representation": view,
        "cache_dir": str(ROOT / "features"),
        "scaling": "standard",
        "train_conformers": count,
        "scaler_conformer": 0,
        "selected_folds": [0],
        "adaptation": "frozen",
        "seed": seed,
    }
    from somerra_openadmet.conformer_sensitivity import fit_stage

    config_path = ROOT / f"{output.name}.json"
    atomic_json(config_path, config)
    return fit_stage(config_path)


def fit_all():
    for count in COUNTS:
        for view in VIEWS:
            for seed in HEAD_SEEDS:
                path = head_path(view, count, seed)
                if completed_head(
                    path,
                    {
                        "representation": view,
                        "train_conformers": count,
                        "scaling": "standard",
                        "seed": seed,
                        "prepared_dir": str(PREPARED),
                        "cache_dir": str(EXTRACTION if count == 1 else ROOT / "features"),
                    },
                ):
                    continue
                print(f"Fitting {view}, train conformers={count}, seed={seed}", flush=True)
                if count == 1:
                    config = {
                        **config_for(path),
                        "protocol": PROTOCOL,
                        "encoder": "atom_jepa",
                        "representation": view,
                        "cache_dir": str(EXTRACTION),
                        "scaling": "standard",
                        "selected_folds": [0],
                        "adaptation": "frozen",
                        "seed": seed,
                    }
                    config_path = ROOT / f"{path.name}.json"
                    atomic_json(config_path, config)
                    screen(config_path)
                else:
                    fit_multi(view, count, seed)
                if not completed_head(
                    path, {"cache_dir": str(EXTRACTION if count == 1 else ROOT / "features")}
                ):
                    raise RuntimeError("Fit did not complete")


def evaluate(features, frame, prepared):
    """CPU-only evaluation; original single-view predictions anchor numerical parity."""
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    output = ROOT / "evaluation"
    if output.exists():
        validate_evaluation(output, prepared, ROOT / "features")
        return
    output.mkdir()
    held = np.flatnonzero(frame.fold.to_numpy() == 0)
    fit_ids = frame.loc[frame.fold != 0, "Molecule_Name"].tolist()
    predictions, metrics, diagnostics = [], [], []
    checkpoints = {}
    batch_hashes, conformer_hashes = {}, {}
    reference_scalers = {}
    for count in COUNTS:
        for view in VIEWS:
            x = features[held, :, :256] if view == "scalar" else features[held]
            for seed in HEAD_SEEDS:
                path = head_path(view, count, seed)
                if not completed_head(path):
                    raise ValueError("Missing fit")
                evidence = json.loads((path / "run_manifest.json").read_text())
                model, saved = restore_head(path / "fold-0.pt")
                if (
                    saved["fit_ids"] != fit_ids
                    or saved["held_ids"] != frame.iloc[held].Molecule_Name.tolist()
                    or saved["training_sha256"] != prepared["training_sha256"]
                    or saved["recipe"] != RECIPE
                ):
                    raise ValueError("Unmatched fit data or recipe")
                key = (view, seed)
                scaler = (model.mean.numpy(), model.scale.numpy())
                if key not in reference_scalers:
                    reference_scalers[key] = scaler
                elif any(
                    not np.array_equal(a, b)
                    for a, b in zip(reference_scalers[key], scaler, strict=True)
                ):
                    raise ValueError("Scaler changed between conformer arms")
                training = evidence["training"]
                pair = (training["batch_order_sha256"], training["successful_updates"])
                historical = json.loads(
                    (
                        PILOT / f"atom_jepa-invariant-standard-seed{seed}" / "run_manifest.json"
                    ).read_text()
                )["training"]
                if pair != (historical["batch_order_sha256"], historical["successful_updates"]):
                    raise ValueError("Molecule order or update count differs from original pilot")
                batch_hashes[str(seed)] = list(pair)
                if count > 1:
                    ckey = f"{count}-{seed}"
                    digest = training["conformer_order_sha256"]
                    if ckey in conformer_hashes and conformer_hashes[ckey] != digest:
                        raise ValueError("Conformer sampling differs between feature views")
                    conformer_hashes[ckey] = digest
                values = predict_conformers(model, x)
                original = pd.read_csv(path / "oof_predictions.csv")
                replay = oof_rows(frame, held, values[:, 0])
                # CSV roundtrips can alter float representation by a few ulps.
                if not original[["Molecule_Name", "endpoint", "fold"]].equals(
                    replay[["Molecule_Name", "endpoint", "fold"]]
                ):
                    raise ValueError("Prediction identities differ")
                delta = float(np.abs(original.y_pred.to_numpy() - replay.y_pred.to_numpy()).max())
                if delta > 1e-4:
                    raise ValueError("Single-conformer prediction replay failed")
                name = f"{view}-train{count}-seed{seed}"
                np.save(output / f"{name}-per-conformer.npy", values, allow_pickle=False)
                checkpoints[str(path / "fold-0.pt")] = sha256_file(path / "fold-0.pt")
                for endpoint_index, endpoint in enumerate(ACTIVITY_ENDPOINTS):
                    observed = frame.iloc[held][endpoint].notna().to_numpy()
                    p = values[observed, :, endpoint_index]
                    diagnostics.append(
                        {
                            "view": view,
                            "train_conformers": count,
                            "seed": seed,
                            "endpoint": endpoint,
                            "mean_conformer_prediction_sd": float(p.std(axis=1).mean()),
                            "mean_conformer_prediction_range": float(np.ptp(p, axis=1).mean()),
                            "p95_conformer_prediction_range": float(
                                np.quantile(np.ptp(p, axis=1), 0.95)
                            ),
                            "single_conformer_replay_max_abs": delta,
                        }
                    )
                for test_count in COUNTS:
                    arm = f"{view}-train{count}-test{test_count}"
                    rows = oof_rows(frame, held, values[:, :test_count].mean(axis=1))
                    result = scores(rows)
                    metric = {
                        "arm": arm,
                        "view": view,
                        "train_conformers": count,
                        "test_conformers": test_count,
                        "seed": seed,
                        "MA-ST-RAE": result["MA-ST-RAE"],
                        "macro_MAE": result["macro_MAE"],
                    }
                    for endpoint, score in result["endpoints"].items():
                        for field in ("ST-RAE", "MAE", "n"):
                            metric[f"{endpoint}_{field}"] = score[field]
                    metrics.append(metric)
                    rows["arm"], rows["seed"] = arm, seed
                    predictions.append(rows)
    pd.DataFrame(metrics).to_csv(output / "scores.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "paired_predictions.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "conformer_sensitivity.csv", index=False)
    atomic_json(
        output / "manifest.json",
        {
            "complete": True,
            "checkpoints": checkpoints,
            "training_sha256": prepared["training_sha256"],
            "cache_manifest_sha256": sha256_file(ROOT / "features" / "manifest.json"),
            "batch_hashes_and_updates": batch_hashes,
            "conformer_sampling_hashes": conformer_hashes,
            "files": {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()},
        },
    )


def report():
    evaluation = ROOT / "evaluation"
    _, prepared = prepared_data(PREPARED)
    validate_evaluation(evaluation, prepared, ROOT / "features")
    values = pd.read_csv(evaluation / "scores.csv")
    predictions = pd.read_csv(evaluation / "paired_predictions.csv")
    summary = (
        values.groupby(["view", "train_conformers", "test_conformers"], sort=False)
        .agg(
            score_mean=("MA-ST-RAE", "mean"),
            score_sd=("MA-ST-RAE", "std"),
            mae_mean=("macro_MAE", "mean"),
            mae_sd=("macro_MAE", "std"),
        )
        .reset_index()
    )
    summary.to_csv(ROOT / "summary.csv", index=False)
    pairs = []
    for view in VIEWS:
        for count in (2, 10):
            pairs += [(f"{view}-train{count}-test1", f"{view}-train1-test1", "augmentation")]
            pairs += [(f"{view}-train{count}-test{count}", f"{view}-train1-test1", "combined")]
        for count in COUNTS:
            for test_count in (2, 10):
                pairs += [
                    (
                        f"{view}-train{count}-test{test_count}",
                        f"{view}-train{count}-test1",
                        "averaging",
                    )
                ]
    for count in COUNTS:
        for test_count in COUNTS:
            pairs += [
                (
                    f"scalar-train{count}-test{test_count}",
                    f"invariant-train{count}-test{test_count}",
                    "readout",
                )
            ]
    contrasts = []
    for candidate, reference, kind in pairs:
        contrast = paired_seed_bootstrap(predictions, candidate, reference)
        contrast["kind"] = kind
        contrast["delta_mean"] = float(
            values.loc[values.arm == candidate, "MA-ST-RAE"].mean()
            - values.loc[values.arm == reference, "MA-ST-RAE"].mean()
        )
        contrasts.append(contrast)
    atomic_json(ROOT / "paired_comparisons.json", contrasts)
    ledger = json.loads(LEDGER.read_text())
    extension_hours = sum(
        r.get("used_hours", r["reserved_hours"])
        for path, r in ledger.items()
        if path.startswith(str(ROOT.resolve()) + "/")
    )
    budget = {
        "cap_gpu_hours": CAP,
        "cumulative_gpu_hours": charged_hours(),
        "extension_gpu_hours": extension_hours,
    }
    atomic_json(ROOT / "budget_summary.json", budget)
    lines = [
        "# Frozen Atom-JEPA conformer sensitivity",
        "",
        "Fold 0, head seeds 42/43/44. MA-ST-RAE is lower-is-better. "
        "Mean ± sample SD across heads; no seed ensembling.",
        "All features standardized using the original conformer on outer-training rows only. "
        "Same MLP, molecule batches, and 4,800 updates per fit.",
        "",
        "| Features | Training conformers | Evaluation conformers | MA-ST-RAE | Macro MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.view} | {row.train_conformers} | {row.test_conformers} | "
            f"{row.score_mean:.6f} ± {row.score_sd:.6f} | {row.mae_mean:.6f} |"
        )
    baseline = pd.read_csv(PILOT / "summary.csv")
    lines += [
        "",
        "## Fixed baseline references",
        "",
        "Original fold-0 scores, three seeds; same MLP and label splits. "
        "No baseline refitting or model-specific tuning in this follow-up.",
        "",
        "| Baseline | MA-ST-RAE |",
        "|---|---:|",
    ]
    for arm in (
        "chemeleon-mean-none",
        "mol_jepa-cls-none",
        "chemeleon-mean-standard",
        "mol_jepa-cls-standard",
    ):
        row = baseline.loc[baseline.arm == arm].iloc[0]
        lines.append(f"| {arm} | {row['MA-ST-RAE_mean']:.6f} ± {row['MA-ST-RAE_sd']:.6f} |")
    lines += [
        "",
        "## Paired comparisons",
        "",
        "Negative differences favor the candidate. Intervals are exploratory 95% cluster "
        "bootstraps, without multiplicity correction.",
        "",
        "| Contrast | Candidate | Reference | Mean difference | 95% interval |",
        "|---|---|---|---:|---|",
    ]
    for item in contrasts:
        low, high = item["delta_95ci"]
        lines.append(
            f"| {item['kind']} | {item['candidate']} | {item['reference']} | "
            f"{item['delta_mean']:.6f} | [{low:.6f}, {high:.6f}] |"
        )
    geometry = json.loads((ROOT / "geometries" / "manifest.json").read_text())
    diagnostic = pd.read_csv(evaluation / "conformer_sensitivity.csv")
    lines += [
        "",
        "## Evidence and limits",
        "",
        "15 new heads: three standardized scalar single-conformer fits plus twelve augmented "
        "fits. Three existing invariant heads reused with checkpoint and prediction checksums.",
        "One sampled conformer per molecule per epoch; predictions, not embeddings, averaged "
        "at evaluation. Scalers are identical across conformer counts and saved in checkpoints.",
        f"Nine additional geometry seeds for 4,905 molecules: {geometry['nonconverged']} finite "
        f"MMFF nonconvergences retained, {geometry['retried']} random-coordinate retries. "
        "No dropped rows. Generated conformers need not be distinct minima.",
        "Largest single-conformer CPU replay error: "
        f"{diagnostic.single_conformer_replay_max_abs.max():.9g} pIC50 (threshold 1e-4). "
        "Encoder state unchanged during extraction.",
        f"Additional GPU time: {extension_hours:.6f} h. Cumulative original pilot plus "
        f"extension: {budget['cumulative_gpu_hours']:.6f} / 4 h; "
        "failed prior extraction attempts and diagnostics included.",
        "Per-endpoint metrics, individual conformer predictions, and prediction variability "
        "are retained in evaluation/. Baseline CheMeleon and Mol-JEPA results remain in the "
        "original pilot report.",
        "This is a follow-up on an already-inspected fold, not independent confirmation. "
        "No configuration selected for deployment. No fine-tuning, blind export, or challenge "
        "submission. All artifacts remain private.",
        "",
    ]
    validation = Path("runs/atom-jepa-conformer-validation.xml")
    if validation.exists():
        import xml.etree.ElementTree as ET

        suites = list(ET.parse(validation).getroot().iter("testsuite"))
        counts = {
            k: sum(int(s.get(k, 0)) for s in suites)
            for k in ("tests", "failures", "errors", "skipped")
        }
        atomic_json(
            ROOT / "test_evidence.json",
            {
                "junit_sha256": sha256_file(validation),
                "counts": counts,
            },
        )
        lines.append(f"Targeted validation: {counts}; see {validation}.")
    (ROOT / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("all", "geometry", "extract", "fit", "report"), default="all"
    )
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    design = {
        "fold": 0,
        "head_seeds": list(HEAD_SEEDS),
        "conformer_seeds": list(CONFORMER_SEEDS),
        "train_counts": list(COUNTS),
        "test_counts": list(COUNTS),
        "views": list(VIEWS),
        "recipe": RECIPE,
        "scaling": "outer-training original conformer, fixed across arms",
        "sampling": "one uniform conformer per molecule per epoch; RNG seed=head seed+100000",
        "evaluation": "mean predictions over first K conformers",
        "fine_tuning": False,
        "cap_includes_original_pilot": True,
        "cap_gpu_hours": CAP,
        "open_source_code": False,
        "model_report_link": "",
        "submission": False,
    }
    path = ROOT / "design.json"
    if path.exists() and json.loads(path.read_text()) != design:
        raise ValueError("Cannot change an existing experimental design")
    atomic_json(path, design)
    status = {"complete": False, "stage": args.stage, "fine_tuning": False, "submission": False}
    try:
        if args.stage in ("all", "geometry"):
            geometry_path = ROOT / "geometries"
            if not geometry_path.exists():
                generate_extra(geometry_path, PREPARED)
            else:
                manifest = json.loads((geometry_path / "manifest.json").read_text())
                verify_files(geometry_path, manifest)
                if not manifest.get("complete"):
                    raise ValueError("Incomplete conformer geometry")
        if args.stage in ("all", "extract"):
            if not (ROOT / "features").exists():
                config = {
                    **config_for(ROOT / "features", 1.5),
                    "base_cache": str(EXTRACTION),
                    "conformer_cache": str(ROOT / "geometries"),
                }
                config_path = ROOT / "extract.json"
                atomic_json(config_path, config)
                cache_stage(config_path)
            else:
                frame, _ = prepared_data(PREPARED)
                load_conformer_features(ROOT / "features", frame)
        if args.stage in ("all", "fit"):
            frame, prepared = prepared_data(PREPARED)
            features, cache = load_conformer_features(ROOT / "features", frame)
            fit_all()
            evaluate(features, frame, prepared)
        if args.stage in ("all", "fit", "report"):
            report()
            status["complete"] = True
        if charged_hours() > CAP:
            raise RuntimeError("Original four-hour cap exceeded")
    except BaseException as error:
        status.update(complete=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["cumulative_gpu_hours"] = charged_hours()
        atomic_json(ROOT / "status.json", status)


if __name__ == "__main__":
    main()
