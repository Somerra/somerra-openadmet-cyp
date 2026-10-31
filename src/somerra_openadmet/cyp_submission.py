"""Build private prediction files from frozen CYP recipes, without strategy search or uploads."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ACTIVITY_ENDPOINTS, EXPECTED_TEST_ROWS, IDENTIFIER_COLUMNS
from .cyp_research import (
    CYPS,
    RUNTIME_FILES,
    lock,
    prepare,
    read,
    regression_metrics,
    run_trial,
    verify,
    write,
)
from .download import sha256_file
from .validation import validate_submission


def checked_trial(path, cyp):
    """Only complete, replay-checked, non-smoke trials can cross the handoff."""
    path = Path(path).resolve()
    job, result = read(path / "job.json"), read(path / "result.json")
    if job["cyp"] != cyp or job["smoke"] or result.get("smoke", True):
        raise ValueError("Wrong CYP or smoke trial is not a submission recipe")
    if result["status"] != "complete" or result.get("reload_max_abs", float("inf")) > 1e-4:
        raise ValueError("Need a complete, replay-checked trial")
    for name, digest in {
        "recipe.py": job["recipe_sha256"],
        "strategy.json": result["strategy_sha256"],
        "best.pt": result["checkpoint_sha256"],
        "validation-predictions.csv": result["validation_predictions_sha256"],
    }.items():
        if sha256_file(path / name) != digest:
            raise ValueError(f"Changed trial artifact: {name}")
    if not np.isfinite(result["metrics"]["ST-RAE"]):
        raise ValueError("Nonfinite trial score")
    for name, digest in read(path / "strategy.json").get("assets_sha256", {}).items():
        if sha256_file(Path(name)) != digest:
            raise ValueError("Declared recipe asset changed")
    for name, digest in job["runtime_sha256"].items():
        if sha256_file(Path(__file__).parent / name) != digest:
            raise ValueError("Trial runtime changed; restore recorded runtime before replay")
    return job, result


def freeze(selections, output, *, budget_gpu_hours=0, fit_seconds=1800):
    """Explicitly select four strategies; never infer them from a mutable incumbent."""
    if set(selections) != set(CYPS):
        raise ValueError("Select exactly one completed recipe for each CYP")
    if (
        not np.isfinite(budget_gpu_hours)
        or budget_gpu_hours < 0
        or not np.isfinite(fit_seconds)
        or fit_seconds <= 15
    ):
        raise ValueError("Invalid build budget or fit cap")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    checked, roots = {}, []
    for cyp in CYPS:
        trial = Path(selections[cyp]).resolve()
        job, result = checked_trial(trial, cyp)
        manifest = verify(job["root"])
        if sha256_file(Path(job["root"]) / "manifest.json") != job["manifest_sha256"]:
            raise ValueError("Research manifest changed")
        roots.append(manifest)
        checked[cyp] = (trial, job, result)
    if any(
        m["source_sha256"] != roots[0]["source_sha256"] or m["files"] != roots[0]["files"]
        for m in roots
    ):
        raise ValueError("Selected recipes must use the same declared data snapshot")
    output.mkdir(parents=True)
    records = {}
    for cyp, (trial, _job, result) in checked.items():
        dest = output / "recipes" / cyp
        dest.mkdir(parents=True)
        for name in ["recipe.py", "strategy.json", "result.json"]:
            shutil.copyfile(trial / name, dest / name)
        records[cyp] = {
            "source_trial": str(trial),
            "files": {
                n: sha256_file(dest / n) for n in ["recipe.py", "strategy.json", "result.json"]
            },
            "development_metrics": result["metrics"],
        }
    plan = {
        "workflow": "frozen-recipe-submission-build-v1",
        "recipes": records,
        "data_config": roots[0]["config"],
        "source_sha256": roots[0]["source_sha256"],
        "runtime_sha256": {
            n: sha256_file(Path(__file__).parent / n) for n in [*RUNTIME_FILES, "cyp_submission.py"]
        },
        "evaluation_sha256": roots[0]["fixed_source_sha256"],
        "folds": list(range(5)),
        "seed_rule": "42 + fold",
        "combination": "equal mean of five folds for each CYP; raw predictions",
        "checkpoint_selection": "primary validation ST-RAE, minimize",
        "budget_gpu_hours": budget_gpu_hours,
        "per_fit_budget_gpu_hours": budget_gpu_hours / 20,
        "fit_seconds": fit_seconds,
        "validation_role": "checkpoint-selected OOF development results; not an independent test",
        "open_source_code": False,
        "model_report_link": "",
        "upload_authorized": False,
    }
    write(output / "plan.json", plan)
    write(
        output / "build-state.json", {"plan_sha256": sha256_file(output / "plan.json"), "fits": {}}
    )
    return plan


def verify_plan(root):
    root = Path(root)
    plan, state = read(root / "plan.json"), read(root / "build-state.json")
    if sha256_file(root / "plan.json") != state["plan_sha256"]:
        raise ValueError("Frozen build plan changed; create a new plan")
    for cyp, record in plan["recipes"].items():
        for name, digest in record["files"].items():
            if sha256_file(root / "recipes" / cyp / name) != digest:
                raise ValueError("Frozen recipe record changed")
    for cyp in CYPS:
        for name, digest in frozen_assets(root, cyp).items():
            if sha256_file(Path(name)) != digest:
                raise ValueError(f"Frozen recipe asset changed: {name}")
    for name, digest in plan["source_sha256"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"Source snapshot changed: {name}")
    for name, digest in {**plan["runtime_sha256"], **plan["evaluation_sha256"]}.items():
        if sha256_file(Path(__file__).parent / name) != digest:
            raise ValueError("Build runtime changed; reproduce with the recorded runtime")
    return plan, state


def frozen_assets(root, cyp):
    return read(Path(root) / "recipes" / cyp / "strategy.json").get("assets_sha256", {})


def checked_fit(root, plan, path, cyp, fold):
    job, result = checked_trial(path, cyp)
    manifest = verify(job["root"])
    if (
        manifest["config"]["fold"] != fold
        or manifest["config"]["seed"] != 42 + fold
        or manifest["source_sha256"] != plan["source_sha256"]
        or job["recipe_sha256"] != plan["recipes"][cyp]["files"]["recipe.py"]
    ):
        raise ValueError("Fit does not match the frozen build plan")
    if read(Path(path) / "strategy.json").get("assets_sha256", {}) != frozen_assets(root, cyp):
        raise ValueError("Fit assets differ from the frozen strategy")
    return job, result


def budget_adjustments(root, plan_sha256):
    path = Path(root) / "budget-adjustments.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    for row in records:
        if (
            row["plan_sha256"] != plan_sha256
            or row["cyp"] not in CYPS
            or row["fold"] not in range(5)
            or not np.isfinite(row["additional_gpu_hours"])
            or row["additional_gpu_hours"] <= 0
        ):
            raise ValueError("Invalid build budget adjustment")
    return records


def allocate(root, cyp, fold, additional_gpu_hours, reason):
    """Record explicitly authorized retry compute without changing recipes or erasing charges."""
    root = Path(root).resolve()
    if (
        cyp not in CYPS
        or fold not in range(5)
        or not reason.strip()
        or not np.isfinite(additional_gpu_hours)
        or additional_gpu_hours <= 0
    ):
        raise ValueError("Specify a CYP, fold, positive extra budget, and reason")
    with lock(root):
        _, state = verify_plan(root)
        budget_adjustments(root, state["plan_sha256"])
        record = {
            "id": uuid.uuid4().hex,
            "time": time.time(),
            "plan_sha256": state["plan_sha256"],
            "cyp": cyp,
            "fold": fold,
            "additional_gpu_hours": additional_gpu_hours,
            "reason": reason,
        }
        with (root / "budget-adjustments.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        return record


def fit(root, mode=None):
    """Replay the frozen recipes across saved folds; never edit or search recipes here."""
    root = Path(root).resolve()
    with lock(root):
        plan, state = verify_plan(root)
        adjustments = budget_adjustments(root, state["plan_sha256"])
        if plan["budget_gpu_hours"] <= 0 and not adjustments:
            raise ValueError("Build launch disabled (budget_gpu_hours=0)")
        for fold in plan["folds"]:
            config = copy.deepcopy(plan["data_config"])
            config.update(
                output_dir=str(root / "data" / f"fold{fold}"),
                fold=fold,
                seed=42 + fold,
                workflow="submission-fit",
            )
            config["campaigns"] = {
                c: {
                    "budget_gpu_hours": plan["per_fit_budget_gpu_hours"],
                    "trial_seconds": plan["fit_seconds"],
                }
                for c in CYPS
            }
            data_root = prepare(config)
            for cyp in CYPS:
                key = f"{cyp}/fold{fold}"
                if key in state["fits"]:
                    checked_fit(root, plan, state["fits"][key], cyp, fold)
                    continue
                # Recover a fit completed before the build-state write was interrupted.
                registry = read(data_root / "campaigns" / cyp / "registry.json")
                completed = [
                    t for t in registry["trials"] if t["status"] == "complete" and not t["smoke"]
                ]
                trial = (
                    completed[0]
                    if completed
                    else run_trial(
                        data_root,
                        cyp,
                        root / "recipes" / cyp / "recipe.py",
                        "Frozen recipe replay for submission; no strategy search",
                        mode=mode,
                        additional_budget_hours=sum(
                            r["additional_gpu_hours"]
                            for r in adjustments
                            if r["cyp"] == cyp and r["fold"] == fold
                        ),
                        expected_assets=frozen_assets(root, cyp),
                    )
                )
                if trial["status"] != "complete":
                    raise RuntimeError(f"Build fit failed: {key}; inspect {trial['output']}")
                checked_fit(root, plan, trial["output"], cyp, fold)
                state["fits"][key] = trial["output"]
                write(root / "build-state.json", state)
        return state


def inference_device(result, requested="auto"):
    import torch

    recorded = result.get("inference_device", "cpu" if result.get("smoke") else "cuda")
    selected = recorded if requested == "auto" else requested
    if selected not in {"cpu", "cuda"}:
        raise ValueError("Inference device must be auto, cpu, or cuda")
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Recorded inference requires CUDA; use a CUDA host or explicitly "
            "request --device cpu for a strict cross-device replay check"
        )
    return torch.device(selected)


def predict_trial(path, frame, device="auto"):
    """Reconstruct model and preprocessing exclusively from a completed frozen fit."""
    import torch

    from .cyp_research_worker import RecipeData, load_recipe

    path = Path(path)
    result = read(path / "result.json")
    device = inference_device(result, device)
    torch.set_float32_matmul_precision(result.get("matmul_precision", "highest"))
    saved = torch.load(path / "best.pt", map_location=device, weights_only=True)
    recipe = load_recipe(path / "recipe.py")
    model = recipe.build_model(len(saved["tasks"]), saved["settings"]).to(device)
    model.load_state_dict(saved["model"])
    adapter = RecipeData(recipe, saved["settings"])
    ds = adapter.dataset(frame, np.full((len(frame), len(saved["tasks"])), np.nan))
    values = adapter.predict(model, ds, device)
    if values.shape != (len(frame), len(saved["tasks"])):
        raise ValueError("Prediction shape mismatch")
    return (values * np.asarray(saved["scale"]) + np.asarray(saved["mean"]))[:, 0]


def build(root, device="auto"):
    """Assemble five-fold OOF evidence and a locally validated 750-row CSV. No upload API."""
    root = Path(root).resolve()
    with lock(root):
        plan, state = verify_plan(root)
        expected = {f"{c}/fold{f}" for c in CYPS for f in plan["folds"]}
        if set(state["fits"]) != expected:
            raise ValueError("All 20 frozen-recipe fits must complete before building predictions")
        test = pd.read_csv(Path(plan["data_config"]["source_dir"]) / "test.csv")
        if len(test) != EXPECTED_TEST_ROWS or test.Molecule_Name.duplicated().any():
            raise ValueError("Expected the 750 unique official blind molecules")
        prediction = test[list(IDENTIFIER_COLUMNS)].copy()
        scores, provenance = {}, {}
        for cyp, endpoint in zip(CYPS, ACTIVITY_ENDPOINTS, strict=True):
            blind, oof = [], []
            for fold in plan["folds"]:
                key = f"{cyp}/fold{fold}"
                trial = Path(state["fits"][key])
                job, result = checked_fit(root, plan, trial, cyp, fold)
                val = pd.read_csv(Path(job["root"]) / "validation.csv")
                values = predict_trial(trial, val, device=device)
                saved = pd.read_csv(trial / "validation-predictions.csv")
                if list(saved.Molecule_Name) != list(val.Molecule_Name) or not np.allclose(
                    values, saved[endpoint], atol=1e-4, rtol=0
                ):
                    raise ValueError("Fit prediction replay differs")
                metrics = regression_metrics(val, values, endpoint)
                if not np.isclose(
                    metrics["ST-RAE"], result["metrics"]["ST-RAE"], atol=1e-5, rtol=0
                ):
                    raise ValueError("Fit replay score differs")
                rows = val[
                    ["Molecule_Name", endpoint, endpoint + "_conf_low", endpoint + "_conf_high"]
                ].copy()
                rows["prediction"], rows["fold"] = values, fold
                oof.append(rows)
                blind.append(predict_trial(trial, test, device=device))
                provenance[key] = {
                    "trial": str(trial),
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "metrics": metrics,
                    "best_epoch": result["best_epoch"],
                    "inference_device": (
                        result.get("inference_device", "cuda") if device == "auto" else device
                    ),
                }
            joined = pd.concat(oof, ignore_index=True)
            official = pd.read_csv(Path(plan["data_config"]["source_dir"]) / "official.csv")
            if joined.Molecule_Name.duplicated().any() or set(joined.Molecule_Name) != set(
                official.Molecule_Name
            ):
                raise ValueError("OOF coverage must match all official rows exactly once")
            scores[cyp] = regression_metrics(joined, joined.prediction.to_numpy(), endpoint)
            joined.to_csv(root / f"{cyp}-oof.csv", index=False)
            prediction[endpoint] = np.mean(blind, axis=0)
        pending = root / "predictions.pending.csv"
        prediction.to_csv(pending, index=False)
        report = validate_submission(pending, test, "activity")
        if not report.valid:
            raise ValueError(report.errors)
        pending.replace(root / "predictions.csv")
        result = {
            "plan_sha256": state["plan_sha256"],
            "fits": provenance,
            "OOF": scores,
            "MA-ST-RAE": float(np.mean([v["ST-RAE"] for v in scores.values()])),
            "validation_role": plan["validation_role"],
            "validation": report.to_dict(),
            "predictions_sha256": sha256_file(root / "predictions.csv"),
            "open_source_code": False,
            "model_report_link": "",
            "submitted": False,
        }
        write(root / "build-result.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="Freeze explicit CYP=TRIAL selections; no training")
    p.add_argument("--select", action="append", required=True, metavar="CYP=TRIAL")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget-gpu-hours", type=float, default=0)
    p.add_argument("--fit-seconds", type=float, default=1800)
    p = sub.add_parser("allocate", help="Record an explicitly authorized retry budget; no launch")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cyp", choices=CYPS, required=True)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--additional-gpu-hours", type=float, required=True)
    p.add_argument("--reason", required=True)
    for command in ["fit", "build", "status"]:
        p = sub.add_parser(command)
        p.add_argument("--root", type=Path, required=True)
        if command == "build":
            p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        if command == "fit":
            p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    args = parser.parse_args(argv)
    if args.command == "plan":
        pairs = [entry.split("=", 1) for entry in args.select]
        if any(len(p) != 2 for p in pairs) or len({p[0] for p in pairs}) != len(pairs):
            parser.error("Use each CYP exactly once as --select CYP=TRIAL")
        result = freeze(
            dict(pairs),
            args.output,
            budget_gpu_hours=args.budget_gpu_hours,
            fit_seconds=args.fit_seconds,
        )
    elif args.command == "allocate":
        result = allocate(args.root, args.cyp, args.fold, args.additional_gpu_hours, args.reason)
    elif args.command == "fit":
        result = fit(args.root, args.wandb_mode)
    elif args.command == "build":
        result = build(args.root, args.device)
    else:
        plan, state = verify_plan(args.root)
        result = {
            "plan": plan,
            "state": state,
            "budget_adjustments": budget_adjustments(args.root, state["plan_sha256"]),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
