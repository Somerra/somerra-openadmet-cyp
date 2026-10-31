"""Single Workbench fold, four independent objectives, immutable research evaluation.

python -m somerra_openadmet.cyp_research --help
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ACTIVITY_ENDPOINTS
from .cyp_evaluation import regression_metrics
from .data_sources.common import read_json, write_json
from .download import sha256_file

CYPS = [ep.split("_")[0] for ep in ACTIVITY_ENDPOINTS]
AUXILIARY = (
    [f"{c}_log2fc_4.9504951e-05M" for c in CYPS]
    + [f"{c}_pIC50_TDI_condition" for c in CYPS]
    + [f"{c.lower()}_pic50_chembl" for c in [*CYPS, "CYP2C19"]]
    + [f"{c.lower()}_max_response" for c in [*CYPS, "CYP2C19"]]
    + [f"{c.lower()}_pic50_qhts" for c in CYPS]
)
# Freeze the evaluation contract; training code is recorded per trial, not a search constraint.
FIXED_FILES = ["cyp_evaluation.py", "metrics.py", "constants.py"]
RUNTIME_FILES = [
    "cyp_research_worker.py",
    "cyp_research.py",
    "research_tracking.py",
    "data_sources/common.py",
]
DEFAULT_ROOT = Path("runs/cyp-recipes-fold0")


def read(path):
    return read_json(Path(path))


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, value)
    tmp.replace(path)


@contextmanager
def lock(root):
    with (Path(root) / "research.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def fixed_hashes():
    return {name: sha256_file(Path(__file__).parent / name) for name in FIXED_FILES}


def tasks_for(cyp):
    return [ACTIVITY_ENDPOINTS[CYPS.index(cyp)], *AUXILIARY]


def split_data(official, pool, workbench, blind, fold):
    """Inherit molecular membership from mt_aux8; never reuse expanded-table fold IDs."""
    for frame, key in [
        (official, "Molecule_Name"),
        (pool, "Molecule_Name"),
        (workbench, "molecule_name"),
    ]:
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"Missing or duplicate {key}")
    if set(official.Molecule_Name) != set(workbench.molecule_name):
        raise ValueError("Official/Workbench ID mismatch")
    if fold not in set(workbench.fold):
        raise ValueError("Unknown Workbench fold")
    aligned = official.set_index("Molecule_Name").loc[workbench.molecule_name].reset_index()
    aligned["workbench_fold"] = workbench.fold.to_numpy()
    indexed = pool.set_index("Molecule_Name")
    matched = indexed.loc[aligned.Molecule_Name]
    if not np.array_equal(matched.parent.to_numpy(), aligned.parent.to_numpy()):
        raise ValueError("Expanded/official molecular identity mismatch")
    if not np.allclose(
        matched[list(ACTIVITY_ENDPOINTS)],
        aligned[list(ACTIVITY_ENDPOINTS)],
        equal_nan=True,
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError("Expanded/official primary labels mismatch")
    if any(f.parent.isna().any() for f in [official, pool, blind]):
        raise ValueError("Missing molecular parent")
    held = set(aligned.loc[aligned.workbench_fold == fold, "parent"])
    blinded = set(blind.parent)
    if held & blinded:
        raise ValueError("Validation overlaps blinded parents")
    aligned["partition"] = np.where(
        aligned.workbench_fold == fold,
        "validation",
        np.where(
            aligned.parent.isin(held),
            "excluded_validation_alias",
            np.where(aligned.parent.isin(blinded), "excluded_blind", "train"),
        ),
    )
    validation = indexed.loc[aligned.loc[aligned.partition == "validation", "Molecule_Name"]]
    validation = validation.reset_index().copy()
    official_index = official.set_index("Molecule_Name")
    bounds = [ep + s for ep in ACTIVITY_ENDPOINTS for s in ["_conf_low", "_conf_high"]]
    for col in ["SMILES", *bounds]:
        validation[col] = official_index.loc[validation.Molecule_Name, col].to_numpy()
    train = pool.loc[~pool.parent.isin(held | blinded)].copy()
    train["SMILES"] = train.Molecule_Name.map(official_index.SMILES).fillna(train.parent)
    for col in bounds:
        train[col] = train.Molecule_Name.map(official_index[col])
    train["sample_weight"] = 1.0
    validation["sample_weight"] = 1.0
    if set(train.parent) & (set(validation.parent) | blinded):
        raise ValueError("Parent leakage")
    return train, validation, aligned


def prepare(config):
    if set(config["campaigns"]) != set(CYPS):
        raise ValueError("Configure each of the four CYP campaigns")
    for settings in config["campaigns"].values():
        if (
            not np.isfinite(settings["budget_gpu_hours"])
            or settings["budget_gpu_hours"] < 0
            or not np.isfinite(settings["trial_seconds"])
            or settings["trial_seconds"] <= 15
        ):
            raise ValueError("Invalid campaign budget or trial cap")
    root = Path(config["output_dir"]).resolve()
    if root.exists():
        saved = verify(root)
        if saved["config"] != config:
            raise ValueError("Existing campaign config differs; choose a fresh output directory")
        return root
    source, workbench = Path(config["source_dir"]), Path(config["workbench_data"])
    source_manifest, wb_manifest = read(source / "manifest.json"), read(workbench / "manifest.json")
    if source_manifest["tasks"] != [*ACTIVITY_ENDPOINTS, *AUXILIARY]:
        raise ValueError("Expected the frozen 26-target bundle")
    hashes = {}
    for base, manifest, names in [
        (source, source_manifest, ["official.csv", "pool.csv", "test.csv"]),
        (workbench, wb_manifest, ["mt_aux8.csv"]),
    ]:
        for name in names:
            path = base / name
            digest = sha256_file(path)
            if digest != manifest["file_sha256"][name]:
                raise ValueError(f"Changed source: {path}")
            hashes[str(path.resolve())] = digest
        hashes[str((base / "manifest.json").resolve())] = sha256_file(base / "manifest.json")
    official, pool, blind = [pd.read_csv(source / f"{n}.csv") for n in ["official", "pool", "test"]]
    from .data_sources.common import parent_smiles

    # Independently standardize blind SMILES as well as trusting the frozen parent column.
    blind = pd.concat([blind, pd.DataFrame({"parent": blind.SMILES.map(parent_smiles)})])
    train, val, assignments = split_data(
        official, pool, pd.read_csv(workbench / "mt_aux8.csv"), blind, config["fold"]
    )
    for cyp in CYPS:
        values = train[tasks_for(cyp)].to_numpy(float)
        if np.isinf(values).any() or not np.isfinite(np.nanstd(values, axis=0)).all():
            raise ValueError(f"Invalid training labels: {cyp}")
        if (np.nanstd(values, axis=0) <= 0).any():
            raise ValueError(f"Empty or constant training head: {cyp}")
        ep = tasks_for(cyp)[0]
        regression_metrics(val, np.full(len(val), np.nanmean(train[ep])), ep)
    root.mkdir(parents=True)
    for name, frame in [("train", train), ("validation", val), ("assignments", assignments)]:
        frame.to_csv(root / f"{name}.csv", index=False)
    manifest = {
        "protocol": "cyp-recipes-v2",
        "config": config,
        "campaign_id": uuid.uuid4().hex[:12],
        "source_sha256": hashes,
        "fixed_source_sha256": fixed_hashes(),
        "files": {
            n: sha256_file(root / n) for n in ["train.csv", "validation.csv", "assignments.csv"]
        },
        "rows": {"train": len(train), "validation": len(val)},
        "validation_counts": val[list(ACTIVITY_ENDPOINTS)].notna().sum().to_dict(),
        "partition_counts": assignments.partition.value_counts().to_dict(),
        "tasks": {c: tasks_for(c) for c in CYPS},
        "validation_role": "adaptively reused development fold; not an unbiased test",
        "external_overlap": "parent-separated; external scaffolds may overlap validation",
        "open_source_code": False,
        "model_report_link": "",
        "submission_authorized": False,
    }
    write(root / "manifest.json", manifest)
    for cyp in CYPS:
        campaign = root / "campaigns" / cyp
        campaign.mkdir(parents=True)
        write(campaign / "registry.json", {"trials": [], "best": {}})
        write(
            campaign / "campaign.json",
            {
                "cyp": cyp,
                "campaign_id": manifest["campaign_id"] + "-" + cyp,
                **config["campaigns"][cyp],
            },
        )
    return root


def verify(root):
    root = Path(root)
    manifest = read(root / "manifest.json")
    if manifest["fixed_source_sha256"] != fixed_hashes():
        raise ValueError("Evaluation contract changed; prepare a new data revision")
    for name, digest in manifest["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Prepared data changed: {name}")
    return manifest


def leaderboard(registry):
    selected = {c: registry["best"].get(c) for c in CYPS}
    scores = {c: v["metrics"]["ST-RAE"] for c, v in selected.items() if v}
    return {
        "best": selected,
        "completed_objectives": len(scores),
        "MA-ST-RAE": float(np.mean(list(scores.values()))) if len(scores) == 4 else None,
    }


def accept_result(registry, trial, result):
    """A successful full trial can only replace its own CYP's incumbent."""
    if trial["smoke"] or result["status"] != "complete":
        return False
    score = result["metrics"]["ST-RAE"]
    if not np.isfinite(score):
        raise ValueError("Nonfinite trial score")
    old = registry["best"].get(trial["cyp"])
    if old and score >= old["metrics"]["ST-RAE"] - 1e-12:
        return False
    registry["best"][trial["cyp"]] = {
        "trial_id": trial["trial_id"],
        "metrics": result["metrics"],
        "checkpoint": str(Path(trial["output"]) / "best.pt"),
        "best_epoch": result["best_epoch"],
        "strategy_record": str(Path(trial["output"]) / "strategy.json"),
    }
    return True


def run_trial(
    root,
    cyp,
    recipe,
    hypothesis,
    smoke=False,
    mode=None,
    *,
    additional_budget_hours=0,
    expected_assets=None,
):
    if not np.isfinite(additional_budget_hours) or additional_budget_hours < 0:
        raise ValueError("Invalid additional budget")
    root = Path(root).resolve()
    manifest = verify(root)
    campaign = root / "campaigns" / cyp
    with lock(campaign):
        config, registry = manifest["config"], read(campaign / "registry.json")
        campaign_config = read(campaign / "campaign.json")
        if any(campaign_config[k] != v for k, v in config["campaigns"][cyp].items()):
            raise ValueError("Frozen CYP allocation changed")
        # Reserve the full cap up front. Unaccounted interrupted jobs retain their reservation.
        committed = sum(
            t.get("used_seconds", t["reserved_seconds"])
            for t in registry["trials"]
            if not t["smoke"]
        )
        remaining = (
            campaign_config["budget_gpu_hours"] + additional_budget_hours
        ) * 3600 - committed
        seconds = 180 if smoke else min(campaign_config["trial_seconds"], remaining)
        if not smoke and seconds <= 15:
            raise ValueError("Research budget exhausted or launch disabled (budget_gpu_hours=0)")
        trial_id = f"{cyp.lower()}-{'smoke-' if smoke else ''}{uuid.uuid4().hex[:10]}"
        output = campaign / "trials" / trial_id
        output.mkdir(parents=True)
        shutil.copyfile(recipe, output / "recipe.py")
        (output / "runtime").mkdir()
        for name in [*RUNTIME_FILES, *FIXED_FILES]:
            snapshot = output / "runtime" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(__file__).parent / name, snapshot)
        trial = {
            "trial_id": trial_id,
            "cyp": cyp,
            "output": str(output),
            "smoke": smoke,
            "reserved_seconds": seconds,
            "status": "running",
            "hypothesis": hypothesis,
            "recipe_sha256": sha256_file(output / "recipe.py"),
            "runtime_sha256": {
                name: sha256_file(Path(__file__).parent / name) for name in RUNTIME_FILES
            },
        }
        job = {
            **trial,
            "workflow": config.get("workflow", "research"),
            "expected_assets_sha256": expected_assets,
            "additional_budget_hours": additional_budget_hours,
            "root": str(root),
            "campaign_id": campaign_config["campaign_id"],
            "wandb_mode": mode or config["wandb"]["mode"],
            "manifest_sha256": sha256_file(root / "manifest.json"),
        }
        write(output / "job.json", job)
        write(
            output / "strategy.json",
            {
                "cyp": cyp,
                "hypothesis": hypothesis,
                "recipe_sha256": trial["recipe_sha256"],
                "runtime_sha256": trial["runtime_sha256"],
                "data_sha256": manifest["files"],
                "evaluation_sha256": manifest["fixed_source_sha256"],
                "selection_metric": "ST-RAE",
                "direction": "minimize",
                "validation_role": manifest["validation_role"],
                "smoke": smoke,
            },
        )
        registry["trials"].append(trial)
        write(campaign / "registry.json", registry)
        started = time.monotonic()
        try:
            with (output / "worker.log").open("w") as log:
                worker = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "somerra_openadmet.cyp_research_worker",
                        str(output / "job.json"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    code = worker.wait(timeout=seconds + 30)
                except BaseException:
                    worker.terminate()
                    try:
                        worker.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait()
                    raise
            result = read(output / "result.json") if (output / "result.json").exists() else {}
            trial["status"] = result.get("status", "failed") if code == 0 else "failed"
            if trial["status"] == "complete":
                trial["promoted"] = accept_result(registry, trial, result)
            else:
                trial["error"] = result.get("error", f"Worker exit {code}; inspect worker.log")
        except subprocess.TimeoutExpired:
            trial["status"] = "timeout"
        except BaseException:
            trial["status"] = "interrupted"
            raise
        finally:
            trial["used_seconds"] = time.monotonic() - started
            if not (output / "result.json").exists():
                write(
                    output / "result.json",
                    {
                        "status": trial["status"],
                        "strategy_sha256": sha256_file(output / "strategy.json"),
                        "error": trial.get("error", "See worker.log"),
                    },
                )
            write(campaign / "registry.json", registry)
            write(campaign / "leaderboard.json", leaderboard(registry))
        if trial["status"] == "complete":
            from .research_tracking import log_campaign

            log_campaign(
                campaign,
                {**manifest, "campaign_id": campaign_config["campaign_id"]},
                registry,
                job["wandb_mode"],
                all_best=status(root)["best"],
            )
        return trial


def status(root):
    """Read-only aggregate; each CYP retains its own lock, spending, and incumbent."""
    root = Path(root)
    combined = {"best": {}}
    campaigns = {}
    for cyp in CYPS:
        path = root / "campaigns" / cyp
        registry = read(path / "registry.json")
        combined["best"].update(registry["best"])
        config = read(path / "campaign.json")
        used = sum(
            t.get("used_seconds", t["reserved_seconds"])
            for t in registry["trials"]
            if not t["smoke"]
        )
        campaigns[cyp] = {**config, "trials": len(registry["trials"]), "committed_seconds": used}
    return {**leaderboard(combined), "campaigns": campaigns}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, default=Path("configs/cyp-autoresearch.json"))
    p = sub.add_parser("trial")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--cyp", choices=CYPS, required=True)
    p.add_argument("--recipe", type=Path)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    p = sub.add_parser("status")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p = sub.add_parser("note", help="Retain the agent's conclusion and next direction")
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--conclusion", required=True)
    p.add_argument("--next-hypothesis", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(prepare(read(args.config)))
    elif args.command == "status":
        print(json.dumps(status(args.root), indent=2))
    elif args.command == "note":
        # Append rather than overwrite the research history.
        with (args.trial / "notes.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "conclusion": args.conclusion,
                        "next_hypothesis": args.next_hypothesis,
                    }
                )
                + "\n"
            )
    else:
        recipe = args.recipe or Path("research/recipes") / args.cyp / "recipe.py"
        trial = run_trial(args.root, args.cyp, recipe, args.hypothesis, args.smoke, args.wandb_mode)
        print(json.dumps(trial), flush=True)
        if trial["status"] != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
