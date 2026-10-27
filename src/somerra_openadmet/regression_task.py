"""One-candidate autoresearch on a frozen regression task and a persistent incumbent."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .campaign_budget import CampaignLedger
from .data_sources.common import write_json
from .download import sha256_file
from .hillclimb import qualifies
from .hillclimb_comparison import compare_runs
from .model_compatibility import (
    IMPLEMENTATION_FILES,
    check_compatibility,
    fitted_identity,
    require_compatible,
)
from .model_recipe import resolve_model
from .reference_member_data import load_frames

ROOT = Path(__file__).resolve().parents[2]
FIXED_FILES = [
    "autoresearch.py",
    "src/somerra_openadmet/regression_task.py",
    "src/somerra_openadmet/model_compatibility.py",
    "src/somerra_openadmet/reference_member.py",
    "src/somerra_openadmet/reference_member_data.py",
    "src/somerra_openadmet/hillclimb.py",
    "src/somerra_openadmet/hillclimb_comparison.py",
    "src/somerra_openadmet/regression_report.py",
    "src/somerra_openadmet/bootstrap_activity.py",
    "src/somerra_openadmet/metrics.py",
    "src/somerra_openadmet/constants.py",
    "src/somerra_openadmet/splits.py",
    "src/somerra_openadmet/features.py",
    "src/somerra_openadmet/campaign_budget.py",
    "src/somerra_openadmet/data_sources/common.py",
    "src/somerra_openadmet/neural.py",
    "src/somerra_openadmet/data.py",
    "src/somerra_openadmet/molecular_similarity.py",
    "src/somerra_openadmet/validation.py",
]
MODEL_FILES = ["src/somerra_openadmet/hillclimb_model.py", "src/somerra_openadmet/model_recipe.py"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hashes(paths):
    return {str(path): sha256_file(ROOT / path) for path in paths}


def runtime():
    return {
        name: importlib.metadata.version(name)
        for name in ["torch", "chemprop", "lightning", "numpy", "pandas", "rdkit", "scikit-learn"]
    }


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    write_json(temp, value)
    temp.replace(path)


@contextmanager
def registry(task_path):
    path = Path(task_path).resolve().parent / "registry.json"
    with path.with_suffix(".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state = json.loads(path.read_text())
        yield state
        atomic_json(path, state)


def read_registry(task_path, task):
    state = json.loads((Path(task_path).resolve().parent / "registry.json").read_text())
    if state["task_id"] != task["task_id"]:
        raise ValueError("Registry belongs to another task")
    return state


def read_task(path, *, check_runtime=False):
    task = json.loads(Path(path).read_text())
    contract = {k: v for k, v in task.items() if k != "task_id"}
    if task["task_id"] != digest(contract) or task["fixed_source_sha256"] != file_hashes(
        FIXED_FILES
    ):
        raise ValueError("Frozen evaluation changed; initialize a new task explicitly")
    if sha256_file(Path(task["bundle"]) / "manifest.json") != task["bundle_sha256"]:
        raise ValueError("Frozen data bundle changed")
    load_frames(Path(task["bundle"]))
    if check_runtime and runtime() != task["runtime"]:
        raise ValueError("Task runtime changed; use the recorded environment")
    return task


def checked_run(run, *, final=False):
    record = json.loads((Path(run) / "run_manifest.json").read_text())
    if (
        not record.get("complete")
        or record.get("smoke")
        or record.get("final") != final
        or record["seed"] != 42
        or record["batch_size"] != 128
        or record["scheduler_horizon"] != 200
        or record["completed_epochs"] != record["epoch_limit"]
        or (not final and record["epoch_limit"] != 100)
    ):
        raise ValueError("Run does not satisfy the frozen completion contract")
    return record


def member(run, settings):
    run = Path(run).resolve()
    record = checked_run(run)
    settings = resolve_model(settings)
    recorded = record.get(
        "model_config",
        resolve_model(
            {"loss_kind": record["loss_kind"], "auxiliary_weight": record["auxiliary_weight"]}
        ),
    )
    if resolve_model(recorded) != settings:
        raise ValueError("Recipe does not match the fitted member")
    return {
        "run": str(run),
        "model_config": settings,
        "score": record["internal_test_metrics"]["MA-ST-RAE"],
        "best_epoch": record["best_epoch"],
        "model_source_sha256": fitted_identity(record),
        "artifacts": {
            name: sha256_file(run / name)
            for name in ["run_manifest.json", "best.ckpt", "internal-test.csv"]
        },
    }


def validate_member(entry):
    if any(
        sha256_file(Path(entry["run"]) / name) != value
        for name, value in entry["artifacts"].items()
    ):
        raise ValueError("Incumbent artifacts changed")

    record = checked_run(entry["run"])
    if entry["model_source_sha256"] != fitted_identity(record):
        raise ValueError("Incumbent fitted source identity differs from its manifest")


def initialize(bundle, reference, task_dir, campaign_ledger, *, check_model_compatibility=False):
    bundle, reference, task_dir = map(lambda p: Path(p).resolve(), (bundle, reference, task_dir))
    CampaignLedger(campaign_ledger).snapshot()
    checked_run(reference)
    compare_runs(bundle, reference, reference)  # Explicit adoption verifies the recorded score.
    ref = json.loads((reference / "run_manifest.json").read_text())
    settings = ref.get(
        "model_config",
        resolve_model({"loss_kind": ref["loss_kind"], "auxiliary_weight": ref["auxiliary_weight"]}),
    )
    initial = member(reference, settings)
    current = file_hashes(IMPLEMENTATION_FILES)
    environment = runtime()
    if initial["model_source_sha256"] != current and check_model_compatibility:
        initial["implementation_compatibility"] = check_compatibility(
            initial, bundle, current, environment
        )
        # Reject source/artifact changes during the explicit check.
        validate_member(initial)
        if current != file_hashes(IMPLEMENTATION_FILES) or environment != runtime():
            raise ValueError("Implementation/runtime changed during compatibility check")
    require_compatible(initial, bundle, current, environment)
    task = {
        "version": 1,
        "bundle": str(bundle),
        "bundle_sha256": sha256_file(bundle / "manifest.json"),
        "campaign_ledger": str(Path(campaign_ledger).resolve()),
        "fixed_source_sha256": file_hashes(FIXED_FILES),
        "runtime": runtime(),
        "seed": 42,
        "epochs": 100,
        "score": "internal-test point MA-ST-RAE",
        "direction": "minimize",
        "checkpoint_selection": "validation point MA-ST-RAE",
        "promotion": {
            "relative_gain": 0.03,
            "max_endpoint_regression": 0.02,
            "paired_upper_delta": 0,
        },
        "internal_test_role": "development set; repeated selection is adaptive",
        "adopted_reference": initial,
    }
    task["task_id"] = digest(task)
    task_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(task_dir / "task.json", task)
    atomic_json(
        task_dir / "registry.json", {"task_id": task["task_id"], "incumbent": initial, "trials": []}
    )
    return task


def recipe_from_file(path):
    return resolve_model(json.loads(Path(path).read_text()))


def editable_recipe():
    import train

    return resolve_model({**json.loads(train.MODEL_RECIPE.read_text()), **train.MODEL_OVERRIDES})


def plan(task_path, settings):
    task = read_task(task_path)
    incumbent = read_registry(task_path, task)["incumbent"]
    validate_member(incumbent)
    return {
        "task_id": task["task_id"],
        "model_config": resolve_model(settings),
        "model_source_sha256": file_hashes(MODEL_FILES),
        "incumbent": incumbent,
        "seed": 42,
        "epochs": 100,
        "score": task["score"],
        "promotion": task["promotion"],
        "final_fit": "separate prepare command",
        "submission": "never automatic",
    }


def train_supervised(
    task_path, task, output, settings, hours, *, final=False, epochs=100, model_hashes=None
):
    """One full-process allowance, including one possible recovery; no reference refit."""
    if not 0 < hours <= 2:
        raise ValueError("Trial allocation must be in (0, 2] hours")
    ledger = CampaignLedger(task["campaign_ledger"])
    frozen_model = model_hashes if model_hashes is not None else file_hashes(MODEL_FILES)
    remaining = hours
    previous = None
    for attempt in range(2):
        read_task(task_path, check_runtime=True)
        if frozen_model != file_hashes(MODEL_FILES):
            raise ValueError("Model implementation changed during trial")
        dest = output / ("fit" if attempt == 0 else "recovery")
        ledger.reserve_many([(dest, remaining)])
        command = [
            sys.executable,
            str(ROOT / "autoresearch.py"),
            "_train",
            "--task",
            str(task_path),
            "--model-config",
            str(output / "model.json"),
            "--output-dir",
            str(dest),
            "--hours",
            str(remaining),
            "--epochs",
            str(epochs),
        ]
        if final:
            command.append("--final")
        if previous is not None:
            command += ["--resume-from", str(previous)]
        start, process = time.monotonic(), None
        try:
            with (output / (dest.name + ".log")).open("w") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                    start_new_session=True,
                )
                while process.poll() is None:
                    elapsed = time.monotonic() - start
                    curve = dest / "curve.jsonl"
                    stale = time.time() - curve.stat().st_mtime if curve.exists() else elapsed
                    if elapsed >= remaining * 3600 or stale > 600:
                        os.killpg(process.pid, signal.SIGKILL)
                        break
                    time.sleep(min(5, max(0.01, remaining * 3600 - elapsed)))
                code = process.wait()
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            spent = (time.monotonic() - start) / 3600 if process is not None else 0.0
            ledger.settle(dest, spent, "exited" if process is not None else "not_started")
        remaining -= spent
        read_task(task_path, check_runtime=True)
        if frozen_model != file_hashes(MODEL_FILES):
            raise ValueError("Model implementation changed during trial")
        if code == 0:
            record = checked_run(dest, final=final)
            if record["model_config"] != settings or record["epoch_limit"] != epochs:
                raise ValueError("Worker output differs from planned recipe/epochs")
            return dest
        recovery = dest / "recovery.json"
        if not recovery.exists() or remaining <= 0.05:
            break
        snapshot = json.loads(recovery.read_text())
        if snapshot["record"]["completed_epochs"] >= epochs:
            break
        previous = dest
    raise RuntimeError("Candidate failed; logs and any checkpoint retained")


def finish_trial(task_path, task, output, record, fitted, *, promote):
    """Compare against the latest incumbent under lock, including concurrent completions."""
    with registry(task_path) as state:
        if state["task_id"] != task["task_id"]:
            raise ValueError("Registry belongs to another task")
        incumbent = state["incumbent"]
        validate_member(incumbent)
        comparison = compare_runs(task["bundle"], incumbent["run"], fitted)
        candidate = checked_run(fitted)
        candidate["paired_comparison"] = comparison
        reference = checked_run(incumbent["run"])
        keep = promote and qualifies(reference, candidate)
        record.update(
            status="complete",
            decision="keep" if keep else "discard" if promote else "reproduced",
            run=str(fitted),
            compared_to=incumbent["run"],
            comparison=comparison,
            score=candidate["internal_test_metrics"]["MA-ST-RAE"],
        )
        atomic_json(output / "trial.json", record)
        state["trials"].append({"output": str(output), **record})
        if keep:
            state["incumbent"] = member(fitted, record["model_config"])
    return record


def run_trial(task_path, output, settings, hours, hypothesis="", *, promote=True):
    task_path, output = Path(task_path).resolve(), Path(output).resolve()
    task = read_task(task_path, check_runtime=True)
    preview = plan(task_path, settings)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "model.json", preview["model_config"])
    for path in MODEL_FILES:
        (output / Path(path).name).write_bytes((ROOT / path).read_bytes())
    record = {
        **preview,
        "status": "running",
        "decision": None,
        "hypothesis": hypothesis,
        "workflow": "autoresearch" if promote else "reproduction",
        "hours": hours,
        "model_config_sha256": sha256_file(output / "model.json"),
    }
    atomic_json(output / "trial.json", record)
    try:
        fitted = train_supervised(
            task_path,
            task,
            output,
            preview["model_config"],
            hours,
            model_hashes=preview["model_source_sha256"],
        )
        if sha256_file(output / "model.json") != record["model_config_sha256"]:
            raise ValueError("Recipe snapshot changed during trial")
        return finish_trial(task_path, task, output, record, fitted, promote=promote)
    except BaseException as exc:
        record.update(status="failed", decision="crash", error=repr(exc))
        atomic_json(output / "trial.json", record)
        with registry(task_path) as state:
            state["trials"].append({"output": str(output), **record})
        raise


def prepare(task_path, output, raw, hours):
    task_path, output = Path(task_path).resolve(), Path(output).resolve()
    task = read_task(task_path, check_runtime=True)
    selected = read_registry(task_path, task)["incumbent"]
    validate_member(selected)
    require_compatible(selected, task["bundle"], file_hashes(IMPLEMENTATION_FILES), runtime())
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "model.json", selected["model_config"])
    record = {"task_id": task["task_id"], "selected": selected, "status": "running"}
    atomic_json(output / "preparation.json", record)
    try:
        fitted = train_supervised(
            task_path,
            task,
            output,
            selected["model_config"],
            hours,
            final=True,
            epochs=selected["best_epoch"],
        )
        from .reference_member import export

        export(fitted, Path(task["bundle"]), raw, output / "predictions")
        record.update(status="prepared_not_submitted", run=str(fitted))
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        atomic_json(output / "preparation.json", record)
    return record


def main(argv=None, default_command=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Adopt a completed reference and freeze a task")
    for name in ["bundle", "reference", "task-dir", "campaign-ledger"]:
        init.add_argument("--" + name, type=Path, required=True)
    init.add_argument(
        "--check-model-compatibility",
        action="store_true",
        help="Explicitly verify saved checkpoint predictions before adopting changed code",
    )
    for name in ["plan", "run", "reproduce", "status", "prepare", "_train"]:
        command = sub.add_parser(name)
        command.add_argument("--task", type=Path, required=True)
        if name in {"plan", "run", "_train"}:
            command.add_argument("--model-config", type=Path)
        if name in {"run", "reproduce", "prepare", "_train"}:
            command.add_argument("--output-dir", type=Path, required=True)
            command.add_argument("--hours", type=float, default=1.0)
        if name == "run":
            command.add_argument("--hypothesis", default="")
        if name == "reproduce":
            command.add_argument("--trial", type=Path, required=True)
            command.add_argument("--preview", action="store_true")
        if name == "prepare":
            command.add_argument("--raw", type=Path, required=True)
        if name == "_train":
            command.add_argument("--epochs", type=int, required=True)
            command.add_argument("--final", action="store_true")
            command.add_argument("--resume-from", type=Path)
    argv = sys.argv[1:] if argv is None else argv
    if default_command and (not argv or argv[0].startswith("--")):
        argv = [default_command, *argv]
    args = parser.parse_args(argv)
    if args.command == "init":
        result = initialize(
            args.bundle,
            args.reference,
            args.task_dir,
            args.campaign_ledger,
            check_model_compatibility=args.check_model_compatibility,
        )
    elif args.command == "status":
        task = read_task(args.task)
        result = {
            **read_registry(args.task, task),
            "budget": CampaignLedger(task["campaign_ledger"]).snapshot(),
        }
    elif args.command == "prepare":
        result = prepare(args.task, args.output_dir, args.raw, args.hours)
    elif args.command == "_train":
        from .reference_member import train as fit

        task = read_task(args.task, check_runtime=True)
        if args.model_config is None or (not args.final and args.epochs != 100):
            raise ValueError("Worker must use a saved recipe and the task epoch cap")
        result = fit(
            Path(task["bundle"]),
            args.output_dir,
            epochs=args.epochs,
            hours=args.hours,
            final=args.final,
            resume_from=args.resume_from,
            model_config=recipe_from_file(args.model_config),
            campaign_ledger=task["campaign_ledger"],
        )
    else:
        if args.command == "reproduce":
            saved = json.loads((args.trial / "trial.json").read_text())
            task = read_task(args.task)
            if saved["task_id"] != task["task_id"] or saved["model_source_sha256"] != file_hashes(
                MODEL_FILES
            ):
                raise ValueError("Reproduction requires the original task and model implementation")
            if sha256_file(args.trial / "model.json") != saved["model_config_sha256"]:
                raise ValueError("Saved recipe changed")
            settings = recipe_from_file(args.trial / "model.json")
        else:
            settings = (
                recipe_from_file(args.model_config) if args.model_config else editable_recipe()
            )
        if args.command == "plan" or (args.command == "reproduce" and args.preview):
            result = plan(args.task, settings)
        else:
            result = run_trial(
                args.task,
                args.output_dir,
                settings,
                args.hours,
                getattr(args, "hypothesis", ""),
                promote=args.command == "run",
            )
    print(json.dumps(result, indent=2, allow_nan=False))
    return result
