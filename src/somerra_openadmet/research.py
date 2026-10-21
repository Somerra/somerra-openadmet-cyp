"""Shared trial records for autoresearch and saved-recipe reproduction."""

import argparse
import json
import time
from pathlib import Path

from .configuration import compose_config
from .download import sha256_file
from .evaluation import evaluate_recipe


def main(*, backend, model_recipe, overrides, recipe_source, argv=None, default_command=None):
    parser = argparse.ArgumentParser(
        description="Preview a research trial, run it, or reproduce a saved recipe."
    )
    parser.add_argument(
        "command", choices=("plan", "run", "reproduce"), nargs="?", default=default_command
    )
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--model-config", type=Path, help="Saved recipe for reproduce only")
    parser.add_argument("--backend", choices=("cpu", "neural"), help="Reproduction backend")
    parser.add_argument(
        "--preview", action="store_true", help="Preview reproduction without training"
    )
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("choose plan, run, or reproduce")
    if args.command == "reproduce":
        if args.model_config is None or args.backend is None:
            parser.error("reproduce requires --model-config and --backend")
        backend, model_recipe = args.backend, args.model_config
        overrides, recipe_source = {}, args.model_config
    elif args.model_config is not None or args.backend is not None or args.preview:
        parser.error("--model-config, --backend and --preview are for reproduce only")
    if backend not in {"cpu", "neural"}:
        raise ValueError("BACKEND must be cpu or neural")
    model = {**json.loads(model_recipe.read_text()), **overrides}
    if "selected_folds" in model:
        raise ValueError("Research trials require all folds; recovery uses the separate CLI")
    catalog_text = args.data_config.read_text()
    config, selection = compose_config(model, json.loads(catalog_text), args.output_dir)
    if backend == "cpu" and config.get("auxiliary_dirs"):
        raise ValueError("CPU evaluation supports official-direct only; use a neural recipe")
    inputs = args.output_dir.with_name(args.output_dir.name + "-inputs")
    for path in (args.output_dir, inputs):
        if path.exists():
            raise FileExistsError(path)
    plan = {
        "workflow": "reproduction" if args.command == "reproduce" else "autoresearch",
        "backend": backend,
        "hypothesis": args.hypothesis,
        "config": config,
        **selection,
        "recipe_sha256": sha256_file(recipe_source),
        "data_catalog_sha256": sha256_file(args.data_config),
        "inputs_dir": str(inputs),
        "external_data": "deferred; not supported in this PR (issues #3 and #4)",
        "validation": "configuration preview only; backend validates datasets at run time",
    }
    if args.command == "plan" or args.preview:
        print(json.dumps(plan, indent=2))
        return plan
    inputs.mkdir(parents=True, exist_ok=False)
    recipe, data = inputs / "model.json", inputs / "data.json"
    recipe.write_text(json.dumps(model, indent=2) + "\n")
    data.write_text(catalog_text)
    source_name = "recipe.json" if args.command == "reproduce" else "train.py"
    (inputs / source_name).write_text(recipe_source.read_text())
    record = {**plan, "status": "running", "elapsed_seconds": None}
    status_file = inputs / "trial.json"

    def save():
        status_file.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    start = time.monotonic()
    save()
    try:
        _, score = evaluate_recipe(
            backend=backend, model_config=recipe, data_config=data, output_dir=args.output_dir
        )
        record.update(status="completed", score=score, decision="pending review")
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        save()
    print(json.dumps(record, indent=2))
    return record
