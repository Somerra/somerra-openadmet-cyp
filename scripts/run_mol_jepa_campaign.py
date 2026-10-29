"""Sequential original-protocol campaign; stop on any failure and retain evidence."""

import argparse
import json
from pathlib import Path

from somerra_openadmet.download import sha256_file
from somerra_openadmet.representation_training import screen_representations
from somerra_openadmet.representations import atomic_json, cache_representations


def completed_stage(config_path):
    """Reuse only complete, matching, checksum-verified artifacts; never overwrite partial runs."""
    config = json.loads(Path(config_path).read_text())
    output = Path(config["output_dir"])
    if not output.exists():
        return None
    path = output / "run_manifest.json"
    if not path.exists():
        raise ValueError(f"Incomplete output: {output}; preserve it and declare a recovery config")
    record = json.loads(path.read_text())
    if not record.get("complete"):
        raise ValueError(f"Incomplete output: {output}; preserve it and declare a recovery config")
    ignored = {"ledger", "campaign_ledger"}
    expected = {k: v for k, v in config.items() if k not in ignored}
    actual = {k: v for k, v in record["config"].items() if k not in ignored}
    if expected != actual:
        raise ValueError("Completed stage configuration differs")
    hashes = {
        "oof_predictions.csv": record.get("oof_sha256"),
        "manifest.json": record.get("cache_manifest_sha256"),
        "activity_submission.csv": record.get("submission_sha256"),
    }
    hashes.update({f"fold-{k}.pt": v for k, v in record.get("checkpoint_hashes", {}).items()})
    if record.get("completed_folds") is not None and record["completed_folds"] != list(range(5)):
        raise ValueError("Completed training stage requires all five folds")
    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text())
        views = manifest.get("views", {"": manifest})
        for view, values in views.items():
            hashes[str(Path(view) / "rows.csv")] = values["rows_sha256"]
            hashes[str(Path(view) / "embeddings.npy")] = values["embeddings_sha256"]
    if not any(hashes.values()):
        raise ValueError("Completed stage has no artifact checksums")
    for name, checksum in hashes.items():
        if checksum is not None and sha256_file(output / name) != checksum:
            raise ValueError(f"Completed artifact changed: {name}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/regression-mol-jepa-campaign-v2"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--plan", type=Path, help="JSON list of operation/name/config stage objects"
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=args.resume)
    status = {
        "complete": False,
        "completed": [],
        "submission": False,
        "open_source_code": False,
        "model_report_link": "",
    }
    stages = [("cache", "representations-chemeleon-cache")]
    stages += [
        ("train", f"representations-chemeleon-{arm}" + ("" if variant == "reference" else "-h100"))
        for arm in ["frozen", "full"]
        for variant in ["reference", "h100"]
    ]
    stages += [("cache", "representations-mol-jepa-cache")]
    stages += [
        ("train", f"representations-mol-jepa-{arm}-{variant}")
        for arm in ["frozen", "tokens", "full", "lora"]
        for variant in ["reference", "h100"]
    ]
    stages = [(operation, name, Path("configs") / (name + ".json")) for operation, name in stages]
    if args.plan:
        stages = [
            (stage["operation"], stage["name"], Path(stage["config"]))
            for stage in json.loads(args.plan.read_text())
        ]
    configs = {}
    # All launches remain subject to the existing authorized shared ledger.
    try:
        for operation, name, config in stages:
            status["stage"] = name
            atomic_json(output / "status.json", status)
            print(name, flush=True)
            configs[name] = json.loads(config.read_text())
            if operation not in {"cache", "train"}:
                raise ValueError("Stage operation must be cache or train")
            result = completed_stage(config) if args.resume else None
            if result is None:
                result = (
                    cache_representations if operation == "cache" else screen_representations
                )(config)
            if not result["complete"]:
                raise RuntimeError(f"Incomplete stage: {name}")
            status["completed"].append(name)
        status["training_complete"] = True
        from somerra_openadmet.regression_report import compare
        from somerra_openadmet.representation_training import predict_representations

        run_paths = {}
        for config in configs.values():
            if "adaptation" in config:
                key = (
                    config.get("encoder", "chemeleon"),
                    config["adaptation"],
                    "h100" if config.get("h100_select", False) else "reference",
                )
                run_paths[key] = Path(config["output_dir"])
        comparisons = {}
        for variant in ["reference", "h100"]:
            for arm in ["frozen", "full"]:
                candidate = run_paths[("mol_jepa", arm, variant)]
                reference = run_paths[("chemeleon", arm, variant)]
                comparisons[f"{arm}-{variant}"] = compare(
                    candidate,
                    reference="y_pred",
                    reference_run=reference,
                )
        for variant in ["reference", "h100"]:
            for candidate, reference in [("tokens", "frozen"), ("lora", "full")]:
                comparisons[f"{candidate}-vs-{reference}-{variant}"] = compare(
                    run_paths[("mol_jepa", candidate, variant)],
                    reference="y_pred",
                    reference_run=run_paths[("mol_jepa", reference, variant)],
                )
        exports = {}
        for name in status["completed"]:
            if "adaptation" not in configs[name]:
                continue
            config = configs[name]
            export_config = {
                "run_dir": config["output_dir"],
                "blind_csv": "data/raw/cyp-challenge-TEST-BLINDED.csv",
                "output_dir": config["output_dir"] + "-blind",
                "device": "cpu",
                "max_gpu_hours": 2,
            }
            path = output / (name + "-predict.json")
            atomic_json(path, export_config)
            exports[name] = completed_stage(path) if args.resume else None
            if exports[name] is None:
                exports[name] = predict_representations(path)
        report = {
            "comparisons": comparisons,
            "exports": exports,
            "screening_only": True,
            "promotion": False,
            "submission": False,
            "fusion_recommendation": (
                "Defer fusion until standalone results are reviewed; fusion remains issue #15."
            ),
            "reference_audits": "docs/chemeleon-implementation.md",
            "recipe_note": (
                "Reference epochs: frozen 300, adaptation 60; "
                "H100 selection uses inner groups only."
            ),
        }
        atomic_json(output / "comparison-report.json", report)
        status["complete"] = True

    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_json(output / "status.json", status)


if __name__ == "__main__":
    main()
