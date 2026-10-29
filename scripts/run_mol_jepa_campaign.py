"""Sequential original-protocol campaign; stop on any failure and retain evidence."""

import json
from pathlib import Path

from somerra_openadmet.representation_training import screen_representations
from somerra_openadmet.representations import atomic_json, cache_representations


def main():
    output = Path("runs/regression-mol-jepa-campaign-v2")
    output.mkdir(exist_ok=False)
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
    # Fixed allocations total 22 hours for Mol-JEPA, leaving 2 hours for investigation.
    try:
        for operation, name in stages:
            status["stage"] = name
            atomic_json(output / "status.json", status)
            print(name, flush=True)
            config = Path("configs") / (name + ".json")
            result = (cache_representations if operation == "cache" else screen_representations)(
                config
            )
            if not result["complete"]:
                raise RuntimeError(f"Incomplete stage: {name}")
            status["completed"].append(name)
        status["training_complete"] = True
        from somerra_openadmet.regression_report import compare
        from somerra_openadmet.representation_training import predict_representations

        comparisons = {}
        for variant in ["reference", "h100"]:
            for arm in ["frozen", "full"]:
                candidate = Path(f"runs/regression-mol-jepa-{arm}-{variant}-v1")
                reference = Path(f"runs/regression-chemeleon-{arm}-{variant}-v1")
                comparisons[f"{arm}-{variant}"] = compare(
                    candidate,
                    reference="y_pred",
                    reference_run=reference,
                    output_path=output / f"comparison-{arm}-{variant}.json",
                )
        exports = {}
        for name in status["completed"]:
            if name.endswith("cache"):
                continue
            config = json.loads((Path("configs") / (name + ".json")).read_text())
            export_config = {
                "run_dir": config["output_dir"],
                "blind_csv": "data/raw/cyp-challenge-TEST-BLINDED.csv",
                "output_dir": config["output_dir"] + "-blind",
                "device": "cpu",
                "max_gpu_hours": 2,
            }
            path = output / (name + "-predict.json")
            atomic_json(path, export_config)
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
        # Reports and export are separate acceptance checks, not implied by training completion.
    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_json(output / "status.json", status)


if __name__ == "__main__":
    main()
