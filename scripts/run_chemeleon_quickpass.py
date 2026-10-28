"""Bounded five-outer-fold benchmark; no inner split or score-based recipe selection."""

import argparse
import gc
import json
import math
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS
from somerra_openadmet.download import sha256_file
from somerra_openadmet.regression_report import compare
from somerra_openadmet.representation_training import (
    Inputs,
    profile_batches,
    recipe_for,
    screen_representations,
)
from somerra_openadmet.representations import (
    Run,
    atomic_json,
    cache_representations,
    load_cache,
    prepared_data,
    provenance,
)


def main(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "starting", "pid": os.getpid(), "complete": False}

    def update(stage, **details):
        status.update(stage=stage, updated_utc=datetime.now(UTC).isoformat(), **details)
        atomic_json(output / "status.json", status)
        print(json.dumps(status), flush=True)

    prepared = Path("runs/regression-corpus-v1")
    checkpoint = Path("runs/chemeleon_mp.pt")
    ledger = Path("runs/regression-gpu-budget.json")
    common = {
        "prepared_dir": str(prepared),
        "device": "cuda",
        "threads": 8,
        "memory_fraction": 0.8,
        "free_memory_gib": 10,
        "ledger": str(ledger),
        "seed": 42,
    }
    protocol = {
        "name": "five_outer_folds_fixed_recipe_quickpass_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "inner_selection": False,
        "outer_early_stopping": False,
        "folds": [0, 1, 2, 3, 4],
        "targets": list(ACTIVITY_ENDPOINTS),
        "max_total_gpu_hours": 1.0,
        "stage_caps": {"cache": 0.10, "profiling": 0.10, "frozen": 0.15, "full": 0.65},
        "batch_rule": "smallest within 5% of peak measured throughput",
        "profiling_labels": "synthetic zeros; all probe weights discarded",
        "lr_rule": "reference LR * max(1, min(sqrt(batch/reference_batch), 4))",
        "epochs": {"frozen": 300, "full": 60},
        "warmup_fraction": 0.05,
        "model_precision": "BF16 if finite-gradient/parity smoke passes; otherwise float32",
        "evaluation_precision": "float32; CPU checkpoint replay <= 1e-4 pIC50",
        "future_mol_jepa": (
            "same folds, targets, fixed-recipe outer-only procedure; method-specific reference LR"
        ),
        "submission": False,
        "open_source_code": False,
        "model_report_link": "",
        **provenance(),
    }
    # Retain the exact pre-run source and environment evidence privately.
    snapshot = output / "source"
    shutil.copytree(
        Path("src/somerra_openadmet"), snapshot, ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy2(__file__, output / Path(__file__).name)
    env_records = Path(sys.prefix) / "conda-meta"
    if env_records.is_dir():
        atomic_json(
            output / "conda-packages.json",
            [json.loads(p.read_text()) for p in sorted(env_records.glob("*.json"))],
        )
    atomic_json(output / "protocol.json", protocol)
    shutil.copy2(ledger, output / "ledger-before.json")
    try:
        update("cache")
        cache_config = {
            **common,
            "blind_csv": "data/raw/cyp-challenge-TEST-BLINDED.csv",
            "checkpoint": str(checkpoint),
            "output_dir": str(output / "cache"),
            "batch_size": "auto",
            "precision": "float32",
            "max_gpu_hours": 0.10,
        }
        atomic_json(output / "cache-config.json", cache_config)
        cache_representations(output / "cache-config.json")
        update("profiling")
        frame, manifest = prepared_data(prepared)
        features, _ = load_cache(output / "cache", frame)
        recipes = {}
        with Run({**common, "output_dir": str(output / "profiling"), "max_gpu_hours": 0.10}) as run:
            # Profiling consumes no real activity labels, even from the training folds.
            dummy_targets = np.zeros((len(frame), len(ACTIVITY_ENDPOINTS)), dtype=np.float32)
            for mode in ("frozen", "full"):
                config = {
                    **common,
                    "adaptation": mode,
                    "checkpoint": str(checkpoint),
                    "precision": "bf16",
                }
                reference = recipe_for(mode, {})
                inputs = Inputs(frame, mode, run.device, features if mode == "frozen" else None)
                batch = profile_batches(
                    config,
                    reference,
                    inputs,
                    dummy_targets,
                    np.arange(len(frame)),
                    frame,
                    run,
                    0 if mode == "frozen" else 1,
                )
                scale = max(1.0, min(math.sqrt(batch / reference["batch_size"]), 4.0))
                recipes[mode] = {
                    **reference,
                    "batch_size": batch,
                    "learning_rate": reference["learning_rate"] * scale,
                    "warmup_fraction": 0.05,
                    "precision": "bf16",
                }
                del inputs
                gc.collect()
            run.record.update(complete=True, recipes=recipes, real_labels_used=False)
        # Freeze both recipes before evaluating any outer holdout.
        atomic_json(
            output / "recipes.json",
            {"recipes": recipes, "training_sha256": manifest["training_sha256"]},
        )
        del features
        for mode in ("frozen", "full"):
            update(mode, recipes=recipes)
            recipe = recipes[mode]
            config = {
                **common,
                **{
                    k: recipe[k]
                    for k in (
                        "epochs",
                        "batch_size",
                        "learning_rate",
                        "weight_decay",
                        "warmup_fraction",
                        "precision",
                    )
                },
                "checkpoint": str(checkpoint),
                "cache_dir": str(output / "cache"),
                "adaptation": mode,
                "h100_select": False,
                "output_dir": str(output / mode),
                "max_gpu_hours": protocol["stage_caps"][mode],
            }
            atomic_json(output / f"{mode}-config.json", config)
            screen_representations(output / f"{mode}-config.json")
        update("report")
        comparison = compare(output / "full", "y_pred", output / "frozen")
        incumbent = Path("runs/regression-nested-incumbent-v1")
        for mode in ("frozen", "full"):
            compare(
                output / mode,
                "nested_ensemble",
                incumbent,
                output / f"{mode}-versus-incumbent.json",
            )
        metrics = {
            mode: json.loads((output / mode / "metrics.json").read_text())
            for mode in ("frozen", "full")
        }
        ledger_data = json.loads(ledger.read_text())
        hours = sum(
            record.get("used_hours", record["reserved_hours"])
            for path, record in ledger_data.items()
            if Path(path).is_relative_to(output.resolve())
        )
        summary = {
            "metrics": metrics,
            "comparison": comparison,
            "charged_gpu_hours": hours,
            "protocol_sha256": sha256_file(output / "protocol.json"),
        }
        atomic_json(output / "summary.json", summary)
        lines = [
            "# CheMeleon outer-fold quick pass",
            "",
            "Fixed recipes; five outer folds; no inner selection or early stopping.",
            "",
            "| Endpoint | Frozen MLP ST-RAE | Full tuning ST-RAE |",
            "|---|---:|---:|",
        ]
        for endpoint in ACTIVITY_ENDPOINTS:
            lines.append(
                f"| {endpoint.split('_')[0]} | "
                f"{metrics['frozen']['endpoints'][endpoint]['ST-RAE']:.6f} | "
                f"{metrics['full']['endpoints'][endpoint]['ST-RAE']:.6f} |"
            )
        lines += [
            f"| Macro | {metrics['frozen']['MA-ST-RAE']:.6f} | "
            f"{metrics['full']['MA-ST-RAE']:.6f} |",
            "",
            f"Charged GPU time: {hours:.4f} hours. "
            f"Full-minus-frozen paired 95% interval: {comparison['delta_95ci']}.",
            "",
            "Screening evidence only. No promotion, blind export, or challenge upload. "
            "Mol-JEPA should use the same saved protocol and folds.",
        ]
        (output / "report.md").write_text("\n".join(lines) + "\n")
        update("complete", status="finished", complete=True, summary=summary)
    except BaseException as exc:
        update(
            status.get("stage", "starting"), status="failed", error=f"{type(exc).__name__}: {exc}"
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args().output_dir)
