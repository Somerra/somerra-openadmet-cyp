"""Predeclared matched frozen campaign, with an incremental four-GPU-hour cap.

The existing global ledger remains authoritative. Failed/retried stages keep
their charges. No encoder adaptation or challenge submission is performed.
"""

import argparse
import json
from pathlib import Path

from somerra_openadmet.frozen_benchmark import PROTOCOL, screen
from somerra_openadmet.frozen_pilot_budget import is_pilot_charge, pilot_ledger_path
from somerra_openadmet.representations import atomic_json

ROOT = Path("runs/regression-frozen-representation-pilot-v1")
EXTRACTION_PREFIX = Path("runs/regression-atom-jepa-representations-v1")
EXTRACTION = Path("runs/regression-atom-jepa-representations-v1-retry-2")
CAP = 4.0


def charged_hours():
    records = json.loads(pilot_ledger_path({}).read_text())
    total = 0.0
    for path, record in records.items():
        if (
            is_pilot_charge(path)
            or path.startswith(str(ROOT.resolve()) + "/")
            or path.startswith(str(EXTRACTION_PREFIX.resolve()))
        ):
            total += record.get("used_hours", record["reserved_hours"])
    return total


def matrix():
    variants = [
        (
            "chemeleon",
            "mean",
            "runs/regression-chemeleon-representations-v1",
            "runs/chemeleon_mp.pt",
        ),
        ("mol_jepa", "cls", "runs/regression-mol-jepa-cache-v1", "runs/mol-jepa-model"),
        ("atom_jepa", "invariant", str(EXTRACTION), "runs/atom-jepa-model"),
    ]
    # Primary results finish before sensitivity analyses. No decisions use scores.
    for scaling in ("none", "standard"):
        for seed in (42, 43, 44):
            for encoder, view, cache, checkpoint in variants:
                yield encoder, view, cache, checkpoint, scaling, seed
    for seed in (42, 43, 44):
        yield "atom_jepa", "scalar", str(EXTRACTION), "runs/atom-jepa-model", "none", seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip only verified completed fits")
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=args.resume)
    status_file = ROOT / "status.json"
    status = (
        json.loads(status_file.read_text())
        if args.resume and status_file.exists()
        else {
            "complete": False,
            "protocol": PROTOCOL,
            "gpu_cap": CAP,
            "folds": [0],
            "seeds": [42, 43, 44],
            "completed": [],
            "fine_tuning": False,
            "submission": False,
            "open_source_code": False,
            "model_report_link": "",
        }
    )
    design = {
        "protocol": PROTOCOL,
        "cap_gpu_hours": CAP,
        "matrix": [list(v) for v in matrix()],
        "primary": "invariant Atom-JEPA, CLS Mol-JEPA, mean CheMeleon; no scaling",
        "fixed_conformer": 0,
        "bootstrap": "1000 paired cluster resamples, seed 42",
    }
    design_path = ROOT / "design.json"
    if design_path.exists() and json.loads(design_path.read_text()) != design:
        raise ValueError("Cannot change the completed pilot design on resume")
    atomic_json(design_path, design)
    try:
        for encoder, view, cache, checkpoint, scaling, seed in matrix():
            name = f"{encoder}-{view}-{scaling}-seed{seed}"
            output = ROOT / name
            if args.resume and (output / "run_manifest.json").exists():
                from somerra_openadmet.frozen_resume import completed_head

                completed_head(
                    output,
                    {
                        "encoder": encoder,
                        "representation": view,
                        "cache_dir": cache,
                        "checkpoint": checkpoint,
                        "scaling": scaling,
                        "seed": seed,
                        "prepared_dir": "runs/regression-corpus-v1",
                        "adaptation": "frozen",
                        "selected_folds": [0],
                    },
                )
                if name not in status["completed"]:
                    status["completed"].append(name)
                continue
            allocation = min(0.10, CAP - charged_hours())
            if allocation <= 0:
                raise TimeoutError("Four-hour incremental cap exhausted")
            config = {
                "protocol": PROTOCOL,
                "encoder": encoder,
                "representation": view,
                "cache_dir": cache,
                "checkpoint": checkpoint,
                "scaling": scaling,
                "prepared_dir": "runs/regression-corpus-v1",
                "output_dir": str(output),
                "seed": seed,
                "selected_folds": [0],
                "adaptation": "frozen",
                "device": "cuda",
                "precision": "float32",
                "threads": 8,
                "max_gpu_hours": allocation,
                "ledger": str(pilot_ledger_path({})),
            }
            config_path = ROOT / f"{name}.json"
            atomic_json(config_path, config)
            status.update(stage=name, charged_gpu_hours=charged_hours())
            atomic_json(status_file, status)
            print(name, flush=True)
            result = screen(config_path)
            if not result["pilot_complete"]:
                raise RuntimeError(f"Incomplete fit: {name}")
            status["completed"].append(name)
            status["charged_gpu_hours"] = charged_hours()
            atomic_json(status_file, status)
        from report_frozen_representation_pilot import report

        report(ROOT)
        status.update(complete=True, charged_gpu_hours=charged_hours())
    except BaseException as error:
        status.update(
            complete=False,
            error=f"{type(error).__name__}: {error}",
            charged_gpu_hours=charged_hours(),
        )
        raise
    finally:
        atomic_json(status_file, status)


if __name__ == "__main__":
    main()
