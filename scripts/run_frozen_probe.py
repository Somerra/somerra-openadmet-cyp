"""Run the predeclared v2 probe and CPU controls without modifying v1 artifacts."""

import argparse
import json
from pathlib import Path

import pandas as pd

from somerra_openadmet.frozen_pilot_budget import CAP, is_pilot_charge, pilot_ledger_path
from somerra_openadmet.frozen_probe import (
    PROTOCOL,
    RECIPE,
    completed,
    implementation_hashes,
    screen,
    split_indices,
    split_record,
)
from somerra_openadmet.representations import atomic_json, prepared_data

VARIANTS = [
    ("chemeleon", "mean", "regression-chemeleon-representations-v1", "chemeleon_mp.pt"),
    ("mol_jepa", "cls", "regression-mol-jepa-cache-v1", "mol-jepa-model"),
    ("atom_jepa", "scalar", "regression-atom-jepa-representations-v1-retry-2", "atom-jepa-model"),
    (
        "atom_jepa",
        "invariant",
        "regression-atom-jepa-representations-v1-retry-2",
        "atom-jepa-model",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("runs"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    data = args.data_root.resolve()
    output = data / "regression-frozen-representation-pilot-v1" / "regularized-probe-v2"
    history = pilot_ledger_path({})
    prepared = data / "regression-corpus-v1"
    frame, _ = prepared_data(prepared)
    design = {
        "protocol": PROTOCOL,
        "implementation_sha256": implementation_hashes(),
        "recipe": RECIPE,
        "variants": [list(v) for v in VARIANTS],
        "seeds": [42, 43, 44],
        "split": split_record(frame, split_indices(frame)),
        "one_evaluation_prediction": True,
        "fine_tuning": False,
    }
    if output.exists():
        if not args.resume or json.loads((output / "design.json").read_text()) != design:
            raise ValueError("Existing v2 study requires unchanged design and --resume")
    else:
        output.mkdir()
        atomic_json(output / "design.json", design)
    records = []
    # All MLP arms use the same ceiling; neither architecture nor recipe is selected
    # after viewing outer scores. Controls are fit on CPU after all MLP arms finish.
    for learner in ("mlp", "controls"):
        for encoder, view, cache, checkpoint in VARIANTS:
            for seed in (42, 43, 44) if learner == "mlp" else (42,):
                name = f"{encoder}-{view}-{learner}-seed{seed}"
                directory = output / name
                config = {
                    "protocol": PROTOCOL,
                    "encoder": encoder,
                    "representation": view,
                    "cache_dir": str(data / cache),
                    "checkpoint": str(data / checkpoint),
                    "prepared_dir": str(prepared),
                    "output_dir": str(directory),
                    "learner": learner,
                    "seed": seed,
                    "precision": "float32",
                    "threads": 8,
                    "device": "cuda" if learner == "mlp" else "cpu",
                    "max_gpu_hours": 0.08 if learner == "mlp" else 0.5,
                    "ledger": str(history),
                }
                if not (args.resume and completed(config)):
                    config_path = output / f"{name}.json"
                    atomic_json(config_path, config)
                    print(f"Fitting {name}", flush=True)
                    screen(config_path)
                record = json.loads((directory / "run_manifest.json").read_text())
                for method, metrics in record["metrics"].items():
                    records.append(
                        {
                            "encoder": encoder,
                            "view": view,
                            "learner": method,
                            "seed": seed,
                            "MA-ST-RAE": metrics["MA-ST-RAE"],
                            "macro_MAE": metrics["macro_MAE"],
                            "selected_epoch": record["selection"].get("selected_epoch"),
                            "epochs_run": record["selection"].get("epochs_run"),
                        }
                    )
    rows = pd.DataFrame(records)
    rows.to_csv(output / "results.csv", index=False)
    summary = rows.groupby(["encoder", "view", "learner"], sort=False)["MA-ST-RAE"].agg(
        ["mean", "std", "count"]
    )
    summary.to_csv(output / "summary.csv")
    charges = json.loads(history.read_text())
    cumulative = sum(
        r.get("used_hours", r["reserved_hours"])
        for path, r in charges.items()
        if is_pilot_charge(path)
    )
    extension = sum(
        r.get("used_hours", r["reserved_hours"])
        for path, r in charges.items()
        if path.startswith(str(output) + "/")
    )
    atomic_json(
        output / "budget_summary.json",
        {
            "cumulative_gpu_hours": cumulative,
            "extension_gpu_hours": extension,
            "cap": CAP,
        },
    )
    atomic_json(output / "status.json", {"complete": True, "fits": 12, "control_sets": 4})
    print(summary.to_string(), flush=True)
    print(f"Cumulative pilot GPU hours: {cumulative:.6f} / {CAP}", flush=True)


if __name__ == "__main__":
    main()
