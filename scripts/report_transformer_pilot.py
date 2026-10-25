"""Render a completed pilot's learning curves and private comparison report."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def report(directory):
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Pilot is incomplete")
    families = ["mt_aux8", "smiles_transformer"]
    curves = {f: json.loads((directory / f"{f}-curves.json").read_text()) for f in families}
    names = {"mt_aux8": "Graph reference", "smiles_transformer": "SMILES transformer"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    shared = {}
    for col, axis in enumerate(["steps", "seconds"]):
        lower = max(curves[f][0][axis] for f in families)
        upper = min(curves[f][-1][axis] for f in families)
        shared[axis] = {"range": [lower, upper]}
        for row, metric in enumerate(["validation_loss", "MA-ST-RAE"]):
            for family in families:
                x = np.array([c[axis] for c in curves[family]])
                y = np.array([c[metric] for c in curves[family]])
                selected = (x >= lower) & (x <= upper)
                axes[row, col].plot(x[selected], y[selected], label=names[family], alpha=0.8)
                shared[axis].setdefault(family, {})[metric] = {
                    "best_observed_in_shared_range": float(y[selected].min()),
                    "last_observed_in_shared_range": float(y[selected][-1]),
                }
            axes[row, col].set_xlabel(
                "Successful optimizer steps" if axis == "steps" else "Elapsed seconds"
            )
            axes[row, col].set_ylabel(
                "Inner-validation weighted MAE" if row == 0 else "Inner-validation MA-ST-RAE"
            )
            axes[row, col].grid(alpha=0.2)
            axes[row, col].legend()
    fig.suptitle("Matched-size pilot: shared training range only (lower is better)")
    fig.savefig(directory / "learning-curves.png", dpi=180)
    fig.savefig(directory / "learning-curves.svg")
    plt.close(fig)
    results = manifest["results"]
    ref, candidate = [results[f] for f in families]
    summary = {
        "outer_macro_relative_change": candidate["outer"]["MA-ST-RAE"] / ref["outer"]["MA-ST-RAE"]
        - 1,
        "throughput_ratio": candidate["examples_per_second"] / ref["examples_per_second"],
        "total_charged_gpu_hours": manifest["gpu_hours"] + manifest["prior_gpu_hours"],
        "common_range_comparison": shared,
        "environment": {
            p: importlib.metadata.version(p)
            for p in ["torch", "chemprop", "numpy", "pandas", "rdkit", "scikit-learn"]
        },
    }
    (directory / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Small SMILES-transformer pilot",
        "",
        "The SMILES transformer did not improve accuracy or training throughput in this pilot.",
        "",
        "| Measure | Graph reference | SMILES transformer |",
        "| --- | ---: | ---: |",
    ]
    for label, a, b in [
        ("Trainable parameters", ref["parameters"], candidate["parameters"]),
        (
            "Outer MA-ST-RAE",
            f"{ref['outer']['MA-ST-RAE']:.6f}",
            f"{candidate['outer']['MA-ST-RAE']:.6f}",
        ),
        ("Best inner-loss epoch", ref["best_epoch"], candidate["best_epoch"]),
        ("Completed epochs", ref["completed_epochs"], candidate["completed_epochs"]),
        (
            "Examples/sec, including validation",
            round(ref["examples_per_second"]),
            round(candidate["examples_per_second"]),
        ),
        (
            "Peak allocated GPU MiB",
            round(ref["peak_memory_bytes"] / 2**20, 1),
            round(candidate["peak_memory_bytes"] / 2**20, 1),
        ),
    ]:
        lines.append(f"| {label} | {a} | {b} |")
    lines += [
        "",
        "| Outer endpoint | Graph reference | SMILES transformer |",
        "| --- | ---: | ---: |",
    ]
    for ep, value in ref["outer"]["endpoints"].items():
        lines.append(
            f"| {ep.split('_')[0]} | {value:.6f} | {candidate['outer']['endpoints'][ep]:.6f} |"
        )
    lines += [
        "",
        "Total charged GPU allocation time, including the failed attempt: "
        f"{summary['total_charged_gpu_hours'] * 3600:.2f} seconds of the 3,600-second cap.",
        "",
        "Both models used eight organizer-only targets and the same "
        "3,128/794/983 fit/inner/outer split. "
        "Both stopped after 40 epochs without improved inner weighted MAE. "
        "The outer fold did not select checkpoints.",
        "",
        "The first attempt failed on a normal FP16 gradient-scaling overflow. "
        "Its artifacts and charge are retained. "
        "The corrected reference skipped one overflow update; the transformer skipped none.",
        "",
        "The figure compares inner validation curves only over common observed step/time ranges. "
        "Throughput includes validation and checkpoint overhead, "
        "and is not an inference benchmark.",
        "",
        "This is an exploratory single-seed, single-fold local adaptation of one R25 member. "
        "It does not establish general transformer sample efficiency "
        "or reproduce the full R25 ensemble. "
        "No model was promoted or submitted.",
        "",
        "![Learning curves](learning-curves.png)",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.run_dir), indent=2))
