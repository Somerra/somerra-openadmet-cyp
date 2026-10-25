"""Report fixed-trial inner-validation tuning without opening outer results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def report(path):
    manifest = json.loads((path / "run_manifest.json").read_text())
    if not manifest["complete"] or manifest["outer_evaluated"]:
        raise ValueError("Expected a completed inner-only study")
    results = manifest["results"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    lines = [
        "# Transformer follow-up: inner-validation results",
        "",
        "All results below are adaptive model-selection evidence on the existing inner split. "
        "The outer holdout was not evaluated.",
        "",
        "| Trial | Selected inner MAE | Inner MA-ST-RAE at selected epoch | Epoch | "
        "Updates | Examples/s | Peak MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trial in manifest["config"]["trials"]:
        name = trial["name"]
        result = results[name]
        selected = result["selected_inner"]
        curves = json.loads((path / name / f"{trial['family']}-curves.json").read_text())
        for ax, key in zip(axes, ["steps", "seconds"], strict=True):
            ax.plot(
                [c[key] for c in curves],
                [c["validation_loss"] for c in curves],
                label=name,
                linewidth=1.2,
                alpha=0.85,
            )
            ax.set_xlabel("Successful optimizer updates" if key == "steps" else "Elapsed seconds")
            ax.set_ylabel("Inner weighted masked MAE (lower is better)")
            ax.grid(alpha=0.2)
        lines.append(
            f"| {name} | {selected['validation_loss']:.6f} | "
            f"{selected['MA-ST-RAE']:.6f} | {result['best_epoch']} | "
            f"{result['steps']} | {result['examples_per_second']:.0f} | "
            f"{result['peak_memory_bytes'] / 2**20:.1f} |"
        )
    axes[1].legend(fontsize=7)
    fig.suptitle("Small transformer follow-up: inner-validation curves")
    fig.savefig(path / "learning-curves.png", dpi=180)
    fig.savefig(path / "learning-curves.svg")
    plt.close(fig)
    total = manifest["gpu_hours"] + manifest["prior_gpu_hours"]
    lines += [
        "",
        f"Selected trial: **{manifest['selected_trial']}** (minimum inner weighted MAE).",
        "",
        f"Follow-up charged time: {manifest['gpu_hours'] * 3600:.1f} seconds. "
        f"Cumulative pilot plus follow-up: {total * 3600:.1f} / 3,600 seconds.",
        "",
        "All SMILES trials retain the same 1,398,248-parameter architecture. Controls use "
        "the original training recipe. Regularized trials use AdamW, weight decay 0.01, "
        "peak learning rate 0.0003, cosine decay, five warmup epochs, BF16, patience 60, "
        "and a minimum of 100 epochs. The augmented trial cycles through eight deterministic "
        "training-only SMILES enumerations.",
        "",
        "Larger batches change optimizer-step counts per epoch. The regularized recipe "
        "changes multiple settings together; its result cannot identify which setting caused "
        "a difference. Runtime includes validation/checkpoint overhead and is hardware-specific. "
        "Seeds are fixed, but GPU kernels are not forced to be bitwise deterministic.",
        "",
        "No outer-fold scoring, promotion, submission, or larger-model search was performed.",
        "",
        "![Learning curves](learning-curves.png)",
        "",
    ]
    (path / "report.md").write_text("\n".join(lines))
    return {
        k: {
            "inner_mae": v["selected_inner"]["validation_loss"],
            "inner_macro": v["selected_inner"]["MA-ST-RAE"],
            "best_epoch": v["best_epoch"],
            "epochs": v["completed_epochs"],
            "examples_per_second": v["examples_per_second"],
        }
        for k, v in results.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.run_dir), indent=2))
