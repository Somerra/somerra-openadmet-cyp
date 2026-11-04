"""Render the paper figures from the accompanying aggregate evidence."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "figure-data.json").read_text())
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "savefig.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


fig, ax = plt.subplots(figsize=(3.35, 1.65))
campaigns = DATA["campaigns"]
counts = [c["completed_scored_evaluations"] for c in campaigns]
labels = [c["label"] for c in campaigns]
ax.barh(range(len(counts)), counts, height=0.65, facecolor="0.75", edgecolor="black", linewidth=0.5)
ax.invert_yaxis()
ax.set_yticks(range(len(counts)), labels, fontsize=7)
ax.set_xlim(0, 180)
ax.set_xticks([0, 50, 100, 150])
ax.set_xlabel("Completed scored evaluations", fontsize=7)
ax.tick_params(axis="y", length=0)
for i, n in enumerate(counts):
    ax.text(n + 3, i, str(n), va="center", fontsize=7)
save(fig, "research-volume")

fig, ax = plt.subplots(figsize=(3.35, 1.15))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(0.5, 0.94, "Potent seed hits  →  structure-selected analogs", ha="center", fontsize=7.4)
ax.text(0.5, 0.74, "Whole nomination families  →  A (112) / B (111)", ha="center", fontsize=7.4)
ax.plot([0.02, 0.98], [0.60, 0.60], color="black", lw=0.6)
ax.text(0.10, 0.35, "Fixed\ntraining", ha="center", va="center", fontsize=8)
ax.annotate("", xy=(0.30, 0.48), xytext=(0.22, 0.35), arrowprops={"arrowstyle": "->", "lw": 0.6})
ax.annotate("", xy=(0.30, 0.18), xytext=(0.22, 0.35), arrowprops={"arrowstyle": "->", "lw": 0.6})
ax.text(0.33, 0.48, "1: validate B  →  score A", va="center", fontsize=8)
ax.text(0.33, 0.18, "2: validate A  →  score B", va="center", fontsize=8)
save(fig, "hit-expansion-logic")

fig, ax = plt.subplots(figsize=(3.35, 1.35))
pairs = DATA["hit_expansion_results"]
for i, p in enumerate(pairs):
    ax.bar(i - 0.17, p["control"], width=0.30, facecolor="white", edgecolor="black", linewidth=0.6, label="Control" if i == 0 else None)
    ax.bar(i + 0.17, p["selected"], width=0.30, facecolor="0.65", edgecolor="black", linewidth=0.6, label="Selected" if i == 0 else None)
    for x, v in [(i - 0.17, p["control"]), (i + 0.17, p["selected"])]:
        ax.text(x, v + 0.025, f"{v:.4f}", ha="center", fontsize=7)
ax.set_xticks(range(len(pairs)), [p["cyp"] for p in pairs])
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.5, 1.0])
ax.set_ylabel("ST-RAE")
ax.legend(frameon=False, ncol=2, fontsize=7, loc="upper right", borderpad=0)
save(fig, "hit-expansion-progress")
