# Somerra OpenADMET CYP

Reproducible tooling and CPU baselines for two tracks of the 2026 OpenADMET CYP
Inhibition Blind Challenge:

1. **Direct inhibition regression** — predict pIC50 for CYP1A2, CYP2C9, CYP2D6,
   and CYP3A4.
2. **Time-dependent inhibition (TDI)** — classify TDI for CYP2D6 and CYP3A4.

The structure track is intentionally out of scope for this first phase.

## What is here

- a revision-pinned Hugging Face dataset downloader with SHA-256 manifests;
- schema and identity audits for every official activity file;
- ECFP4 fingerprints plus ten RDKit physicochemical descriptors;
- scaffold-disjoint cross-validation to reduce analogue leakage;
- exact local implementations of the primary ST-RAE and MCC metrics;
- Ridge, logistic-regression, ExtraTrees, and LightGBM baselines;
- leakage-aware out-of-fold MCC threshold selection for TDI;
- strict activity/TDI submission validation;
- saved models, OOF predictions, metrics, and run manifests.

Nothing in this repository uploads a challenge submission.

## Quick start

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
make setup
make download
make audit

# Fast transparent controls
uv run openadmet-cyp baseline activity \
  --model ridge --data-dir data/raw --output-dir runs/activity-ridge
uv run openadmet-cyp baseline tdi \
  --model logistic --data-dir data/raw --output-dir runs/tdi-logistic

# Stronger simple CPU baselines
make baseline-activity
make baseline-tdi
```

Equivalent pip setup, if `uv` is unavailable:

```bash
python3 -m venv .venv
.venv/bin/pip install -e . pytest pytest-cov ruff
.venv/bin/openadmet-cyp download --output data/raw
```

Each baseline run writes:

- `oof_predictions.csv` — out-of-fold predictions and fold assignments;
- `metrics.json` — per-endpoint and macro metrics;
- one serialized full-data model per endpoint;
- a validated `activity_submission.csv` or `tdi_submission.csv`;
- `run_manifest.json` — configuration, feature dimensions, validation report, and hash.

Run directories must not already exist. This prevents accidental overwrites of evidence.

## Validation philosophy

The blinded test set is an analogue-expansion set. Random train/validation splits can
place close chemical relatives on both sides and substantially overstate performance.
Our default split keeps generic Bemis–Murcko scaffolds disjoint. We report:

- direct inhibition: per-endpoint OOF ST-RAE and macro-average MA-ST-RAE;
- TDI: OOF MCC at 0.5 plus a separately labeled OOF-selected threshold.

The threshold-selected MCC is a **model-selection estimate**, not an unbiased final
performance claim. Use an untouched final selection fold before choosing among many
model variants.

## Commands

```bash
uv run openadmet-cyp download --output data/raw
uv run openadmet-cyp audit --data-dir data/raw
uv run openadmet-cyp baseline activity --model lightgbm --output-dir runs/activity-lgbm
uv run openadmet-cyp baseline tdi --model lightgbm --output-dir runs/tdi-lgbm
uv run openadmet-cyp validate activity runs/activity-lgbm/activity_submission.csv
uv run openadmet-cyp score activity path/to/labeled_truth.csv path/to/predictions.csv
```

## Why these baselines

The official tutorial starts with RDKit descriptors and LightGBM but explicitly warns
that random splits are optimistic and recommends chemistry-aware splits. Public
competition work independently converges on Morgan fingerprints, tree models,
scaffold/cluster splits, class balancing, and per-endpoint handling of sparse labels.
Those are the lowest-complexity methods worth trusting.

See [docs/baseline-research.md](docs/baseline-research.md) for the source audit and
[docs/roadmap.md](docs/roadmap.md) for the planned path toward a competitive Somerra
model.

## Data and licensing

Challenge data are downloaded directly from
[`openadmet/cyp-challenge-train-test`](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)
at a pinned Git revision and remain under the dataset's Apache-2.0 terms. Raw files,
model artifacts, and submissions are gitignored.
