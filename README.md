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
- cross-fitted, ST-RAE-aligned activity ensembles over completed baseline runs.

Nothing in this repository uploads a challenge submission.

## Environment

Conda is the sole supported environment manager. The authoritative specification is
`environment.yml`, resolved from conda-forge. Python is pinned to 3.12 and runtime plus
development dependencies are declared together so local runs and CI use the same stack.

Create or update the environment and install this project in editable mode:

```bash
make setup
```

This runs `conda env update --prune` and then installs only the local package without
build isolation or dependency resolution; all third-party dependencies remain
Conda-managed.

## Quick start

Requires Conda; Miniforge is recommended for a clean conda-forge installation.

```bash
make setup
make download
make verify-data
make audit

# Fast transparent controls
conda run -n somerra-openadmet-cyp openadmet-cyp baseline activity \
  --model ridge --data-dir data/raw --output-dir runs/activity-ridge
conda run -n somerra-openadmet-cyp openadmet-cyp baseline tdi \
  --model logistic --data-dir data/raw --output-dir runs/tdi-logistic

# Stronger simple CPU baselines
make baseline-activity
make baseline-tdi
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
conda run -n somerra-openadmet-cyp openadmet-cyp download --output data/raw
conda run -n somerra-openadmet-cyp openadmet-cyp verify-data --data-dir data/raw
conda run -n somerra-openadmet-cyp openadmet-cyp audit --data-dir data/raw
conda run -n somerra-openadmet-cyp openadmet-cyp baseline activity --model lightgbm --output-dir runs/activity-lgbm
conda run -n somerra-openadmet-cyp openadmet-cyp baseline tdi --model lightgbm --output-dir runs/tdi-lgbm
conda run -n somerra-openadmet-cyp openadmet-cyp ensemble activity \
  --run-dir runs/activity-lgbm runs/activity-extra-trees runs/activity-ridge \
  --output-dir runs/activity-ensemble
conda run -n somerra-openadmet-cyp openadmet-cyp validate activity runs/activity-lgbm/activity_submission.csv
conda run -n somerra-openadmet-cyp openadmet-cyp score activity path/to/labeled_truth.csv path/to/predictions.csv
```

Baseline commands default to eight model-training threads. Use `--n-jobs` to change the
budget; the selected value is written into `run_manifest.json`.

## Autoresearch entrypoint

Start here from the repository root, after preparing the selected data:

```bash
# Preview effective settings and source selection; no training or run files.
python autoresearch.py plan --output-dir runs/regression-trial-001

# Explicitly execute ONE trial after agreeing the hypothesis and compute budget.
python autoresearch.py run --output-dir runs/regression-trial-001 \
  --hypothesis "Removing ExtraTrees preserves quality with less complexity"
```

Edit the recipe block in `train.py` before previewing a proposed change; the
hypothesis string does not modify model settings. `autoresearch.py` is the fixed
launcher; `src/somerra_openadmet/research.py` owns orchestration. Each run snapshots
its inputs and writes `trial.json` with completion/failure status and elapsed time.
Completed trials await review: comparison, keep/discard, and promotion are separate.
`plan` validates source composition, not dataset contents or backend readiness.
There is no autonomous scheduler or automatic repetition. The older `python train.py`
invocation remains a compatibility alias for one run.

**External data is optional, not a proven improvement.** The matched neural screen
scored 0.772899 with ChEMBL/PubChem versus 0.770953 without them (lower is better).
The paired delta CI crosses zero, and the recovered external run has incomplete
provenance. Neither source has an isolated matched ablation. Maintained recipes
therefore leave external sources off; see [the evidence](docs/data-sources.md#what-the-data-comparison-actually-established).

## Why these baselines

Regression research follows an [autoresearch-style protocol](program.md): edit one
model recipe in `train.py`, keep the evaluator fixed, and retain only the best
configurations. One [data catalog](configs/README.md) registers all sources; each
model recipe chooses its `data_sources`. Model experiments freeze that selection;
data experiments change only the selection, keeping model/training settings fixed.
Each invocation runs one trial, not an unattended loop.
The [data-source guide](docs/data-sources.md) explains what each source contributes,
why it might help regression, and what our experiments have actually established.

The ongoing regression campaign adds persisted molecular cluster folds, nested
CPU evaluation, paired cluster bootstrap, auxiliary assay preparation, and optional
CheMeleon experiments. See [the campaign runbook](docs/regression-campaign.md) for
commands, results, and outstanding promotion gates.

The official tutorial starts with RDKit descriptors and LightGBM but explicitly warns
that random splits are optimistic and recommends chemistry-aware splits. Public
competition work independently converges on Morgan fingerprints, tree models,
scaffold/cluster splits, class balancing, and per-endpoint handling of sparse labels.
Those are the lowest-complexity methods worth trusting.

See [docs/baseline-research.md](docs/baseline-research.md) for the source audit and
[docs/baseline-results.md](docs/baseline-results.md) for the first full baseline run.
[docs/next-submission.md](docs/next-submission.md) records the Scigantic method audit
and the cross-fitted ensemble candidate derived from it.
[docs/roadmap.md](docs/roadmap.md) describes the planned path toward a competitive
Somerra model.

## Data and licensing

Challenge data are downloaded directly from
[`openadmet/cyp-challenge-train-test`](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)
at pinned revision `146282d2ce1532f1803679a834a9bec9dac1e482` (confirmed current on
2026-09-30) and remain under the dataset's Apache-2.0 terms. The manifest verifier
checks every expected file's size and SHA-256 before modeling. Raw files, model
artifacts, and submissions are gitignored.
