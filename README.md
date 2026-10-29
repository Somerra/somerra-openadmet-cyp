# Somerra OpenADMET CYP

GPU regression autoresearch and reproducible baseline tooling for two tracks of the 2026 OpenADMET CYP
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

The standalone `openadmet-molecular-split generate|audit` command builds and audits
structure-matched 60/20/20 splits. See [molecular similarity and split matching](docs/molecular-similarity-splits.md)
for the objective, acceptance gate, private artifacts, and evaluation limitations.
The primary regression task adopts reviewed assignments explicitly and freezes them for subsequent trials.

Training commands do not upload submissions. The separate, explicitly executed
Workbench submission watcher supports authorized prediction-only uploads; see the
[October 28 run and official result](docs/briford-run-20261028.md).

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
The primary regression task uses a frozen molecular 60/20/20 train/validation/internal-test split.
Validation selects checkpoints; internal test is a strategy-development set. The
official blinded test is scored only by the challenge. Historical CPU commands
use scaffold/cluster cross-validation and report:

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

## Autoresearch is the primary entrypoint

`autoresearch.py` now runs one GPU candidate on the frozen molecular regression
task. It reuses the saved incumbent, selects a checkpoint on validation, compares
that checkpoint on internal test with a paired scaffold bootstrap, and records a
keep/discard/crash decision. Reproduction uses the same training and evaluation
functions without importing the editable `train.py` settings or promoting a model.

Initialize once in the GPU environment from the prepared bundle, completed
100-epoch reference, and existing campaign ledger:

```bash
python autoresearch.py init --bundle /absolute/path/to/frozen-bundle \
  --reference /absolute/path/to/reference-run --task-dir runs/regression-task \
  --campaign-ledger /absolute/shared/path/to/campaign-ledger.json

# Edit MODEL_OVERRIDES in train.py, then preview and run one candidate.
python autoresearch.py plan --task runs/regression-task/task.json
python autoresearch.py run --task runs/regression-task/task.json \
  --output-dir runs/candidate-001 --hours 1 \
  --hypothesis "Describe the model change made in train.py"

python autoresearch.py status --task runs/regression-task/task.json
python autoresearch.py reproduce --task runs/regression-task/task.json \
  --trial runs/candidate-001 --output-dir runs/reproduction-001 --hours 1
# Add --preview to inspect a reproduction without training.

# Separate final fit of the selected incumbent; prepares predictions, never uploads.
python autoresearch.py prepare --task runs/regression-task/task.json \
  --output-dir runs/submission-preparation-001 --raw data/raw --hours 1
```

`configs/regression/incumbent.json` holds the maintained model recipe. Trial
snapshots, model-source hashes, predictions, curves and decisions live under
ignored run directories. A persistent registry holds the current incumbent and
trial history. Concurrent completions compare against the latest incumbent under
a lock. No baseline rerun or final fit is added to ordinary trials.

**Preserve artifacts privately:** `task.json` and `registry.json` are indexes, not
self-contained backups. Preserve the full adopted-reference and current-incumbent
run directories (including `run_manifest.json`, `best.ckpt`, and
`internal-test.csv`), the frozen data bundle, allocation ledger, and any trial
snapshots needed for reproduction. Preserve the ledger's lock path and stop
writers before taking a consistent backup. Keep the recorded source revision and
runtime as well. A fresh clone/container does not include ignored `runs/` artifacts.
Restore these private directories at their recorded absolute paths before
continuing; relocation/import is not currently supported. Do not rewrite paths or
hashes in a frozen task. Missing or modified artifacts intentionally fail validation.
Never commit checkpoints or predictions to make the task portable.

Adoption preserves the implementation hashes in the fitted run's manifest; it
never substitutes the current tree's hashes. If model or trainer code differs,
`init` refuses by default. Use `init --check-model-compatibility` to explicitly
reload the checkpoint and compare all saved internal-test predictions (absolute
tolerance 1e-4, matching molecule identities and endpoint order). The task records
the original and compatible hashes, checkpoint/prediction/manifest hashes, bundle,
recipe, runtime and observed error. Missing legacy modules remain marked missing.
`prepare` requires either the original implementation or this exact recorded
compatibility check, and rejects subsequent changes. This validates checkpoint
inference, not equivalence of retraining under arbitrarily changed training code;
training-semantic changes still require a newly evaluated candidate. Existing tasks
must remain on their recorded revision; initialize a new directory for this update
and reuse the existing allocation ledger.

Data hashes, evaluation code, seed 42, batch 128, the 100-epoch cap and runtime
versions are frozen in `task.json`. Model configuration and the model implementation
in `hillclimb_model.py`/`model_recipe.py` may evolve between trials; changing the
evaluator requires a new task. Reproduction verifies the saved model implementation
hashes and recipe, so restore the recorded implementation if it has since changed.
An agent proposes the next change and invokes the next trial; the CLI does not
invent hypotheses or repeat indefinitely. See [program.md](program.md).

The task uses the pinned 26-task pool, including official auxiliary assays and
Scigantic derivatives. Fresh raw-source acquisition remains deferred to
[ChEMBL #3](https://github.com/Somerra/somerra-openadmet-cyp/issues/3) and
[PubChem #4](https://github.com/Somerra/somerra-openadmet-cyp/issues/4).

Historical CPU/fold-based research is explicit: `autoresearch.py legacy plan|run|reproduce`.
The previous fixed-series GPU study remains available under `autoresearch.py hillclimb`.
Neither is selected by the primary `plan`, `run` or `reproduce` commands.

## Why these baselines

Regression research follows an [autoresearch-style protocol](program.md): edit one
model recipe in `train.py`, keep the evaluator fixed, and retain only the best
configurations. The current model task freezes its prepared data bundle and source
selection. A separate data study can vary source/task selection while holding the
model fixed. The historical [data catalog](configs/README.md) documents source locations.
Each invocation runs one trial, not an unattended loop.
The [data-source guide](docs/data-sources.md) explains what each source contributes,
why it might help regression, and what our experiments have actually established.

The historical regression workflow includes persisted molecular cluster folds, nested
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

### Published regression reference

`python autoresearch.py reference --help` verifies the pinned Scigantic R25 submission
from frozen predictions without training. See [the baseline and experiment plan](docs/scigantic-baseline.md)
for measured reconstruction agreement, evaluation limitations, and the proposed
organizer-only training baseline. This reference has no independent local CV score.

The current Workbench/briford baseline is a source-data reproduction of the
historical four-component submission. See [the recipe, splits and commands](docs/briford-reproduction.md).
Scigantic model work remains paused; previous evaluation assignments are unchanged.
