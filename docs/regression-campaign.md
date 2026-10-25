# Regression reproducibility and recipe evidence

Autoresearch is the primary campaign interface; see [program.md](../program.md).
This runbook preserves reproducibility procedures, completed experiment evidence
and best-recipe decisions. It is not a competing orchestration workflow. Research
and saved-recipe reproduction share `evaluation.evaluate_recipe` and its CPU/neural
backends. The lower-level commands below remain useful for audits and reproduction.

The campaign uses public data and a maximum planned allocation of 72 GPU-hours.
Existing runs remain immutable. `regression export` creates a candidate; it does
not promote or upload it. Screening scores are model-selection evidence.

## Current workflow

Follow [program.md](../program.md): `train.py` is the editable model recipe;
its `data_sources` selects inputs from one shared catalog, and the evaluator stays fixed.
See [the catalog](../configs/README.md) for source locations and best recipes.
See [the data-source guide](data-sources.md) for source-specific value, limitations,
and the distinction between biological relevance and demonstrated modeling gains.

```bash
openadmet-cyp regression prepare --output-dir data/prepared/official-direct
python autoresearch.py plan --output-dir runs/regression-trial-001
python autoresearch.py run --output-dir runs/regression-trial-001
openadmet-cyp regression evaluate --model-config configs/models/frozen-challenger.json \
  --data-config configs/data.json --output-dir runs/regression-frozen-trial-001
```

The frozen recipe requires embeddings prepared at its configured path. Use
`regression embed --prepared-dir data/prepared/official-direct --checkpoint
runs/chemeleon_mp.pt --output-dir models/chemeleon/official-direct-embeddings`.
Prepare official auxiliary snapshots with `regression auxiliary` at the catalog location.
For fresh setup of all sources, use `make prepare-regression-data`.
Existing snapshots and runs are unchanged; the shared catalog can reference them.

## Historical reproduction

External integrations have been removed from this PR and deferred to
[ChEMBL #3](https://github.com/Somerra/somerra-openadmet-cyp/issues/3) and
[PubChem #4](https://github.com/Somerra/somerra-openadmet-cyp/issues/4).
The external results below are historical evidence only. Reproduction requires
archived code at commit fb848c4, not the current official-only trainer.

The commands below document the completed campaign, not the maintained config
catalog. Before using a historical command, retrieve its exact flat config from
commit `9b43c07c2c9c3afc79a445634097b4481b79d0ac` into ignored `runs/` and replace
the `configs/` argument accordingly. Do not recreate failed trial configs in Git.

Run CPU commands in the existing supported Conda environment. Neural commands need
the separate `environment-neural.yml` environment; install this package there with
`python -m pip install --no-build-isolation --no-deps -e .`.

```bash
openadmet-cyp regression status
openadmet-cyp regression prepare --output-dir runs/regression-corpus-v1
openadmet-cyp regression auxiliary --output-dir runs/regression-auxiliary-v1
openadmet-cyp regression diagnose \
  --run-dir runs/candidate-20261002-activity-metric-ensemble-v1 \
  --output-dir runs/regression-diagnostics-v1
openadmet-cyp regression evaluate --config configs/regression-controls.json
openadmet-cyp regression compare --run-dir runs/regression-nested-controls-v1
openadmet-cyp regression embed --prepared-dir runs/regression-corpus-v1 \
  --checkpoint runs/chemeleon_mp.pt --output-dir runs/regression-embeddings-v1
openadmet-cyp regression evaluate --config configs/regression-embeddings.json
openadmet-cyp regression evaluate --config configs/regression-incumbent.json
openadmet-cyp regression compare --run-dir runs/regression-nested-embeddings-v1 \
  --reference-run runs/regression-nested-controls-v1 --reference nested_ensemble
openadmet-cyp regression screen-neural --config configs/regression-chemeleon.json
openadmet-cyp regression screen-neural --config configs/regression-chemeleon-auxiliary.json
openadmet-cyp regression screen-neural --config configs/regression-chemprop-scratch.json
openadmet-cyp regression screen-neural --config configs/regression-chemeleon-interval.json
openadmet-cyp regression export --run-dir runs/regression-nested-controls-v1 \
  --output-dir runs/regression-nested-controls-candidate-v1
```

The descriptor-pretrained checkpoint comes from
https://zenodo.org/records/15460715/files/chemeleon_mp.pt and has SHA-256
`c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651`.
The frozen encoder uses no challenge labels. Embedding manifests bind row order
and fold assignments to the prepared training CSV checksum. Use the same checkpoint
when exporting a model evaluated with embeddings.

## Evaluation contract

Standardized parent identities ignore stereochemistry for exclusion/grouping, while
original SMILES remain available to feature generation. ECFP4 Butina clustering uses
distance 0.4 and deterministic assignment balancing endpoint label counts. Butina
groups are disjoint; this does not imply every cross-fold molecular pair has
similarity below 0.6. The first corpus has 4,905 rows, 4,901 parents and 4,565 clusters.
No official training parent overlaps the blinded test parents.

CPU evaluation uses five outer folds, three inner group folds, and simplex weights
fitted only from inner OOF predictions. The paired bootstrap resamples clusters
jointly across endpoints. Its numerical gate is a 3% macro improvement, an upper
95% delta bound below zero, and no endpoint regression greater than 0.02. Comparisons
can use a named model within the same run or `--reference-run` with identical rows,
labels, and folds. The incumbent-parity configuration includes LightGBM radii 2/3,
ExtraTrees radius 2, and Ridge radius 2 with 4,096 simplex search samples. Passing a
numerical comparison alone does not authorize promotion.

Neural screening uses fixed-epoch five-fold training with masked Huber regression
on official direct and auxiliary targets. Losses balance observed heads within source families, then
families; batches are not source-balanced. Normalization is learned only on training
labels. The official-only interval ablation uses distance outside confidence bounds
plus 0.1 times Huber loss in normalized units. Auxiliary observations
matching or similar to an outer holdout molecule are excluded before fitting.
It is explicitly screening, not nested selection of a neural ensemble. Each run
records elapsed GPU time and enforces its configured allocation. New screening runs
reserve allocations in a file-locked `runs/regression-gpu-budget.json` ledger, which
also counts completed pre-ledger runs and embeddings. Embedding extraction/export
does not yet reserve allocations, so this is not a universal GPU-time governor.
The CUDA allocator
is capped at 25% of GPU memory to coexist with the existing GPU workload.

## Model usefulness: evidence

Nested CPU controls: MA-ST-RAE 0.748555, compared with 0.794059 for LightGBM in the
same run. The historical cross-fitted score was 0.746650 on different folds, so this
is not evidence that the nested model improves the incumbent.

The 30-epoch official-label CheMeleon screen scored 0.860705. It is not a promoted
model. CYP2D6 scored 1.115092, reinforcing the need for auxiliary-data and loss
experiments before a final neural choice.

The frozen CheMeleon-feature nested ensemble scored 0.739029 versus the initial
controls' 0.748555: a 1.27% improvement, with paired-cluster delta CI
[-0.020635, 0.002896]. It fails the 3% and confidence gates. The official auxiliary
screen scored 0.879526, also failing to improve the CPU reference.

The completed full incumbent-family nested reference scores 0.743511. Against that
reference, frozen features improve only 0.60%, with delta CI [-0.015325, 0.007675],
and fail promotion. Scratch Chemprop scores 0.798617. Reloading an official-label
CheMeleon fold checkpoint on CPU reproduced stored GPU OOF predictions to a maximum
absolute difference of 8.73e-7 pIC50 on 64 held-out molecules.

Both CPU candidates export 750-row submission CSVs that pass validation without
warnings. Neither is promoted or uploaded.

Incumbent CYP2D6 OOF errors show regression toward the mean: mean signed error is
+1.863 for observed pIC50 <= 4 and -1.494 for pIC50 > 6. These are training-only
diagnostics, not an explanation proven against hidden test labels. Per-bin ST-RAE
can have a near-zero denominator and must be interpreted alongside absolute bias.

## Deferred external-data work

ChEMBL and PubChem integration and source-specific evaluation are tracked in
[issue #3](https://github.com/Somerra/somerra-openadmet-cyp/issues/3) and
[issue #4](https://github.com/Somerra/somerra-openadmet-cyp/issues/4).
The joint external screen scored 0.772899 versus 0.770953 for its matched official
bundle control; external-minus-control delta was +0.001946 with paired 95% CI
[-0.013851, 0.017249]. This is not evidence of a clear benefit. The recovered
external result has incomplete original provenance and is ineligible for promotion.

The official control completed five folds and 6,525 observed endpoint predictions,
using 0.045299 GPU-hours. It was still worse than the CPU incumbent. Its loss
implementation differs from the earlier 30-epoch official auxiliary run, so those
scores do not isolate training duration. Saved runs, checkpoint audits and original
configs are preserved; the full historical runbook and external code remain in
commit fb848c4. Historical fold recovery and external reruns require archived code.
The current entrypoint runs all folds; budget accounting remains shared.

### Outstanding work

The implementation and run artifacts record actual progress, not completion of
the full proposed campaign. Censored-target training, source-balanced batches,
secondary analog evaluation, fully nested neural selection, universal GPU accounting,
and final multi-seed neural fitting remain unimplemented. Historical trial recipes
remain available from Git history and run provenance; consult their `metrics.json` and
`run_manifest.json` for completion, not the existence of an output directory. Final
promotion requires comparison against the completed incumbent-parity run. No failed
screen should be promoted merely because its experiment finished.


### Evaluation corrections after review

Neural fitting now uses standardized parent SMILES and median targets per parent in
both direct-only and auxiliary trials; validation uses the same parent representation.
Historical neural scores above predate this correction. New matched trials are needed
before comparing data choices; this code review did not rerun training.
CPU features continue to use the submitted SMILES. Trial records distinguish
`nested_ensemble` CPU scores from `neural_screen` fixed-epoch scores; both report
MA-ST-RAE, minimized, but a screen is not nested model selection.

CPU provenance is saved before featurization/training. Missing Git metadata is
recorded as null while source hashes are retained. Failed runs retain an incomplete
manifest. Comparison reports refuse overwrites, including the default path.

The GPU ledger defaults to `runs/regression-gpu-budget.json` at the checkout root,
independently of output directories. Set `SOMERRA_GPU_LEDGER` to one absolute shared
path when using multiple worktrees or checkouts; status reads the same ledger.
Outstanding reservations retain their full charge until usage is established.
