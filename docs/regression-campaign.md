# Regression campaign

The campaign uses public data and a maximum planned allocation of 72 GPU-hours.
Existing runs remain immutable. `regression export` creates a candidate; it does
not promote or upload it. Screening scores are model-selection evidence.

## Reproduction

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
openadmet-cyp regression fetch-chembl --output-dir runs/regression-chembl37-v1
openadmet-cyp regression fetch-pubchem --output-dir runs/regression-pubchem1851-v1
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
openadmet-cyp regression screen-neural --config configs/regression-chemeleon-external.json
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
and BCE binary heads. Losses balance observed heads within source families, then
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

## Initial evidence

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
warnings. Neither is promoted or uploaded. ChEMBL 37 preparation retained 29,256
records (13,778 exact potency measurements); PubChem AID 1851 retained 62,712 binary
outcomes from 16,135 parents. Censored ChEMBL records are preserved separately and
are not fed to exact-target training. PubChem inactive outcomes remain binary, not
invented potency values. Raw responses, exclusions, and hashes accompany snapshots.

Incumbent CYP2D6 OOF errors show regression toward the mean: mean signed error is
+1.863 for observed pIC50 <= 4 and -1.494 for pIC50 > 6. These are training-only
diagnostics, not an explanation proven against hidden test labels. Per-bin ST-RAE
can have a near-zero denominator and must be interpreted alongside absolute bias.

## Remaining campaign gates

### Recovery of the interrupted external-data screen

The original external run saved folds 0–3 but exited without a final manifest.
The exact termination cause is unknown. CPU reload reproduced all saved OOF
predictions within 2.31e-6 pIC50, below the predeclared 1e-4 tolerance. Original
artifact hashes, input hashes, and validation evidence are retained in
`runs/regression-external-recovery-audit-v1/audit.json`.

Recovery preserves the original files, selects only fold 4 (seed 46), and keeps
the original 10-epoch model settings. New runs write startup/per-fold manifests.
`requested_folds_complete` indicates selected-fold completion; `complete` remains
false for a fold-only run. The original four-hour allocation remains charged as
`interrupted_unknown_usage`; recovery reserves at most one additional hour.

```bash
openadmet-cyp regression screen-neural \
  --config configs/regression-chemeleon-external-fold4.json
openadmet-cyp regression merge-neural \
  --original-config configs/regression-chemeleon-external.json \
  --recovered-dir runs/regression-chemeleon-external-fold4-v1 \
  --audit runs/regression-external-recovery-audit-v1/audit.json \
  --output-dir runs/regression-chemeleon-external-completed-v1
openadmet-cyp regression compare \
  --run-dir runs/regression-chemeleon-external-completed-v1 \
  --reference-run runs/regression-nested-incumbent-v1 --reference nested_ensemble
```

The merge validates disjoint, complete observed-label coverage and unchanged audited
inputs. It does not reconstruct missing original training provenance. Combined
results are provisional screening evidence, explicitly ineligible for promotion;
any promising result needs a fully traced rerun and the remaining campaign gates.

Recovery completed successfully: fold 4 produced 1,305 observed endpoint predictions
and used 0.051796 GPU-hours (about 3.1 minutes). Original checkpoint and OOF hashes
remained unchanged. The five-fold external-data screen scored **0.772899** versus
the nested incumbent's **0.743511**: 3.95% worse, with paired-cluster score-delta
95% CI **[0.008644, 0.051533]**. It fails the numerical promotion gate.

| Endpoint | External screen | Nested incumbent | Delta (lower is better) |
| --- | ---: | ---: | ---: |
| CYP1A2 | 0.907547 | 0.832707 | +0.074840 |
| CYP2C9 | 0.684527 | 0.683655 | +0.000871 |
| CYP2D6 | 0.977178 | 0.915334 | +0.061844 |
| CYP3A4 | 0.522342 | 0.542348 | -0.020006 |

Only CYP3A4 improved in this comparison. This is not evidence of an overall gain
from external data, and the screen does not isolate external data from differences
in architecture or training duration. No model was promoted or uploaded.

### Matched external-data control (fixed before viewing results)

The next authorized run uses `configs/regression-chemeleon-official-auxiliary-10epoch.json`:
five existing folds, 10 epochs, seed 42 plus fold, the same CheMeleon checkpoint,
learning rate, optimizer, Huber/BCE implementation, and neural environment. It
retains official direct and official auxiliary TDI/Emax/log2fc targets, removing
only ChEMBL and PubChem. No hyperparameter sweep or endpoint-specific selection.

The primary contrast is **external screen minus matched official-auxiliary control**
in macro MA-ST-RAE (negative favors external data), with 1,000 paired cluster bootstrap
samples and seed 42. All four endpoint deltas are reported. Incumbent comparison is
context, not a substitute for the primary contrast. This tests adding the two
external sources together as implemented, including changes in examples, task heads,
source weighting, and optimizer steps per epoch; it is not a step-matched causal
test of data content alone. The recovered external run's provenance caveat persists.

```bash
openadmet-cyp regression screen-neural \
  --config configs/regression-chemeleon-official-auxiliary-10epoch.json
openadmet-cyp regression compare \
  --run-dir runs/regression-chemeleon-external-completed-v1 \
  --reference-run runs/regression-chemeleon-official-auxiliary-10epoch-v1 \
  --reference y_pred \
  --output runs/regression-chemeleon-official-auxiliary-10epoch-v1/external-comparison.json
```

The matched control completed all five folds (6,525 observed endpoint predictions),
using 0.045299 GPU-hours. Its macro score is **0.770953**, versus **0.772899** with
ChEMBL/PubChem included. The primary external-minus-control difference is
**+0.001946** (external 0.25% worse), with paired-cluster 95% CI
**[-0.013851, 0.017249]**. This does not establish an overall benefit or harm from
the external-data addition in this configuration.

| Endpoint | Official-auxiliary control | With external data | External minus control |
| --- | ---: | ---: | ---: |
| CYP1A2 | 0.901525 | 0.907547 | +0.006022 |
| CYP2C9 | 0.672128 | 0.684527 | +0.012399 |
| CYP2D6 | 0.977848 | 0.977178 | -0.000670 |
| CYP3A4 | 0.532311 | 0.522342 | -0.009969 |

The control itself is 3.69% worse than the nested incumbent, with score-delta CI
[0.007931, 0.048730]. Neither neural configuration qualifies for promotion.
Comparison reports are saved separately in the control directory as
`external-comparison.json` (primary) and `comparison.json` (incumbent context).
The earlier 30-epoch official-auxiliary screen used per-head loss averaging rather
than the current source-family balancing, so the 10/30-epoch scores are not a clean
training-duration ablation.

Astra's next proposed step is a read-only OOF error-complementarity audit of the
fully traced control versus the incumbent, by endpoint and molecular cluster.
This would assess whether a later properly nested ensemble experiment is worth
running. It is exploratory: no blend-weight fitting or promotion claims from
reusing these OOF results. No further training was launched with this control run.

### Outstanding work

The implementation and run artifacts record actual progress, not completion of
the full proposed campaign. Censored-target training, source-balanced batches,
secondary analog evaluation, fully nested neural selection, universal GPU accounting,
and final multi-seed neural fitting remain unimplemented. Scratch, interval, external,
and incumbent-parity configurations are runnable; consult their `metrics.json` and
`run_manifest.json` for completion, not the existence of an output directory. Final
promotion requires comparison against the completed incumbent-parity run. No failed
screen should be promoted merely because its experiment finished.
