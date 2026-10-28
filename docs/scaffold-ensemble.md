# Scaffold folds and a five-model reference ensemble

The user authorized testing scaffold-separated splits and ensembling on 2026-10-28.
This is a new evaluation protocol; the existing molecular-matching task, checkpoints,
and registry remain historical evidence. Our matched depth-four submission scored
0.7952 officially, versus 0.8032 for depth six, reversing their validation ranking.
Use the depth-four recipe unchanged for this first comparison.

## Preferred next candidate: one model with full training coverage

Following the user’s review direction, prefer a single final model over the current
ensemble of models that each train on only three of five folds. The completed
ensemble remains an experimental comparison; its preparation does not make it the
recommended submission. No further ensemble training or upload is implied.

Keep scaffold separation for development evaluation. For the next single-model
candidate, freeze an 80/20 scaffold train/validation assignment before fitting,
using the same assignment for official and auxiliary sources. Select the recipe
and epoch count using validation only. Once fixed, refit one model on all eligible
labeled official training records plus eligible auxiliary records for that fixed
number of epochs. Keep exact official-blind parents excluded across sources.
The final refit has no validation-based checkpoint selection and no independent
local test score; its predictions require official evaluation.

This is the proposed next experiment, not an already executed result or a claim
that the single model will win. The existing five-fold OOF result remains a
single-model scaffold-generalization diagnostic. A five-model average can reduce
variance, but each constituent here sees roughly 19,745 of 32,909 eligible pool
records (60%), versus full coverage in the proposed final refit. Neither scaffold
separation nor ensembling has yet demonstrated an official improvement in our runs.

## Frozen experiment

Use the same pinned 26-task source pool, sample weights, molecular parent standardization,
loss, model configuration, and training-only target scaling as the reference. Compute
non-chiral RDKit Bemis–Murcko scaffolds on standardized parents across the entire pool.
Identical scaffolds, including the empty scaffold for acyclic molecules, stay together
across official and auxiliary sources. Exclude exact official-blind parents everywhere;
blind scaffolds are not removed, and blind labels are unavailable.

Assign whole scaffolds deterministically to five groups with seed 42, balancing pool
record counts, official record counts, and availability of each scored endpoint. Target
values, model predictions, and official-test similarity do not enter assignment. Require
at least 100 labels per scored endpoint in each fold; fail rather than split scaffolds.
This adopts Scigantic's rotation, not its exact fold assignments or full stacking recipe.
Scaffold separation is stricter chemically; better leaderboard alignment remains a hypothesis.

For held-out fold k, validate on (k+1) modulo 5 and train on the other three folds.
All source records follow their scaffold assignment, including auxiliary-only molecules.
Train five depth-four models for 100 epochs, batch 128, scheduler horizon 200, seed 42
for each run. Only validation selects the best checkpoint. Preserve curves, selected
checkpoints, reload parity, validation predictions, and held-out predictions.

The first series has a 0.65 GPU-hour ceiling within the existing shared 12-hour ledger;
each fold is capped at 0.13 hours. The supervisor stops on a failed fold, stalled worker,
source drift, or exhausted allocation. It does not reset historical spending. A systemd
service survives chat disconnection and imposes an additional wall-time limit. CPU
prediction export follows training. All completed artifacts are retained on interruption.

## Evaluation and interpretation

Concatenate exactly one held-out prediction per official record to compute pooled OOF
MA-ST-RAE, using the existing challenge scorer. Also retain each fold's score. This OOF
result measures single models on held-out scaffolds; it is NOT the five-model ensemble's
internal score. Never average all five models on labeled training molecules: some of
those models trained on each molecule. Scores on our old molecular holdout and these
new scaffold folds have different denominators/populations and are not paired gains.

Predict all 750 official blind molecules with each selected checkpoint, then average
predictions with five fixed weights of 0.2. Verify model/data/checkpoint provenance,
finite outputs, and exact molecule identity/order. Reject missing or mismatched folds.
Write a validated prediction CSV and private provenance manifest; never upload from
this workflow. A later authorized leaderboard submission can compare this combined
change against the matched single-model reference. It will not separately identify
the effects of scaffold splitting, smaller per-model training pools, and averaging.

No learned ensemble weights, endpoint-specific corrections, extra model recipes,
additional seeds, or refit on validation data are included in this first experiment.

## Commands

Run from this branch with `PYTHONPATH=src` and the existing GPU Python environment:

```bash
python autoresearch.py scaffold prepare --bundle BASE --model MODEL_JSON --output NEW
python autoresearch.py scaffold run --output NEW --raw RAW --ledger SHARED_LEDGER --hours 0.65
python autoresearch.py scaffold status --output NEW
```

The output root contains `plan.json`, frozen `folds.csv`, and `model.json`. Each
`fold-K/` contains its data manifest, fit log, checkpoints, predictions, and evaluation.
`series.json` records execution progress and spending. Successful completion creates
`ensemble/activity_submission.csv`, `ensemble/oof.csv`, and `ensemble/manifest.json`.
All predictions stay private. Open Code remains false and report links remain empty.

## Completed first series

All five folds completed 100 epochs successfully, charging 0.3292 GPU-hours
(19.75 minutes of supervised training-process time). Selected epochs were
86, 74, 50, 66, and 72. Pooled held-out OOF MA-ST-RAE was 0.69804810;
this is a single-model scaffold-generalization score, not an ensemble test score.
The equally weighted ensemble produced 750 valid blind predictions with no
validation errors or warnings. It has not been submitted.

Private artifacts: `/mnt/data/somerra-openadmet-runs/scaffold-ensemble-20261028`.
Prediction CSV SHA-256:
`ba6fbe30ae24b640ffdaef18c5b88177caa762ebc34873bfea00ce49b1c4706f`.
Validation: 50 focused tests passed, including shared reference, hill-climbing,
autoresearch, submission-selection, and scaffold-ensemble tests; Ruff and diff
whitespace checks passed.

## Review diagnostic: prediction spread

The requested audit compares all 750 blind prediction rows in identical order.
Population standard deviations (`ddof=0`) are:

| Endpoint | Observed official-training labels | Depth four blind | Depth six blind | Ensemble blind | Ensemble shrinkage versus mean member SD |
|---|---:|---:|---:|---:|---:|
| CYP1A2 | 1.0301 | 0.6097 | 0.5171 | 0.5316 | 11.18% |
| CYP2C9 | 0.7820 | 0.5902 | 0.5863 | 0.5696 | 7.03% |
| CYP2D6 | 0.9158 | 0.4571 | 0.4574 | 0.4163 | 9.36% |
| CYP3A4 | 1.0930 | 0.6847 | 0.7003 | 0.6772 | 5.55% |

Averaging does contract spread relative to its constituent models. The blind label
spread is unknown, however, and the observed training labels describe a different
population with endpoint-dependent missingness. Neither matching their variance
nor restoring a previous model's variance is established as correct calibration.
The official performance gap cannot be attributed to calibration from these moments.
No recalibration has been fitted or applied, and the prediction CSV is unchanged.

Private reproducible audit: the run's `review-audit/spread.py` and `spread.json`.
The JSON records input hashes, means, SDs, 5th/50th/95th percentiles, member correlations,
and both single-model and member-relative spread ratios. Every molecule identity/order
and the exact equal-weight average were checked.

OOF predictions can provide inputs to a calibrator, but the fitted calibrator's score
on those same rows is a training score. Evaluate any proposed per-endpoint affine
correction on labels excluded from both calibration fitting and base-model selection.
Simply splitting the existing OOF table is not fully nested evaluation: other fold
models may have trained or selected checkpoints using labels in the evaluation fold.
Use a separate calibration/evaluation design or properly nested scaffold folds before
claiming an unbiased gain, and freeze coefficients before applying to blind predictions.
Transfer from smaller OOF models to a full-data refit is another hypothesis to test.

## Restart limitation

The series runner currently requires a fresh output directory and creates `series.json`
exclusively. It does not resume or skip completed folds after failure. Retained curves,
checkpoints and exports are useful diagnostic artifacts, but there is no supported
CLI path to finish only missing folds in that same series. A fresh `prepare` and `run`
repeats the five fits and charges all repeated work to the existing shared ledger.
Do not delete the old state, overwrite completed artifacts, or reset spending to bypass
this restriction. Automatic resume is deferred while the single-model workflow is the
priority; it would require validated reuse of each member and explicit accounting for
failed/recovered attempts.
