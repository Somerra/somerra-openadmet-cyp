# Regression calibration policy

Effective 2026-11-02. Historical Workbench calibration is the default calibrated
variant to prepare for future regression submission candidates, alongside an
immutable raw baseline. This documents the user's November 1 authorization to
use the historical constants and November 2 request to preserve the policy after
the matched submission result. A different calibration is a separately declared
experiment; document the reason for promoting a raw or alternative variant.
Preparation does not authorize another upload, new training, or organizer contact.

## What the operation does

Calibration changes the location and spread of the predictions, separately for
each CYP. It is postprocessing: reuse the saved raw predictions and OOF predictions.
No neural-network retraining or checkpoint change is required. First undo training
target normalization to obtain pIC50, then average folds/models using the frozen
ensemble weights, then calibrate that final prediction vector once.

For endpoint e, let p be the final raw predictions on the entire 750-compound
official blind cohort. Let r be Pearson correlation between that candidate's
aligned raw OOF predictions and observed official direct-inhibition labels.
Missing labels are omitted; predictions for labeled rows must be finite. Preserve
the exact OOF rows, fold assignments, masks, aggregation and source hashes. An OOF
row must come from its held-out model, not an average that includes models trained
on that row. Do not substitute a correlation from another candidate without
explicitly labeling that as a different diagnostic.

Define adjusted correlation q = r * factor and target SD = q * population SD.
The frozen historical constants are:

| CYP | Population mean | Population SD | OOF factor | Final override |
|---|---:|---:|---:|---|
| CYP1A2 | 4.412 | 1.553 | 1.32 | None |
| CYP2C9 | 4.830 | 1.101 | 1.23 | None |
| CYP2D6 | 3.107 | 1.599 | 1.66 | Target mean 3.57; target SD 0.90 |
| CYP3A4 | 4.880 | 1.272 | 1.07 | None |

Apply the CYP2D6 override after computing and validating adjusted correlation,
matching the existing implementation. For each prediction:

```text
calibrated_pIC50 = max(1.0,
    target_mean + (raw_pIC50 - mean(raw_cohort))
                  * target_sd / sample_sd(raw_cohort))
```

Use sample SD (`ddof=1`). Compute cohort moments once over all 750 predictions per
CYP, including compounds without any locally known label. Do not normalize each
inference batch, fold or chemical subgroup separately. No official blind labels
enter the calculation. Normalizing to the unlabeled prediction cohort makes this
a transductive transform. Before flooring, the output has the target mean and SD;
flooring can change those moments and introduce ties in otherwise preserved ranks.

The canonical implementation is
[`briford.calibrated`](../src/somerra_openadmet/briford.py), with `MOMENTS` and
`OOF_FACTORS` in the same module. Do not silently clamp invalid correlations:
the implementation rejects nonfinite adjusted correlation or values outside
0 < q <= 1, including before the CYP2D6 override, and nonpositive cohort SD.
Callers must also reject missing/nonfinite predictions, undefined correlations,
wrong units, duplicate/missing IDs and a cohort with fewer than two rows.

These constants originate in the historical Workbench/briford recipe, reproduced
internally in [the October 28 run](briford-run-20261028.md). They include borrowed
population moments, OOF extrapolation factors and a leaderboard-informed CYP2D6
override. This is **leaderboard-informed calibration**, not calibration estimated
solely from our labels. The user's authorization lifted the project hold; it does
not establish organizer-specific clearance. The separate hold on third-party
Scigantic prediction reuse remains unchanged.

## Matched official evidence

The latest-selected candidate was submitted raw on November 1 and with only the
historical transform changed on November 2. Models, ensemble weights, molecule
IDs and the underlying raw prediction file were unchanged. These are recorded
official leaderboard observations, not local OOF scores:

| Metric | Raw | Historical calibration |
|---|---:|---:|
| MA-ST-RAE (lower is better) | 0.7151 | 0.4612 |
| MA-MAE | 0.8974 | 0.6498 |
| MA-R² | 0.2154 | 0.5993 |
| MA-Spearman | 0.7405 | 0.7405 |
| MA-Kendall | 0.5642 | 0.5642 |

MA-ST-RAE fell by 0.2539, approximately 35.5% relative to the raw score, based on
the portal's rounded values. This is a matched calibration comparison for this
candidate, unlike subtracting a local raw score from a different model's official
score. It supports retaining calibration in submission preparation; one official
comparison does not establish a universal benefit or independent final-test gain.
The calibrated result still trails our historical Workbench score of 0.4349.
Rank 42 was observed at 2026-11-02 01:06:18 UTC; rank is not a permanent property.

The actual target moments for this candidate, rounded here for readability, were:

| CYP | OOF Pearson | Adjusted Pearson | Target mean | Target SD |
|---|---:|---:|---:|---:|
| CYP1A2 | 0.600211 | 0.792279 | 4.412 | 1.230409 |
| CYP2C9 | 0.714576 | 0.878929 | 4.830 | 0.967701 |
| CYP2D6 | 0.455017 | 0.755329 | 3.570 | 0.900000 |
| CYP3A4 | 0.821597 | 0.879109 | 4.880 | 1.118227 |

Use full-precision parameters and hashes from the
[evidence record](results/calibration-policy-20261102.json), not this rounded table.
The CYP2D6 override ignores its adjusted correlation when setting final moments.

## Internal evaluation and own-data calibration

Keep raw, historical and own-data variants separate. Historical calibration is
not a training objective or an automatic replacement for raw strategy scores.
Freeze a comparison's transform, cohort normalization scope and model predictions
before scoring; never fit transform parameters to its scoring labels.

Our three-rotation hit-expansion benchmark favored raw predictions. For example,
latest-selected raw MA-ST-RAE was 0.6418, versus 1.1382 under the common frozen
Workbench-reference transform. That common-reference diagnostic reused Workbench
correlation-derived target SDs across strategies; it is distinct from the
candidate-specific OOF correlations used for the official submission. Each internal
test cohort was normalized separately. A separate candidate-specific historical
diagnostic was also poor (1.1384), so this distinction does not reverse the finding.

This does not contradict the matched official result. The benchmark's chemical
selection resembles hit expansion, but sparse observed labels need not represent
the official potency distribution. Its 42 observed CYP2D6 labels average 4.983,
whereas the historical target mean is 3.57. Official blind labels remain unknown;
this comparison does not prove what their mean is. Preserve this limitation rather
than changing the split or fitting constants to make an observed ranking favorable.

Own-data calibration must use only our development labels/predictions, with its
transform family, fitting objective, regularization, fitting rows and clipping
declared. The completed postprocessing experiment used per-CYP ridge affine
calibration, fitting the two non-test groups for each rotation. Those groups also
selected that rotation's model checkpoints: this is selection-reused development
evidence, not a disjoint calibration sample. Its scoring group was excluded from
that rotation's fitting and selection. Do not pool other rotations' predictions
into that calibrator. Report prior adaptive exposure and sparse assay coverage.

A stricter independent assessment requires checkpoint selection and calibrator
fitting roles to be separated in the model-selection design. That can require
additional model fits; merely applying calibration to existing predictions does
not. Do not retrain for a requested postprocessing-only comparison without a
separately authorized change in scope. Existing saved Workbench folds, model
artifacts and historical evaluation contracts remain preserved.

## Reproducibility and submission checklist

1. Freeze candidate identity, checkpoints, fold assignments, model-combination
   weights and raw pIC50 predictions. Preserve raw output in its original file.
2. Save the OOF prediction and label sources, row masks/counts, correlations,
   historical constants, adjusted correlations and resulting target moments.
3. Record the complete cohort IDs, raw cohort mean/sample SD, transform code hash,
   final parameters and floor counts. Derive a new calibrated file directly from
   raw output; do not overwrite or calibrate an already transformed file.
4. Independently reproduce the equation and validate all 750 expected IDs/SMILES,
   four finite pIC50 columns, CSV schema and input/output SHA-256 values.
5. Record calibration provenance and evaluation roles beside every reported score.
   Checkpoint-selected OOF, OOF-tuned results and repeatedly selected strategies
   are development evidence. Conditional prediction bootstraps do not capture
   model or calibrator fitting uncertainty.
6. Upload only when authorized, respecting the live portal cooldown. Send the
   prediction CSV only, Open Code false, report link empty. Keep all code, reports,
   attribution and checkpoints private. Preserve receipts and timestamped score
   observations; this policy itself authorizes no further submission.

The historical preparation manifest remains an immutable pre-upload record. Its
`prepared_not_submitted` status is superseded operationally by the separate
authorized upload manifest and accepted receipt listed in the evidence record;
do not rewrite the historical preparation record to erase that sequence.
