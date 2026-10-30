# Three frozen representations on Workbench folds

All 15 GPU fits completed: CheMeleon, Mol-JEPA, and Atom-JEPA on the same five
saved `mt_aux8` folds, 4,905 molecules, four scored tasks and four auxiliary tasks.
The encoders remained frozen. There was no calibration, conformer averaging,
repeat-seed averaging, blind prediction export, or submission.

Mol-JEPA has the lowest pooled OOF error, but the paired intervals do not establish
a clear winner among these frozen representations. The historical end-to-end
Chemprop model is substantially stronger than every frozen probe in point scores.

## Matched results

Lower is better. These are pooled checkpoint-selected OOF development scores,
not official scores. Each molecule has one held-out prediction per representation.

| Representation | MA-ST-RAE | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---:|---:|---:|---:|---:|
| CheMeleon mean | 0.797392 | 0.904175 | 0.732909 | 0.989061 | 0.563424 |
| Mol-JEPA CLS | **0.778027** | **0.876324** | **0.726427** | 0.975791 | **0.533567** |
| Atom-JEPA scalar | 0.794890 | 0.882301 | 0.754471 | **0.959171** | 0.583615 |

| Fold | CheMeleon | Mol-JEPA | Atom-JEPA |
|---|---:|---:|---:|
| 0 | 0.851710 | 0.803609 | **0.793234** |
| 1 | **0.754594** | 0.769044 | 0.773270 |
| 2 | **0.803028** | 0.804714 | 0.834980 |
| 3 | 0.787215 | **0.765121** | 0.804692 |
| 4 | 0.816878 | **0.768682** | 0.789665 |

Mol-JEPA leads on three endpoint point scores and wins three of five folds against
CheMeleon, four against Atom-JEPA. Atom-JEPA has the lowest CYP2D6 point score.
The pooled score is recomputed on concatenated predictions; it is not the average
of the five fold scores, whose ST-RAE denominators differ.

## Paired uncertainty

We resampled the saved Workbench scaffold groups 2,000 times with seed 42, pairing
all models and endpoints and recomputing the evaluation-mean denominators in each
resample. Negative differences favor the first model.

| Comparison | Point difference | 95% percentile interval |
|---|---:|---|
| Mol-JEPA minus CheMeleon | -0.019365 | [-0.042913, 0.002171] |
| Mol-JEPA minus Atom-JEPA | -0.016862 | [-0.038712, 0.004804] |
| Atom-JEPA minus CheMeleon | -0.002502 | [-0.028483, 0.021376] |

Every interval includes zero. This is not evidence of equivalence, either.
Intervals condition on the fixed fitted OOF models; they do not include refitting,
seed variability, adaptive selection, or multiple-comparison correction. The
historical cross-fold connectivity exceptions also remain in this analysis.

## What was controlled

All fits used the same ordered rows, eight task columns, saved folds, training-only
feature/target normalization, training-fold-derived task weights, batch order for
common epochs, optimizer, regularization, and checkpoint-selection rule. There
were 6,525 observed scored labels and 17,500 auxiliary labels. Primary weights
were normalized inverse training-label counts; auxiliaries were weighted 0.3 each.
The [protocol](workbench-representations.md) specifies the loss denominator and
the eight-task validation criterion.

This isolates performance under a common frozen-head recipe, not intrinsic encoder
quality or a causal pretraining-architecture effect. CheMeleon's 2048-dimensional
features give its head 1,054,216 parameters; Mol-JEPA's 512-dimensional CLS features
give 267,784; Atom-JEPA's 256-dimensional scalar features give 136,712. Pretraining
and input modalities also differ. No Mol-JEPA token head, Atom-JEPA invariant head,
or encoder fine-tuning was tested here.

The pinned Workbench split is disjoint by its saved isomeric scaffold strings.
Two stereo-insensitive connectivities span folds: `LOUPRKONTZGTKE` (0/1) and
`MJIHNNLFOKEZEW` (0/4), covering four rows. These exact historical exceptions were
preserved for parity and recorded in the design; additional overlap is rejected.

## Checkpoints and cost

| Representation | Selected epochs, folds 0–4 | Stopped at epochs |
|---|---|---|
| CheMeleon | 4 / 4 / 5 / 4 / 6 | 44 / 44 / 45 / 44 / 46 |
| Mol-JEPA | 10 / 14 / 5 / 7 / 18 | 50 / 54 / 45 / 47 / 58 |
| Atom-JEPA | 25 / 35 / 11 / 23 / 48 | 65 / 75 / 51 / 63 / 88 |

Every fit stopped by patience 40 before the 400-epoch ceiling. Selection uses
weighted standardized validation MAE across all eight tasks, not scored-task
MA-ST-RAE. Validation selects the checkpoint and supplies OOF, so these are
development results. There is no independent internal-test estimate or official
test score for this study.

The H100 supervisor charged **0.020164 GPU-hours (72.59 seconds)** against the
0.75-hour series cap, including worker startup, cache verification, all fits,
prediction replay and final input checks. Frozen caches were reused; their
historical extraction costs are not included in this incremental charge. No new
allocation was created. The shared campaign remains below its 20-hour ceiling.

## Historical Chemprop context

The saved Workbench `mt_aux8` end-to-end Chemprop model scores **0.687887** raw OOF
MA-ST-RAE on these exact rows and folds, with endpoint scores 0.796948 / 0.606362 /
0.888277 / 0.459962 for CYP1A2 / CYP2C9 / CYP2D6 / CYP3A4. Its five saved OOF files
were checked against completion-record hashes and scored with the same evaluator.

This is useful context, not a fourth matched frozen-representation arm: its
encoder is trained, and its head, optimization recipe, and full-table task-weight
calculation differ. The 0.687887 number is neither the complete four-configuration
Workbench ensemble nor its calibrated official 0.4349 score.

These frozen probes do not justify replacing that baseline. A subsequent comparison
could test task-specific encoder adaptation on the same folds and targets, while
keeping this completed study fixed. Auxiliary-task benefit was not isolated here;
there is no matched no-auxiliary arm.

## Evidence

Training implementation commit: `75ac33e`. The private
[evidence snapshot](results/workbench-representations-20261030.json) records exact
metrics, all 15 fits, data/source hashes, per-fold weights, replay results,
bootstrap intervals, accounting, and the contextual Chemprop score.

Artifacts are under
`/mnt/data/somerra-openadmet-runs/workbench-representations-20261030`:
`design.json`, per-fit checkpoints/curves/OOF predictions, aggregate OOF CSVs,
`results.json`, `validation.json`, and the CPU `audit.py` script.

Independent CPU replay verified every checkpoint, exact molecule coverage,
training-only scalers, selected epochs, equal per-fold task weights and label
counts, and identical molecular order for common epochs across models. Maximum
GPU-to-CPU prediction difference was 3.8147e-6, below 1e-4. Full tests passed:
267 passed, 10 skipped with CUDA hidden; lint and formatting passed. No code,
checkpoints, reports, or prediction files were sent to the challenge.
