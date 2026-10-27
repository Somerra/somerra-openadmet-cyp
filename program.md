# Regression autoresearch task

Use `autoresearch.py plan|run|reproduce|status|prepare`. The default path is the
frozen molecular regression task. Historical CPU/fold-based commands require the
`legacy` prefix; the previous fixed-series GPU comparison uses `hillclimb`.

## Current task

Keep the reviewed 60/20/20 molecular-parent assignments fixed. Their SHA-256 is
`9e69d2ec25402ad2c1d96b8f4169785926917af3db1531d552515b0167780318`.
Use the pinned 26-task pool and preserve record weights. Exclude validation,
internal-test and official-test parents from every training source; retain
analogues. Fit scalers and loss-normalization statistics on training rows only.

Validation selects checkpoints by point MA-ST-RAE. Score each selected checkpoint
on internal test to compare model strategies. Internal test is a development set;
repeated hill climbing is adaptive. Report point scores and official-style
bootstrap summaries separately. Only challenge evaluation supplies an official
blind-test score. Do not introduce another confirmation partition.

The adopted single-member reference scores 0.66080094 on validation and 0.65999603
on internal test, with its best checkpoint at epoch 69. Preserve its original
artifacts rather than retraining it for each candidate.

Initialize a task once using the prepared bundle, completed reference and existing
shared allocation ledger. `task.json` freezes data/evaluator hashes, runtime
versions, seed 42, batch 128, 100 epochs, the 200-epoch scheduler horizon and the
promotion rule. Changes to this contract require explicit task initialization.

## One trial

1. Choose a concrete model hypothesis within the authorized experiment scope.
2. Edit `MODEL_OVERRIDES` in `train.py` or provide `--model-config`. The maintained
   recipe is `configs/regression/incumbent.json`. Evaluation/data fields are rejected
   as model overrides. Keep source selection fixed for this model study.
3. Run `autoresearch.py plan --task PATH`. This does not train, reserve GPU time,
   create trial outputs or modify the registry.
4. Run `autoresearch.py run --task PATH --output-dir NEW_PATH --hours HOURS
   --hypothesis TEXT`. It trains one candidate, preserving inputs, implementation
   hashes, predictions and curves. No reference refit or final fit is appended.
5. The evaluator compares against the latest incumbent under a registry lock.
   Keep only a complete candidate with >=3% lower internal-test point MA-ST-RAE,
   no endpoint increase above 0.02, and a paired scaffold-bootstrap 95% upper
   candidate-minus-reference delta below zero. Otherwise discard; failures are
   recorded as crashes. All trial artifacts remain private under run directories.
6. Review the result, retain the best maintained recipe and select the next
   hypothesis. Do not infer permission for unbounded trials from this file.

Model families may evolve in `hillclimb_model.py` and their configuration in
`model_recipe.py`; the initial adapter supports a configurable Chemprop bond MPNN.
Data loading, target preparation, splits, metrics and checkpoint selection remain
fixed. Each trial snapshots the model configuration and implementation. Source
changes during a trial invalidate its comparison.

`reproduce --task PATH --trial SAVED_TRIAL --output-dir NEW_PATH` ignores the current
`train.py` overrides, verifies the saved task/model hashes and uses the same
training/evaluation functions. Reproduction never promotes a candidate. If model
implementation has changed, restore the recorded implementation before replaying.
Reproducibility does not imply bitwise equivalence of nondeterministic GPU kernels.

## Completed experiment: task normalization

Keep architecture, learning rates, data, record weights, auxiliary weight, MAE,
seed and epochs fixed. Compare the reference's normalization over observed labels
with equal task normalization based on full training-set observation counts.
For N training rows, T tasks and n_j observed labels for task j, use fixed scales
N/(T*n_j); each minibatch sums weighted masked errors times those scales and divides
by its row count. Averaging across rows recovers the mean of per-task weighted MAEs.
This avoids estimating label-frequency weights from validation/test or from each
minibatch. Zero-label tasks are rejected. All 26 tasks are available in this study.

The 2026-10-27 candidate completed 100 epochs and selected epoch 97: validation
MA-ST-RAE 0.77174024, internal test 0.73982985. The paired candidate-minus-reference
delta was +0.07983382, 95% CI [0.04825058, 0.11272308]. The registry discarded it
and retained the reference. This result is conditional on the single seed and
fixed training policy. Keep observation normalization in the maintained recipe;
the unsuccessful candidate remains reproducible from its private trial snapshot.

The prior interval-loss variant changed both loss shape and effective scored-task
weight because gradients vanish inside the confidence interval. Its worse observed
score does not isolate loss shape. The half-auxiliary-weight result was inconclusive
under paired resampling. Endpoint differences remain descriptive single-run results.
Use one seed; do not add seed ensembles or extra seed trials.

## GPU execution and persistence

Reuse the allocation-specific ledger initialized from the completed study; never
reset spending when changing worktree, output directory or coordinator. The worker
claims a matching reservation. Charge the full subprocess lifetime, including
startup, failures and recovery. An abandoned reservation retains its full charge
until reconciled. One recovery is permitted inside a trial's original allowance.

Cache graphs and use GPU training, TF32 matrix operations, full-FP32 evaluation and
four loader workers. The supervisor monitors progress, stops stalled/deadline jobs,
and preserves checkpoints and the committed curve prefix. Keep the recipe and
implementation unchanged while the trial is running. Use a persistent supervisor
for long runs. Budget bookkeeping supports execution; prioritize correct evaluation
and useful experiments over additional budget infrastructure.

## Final fitting and challenge alignment

`prepare` is separate from ordinary trials. It fits the selected single model for
its validation-selected epoch count on all eligible labeled data, restores internal
holdouts, retains official-test exclusions, and validates the 750-row prediction
CSV. It never uploads. Existing prepared reference predictions remain available.

The frozen audit reports median nearest-training distances of 0.53485 for internal
test and 0.50000 for official test against the same retained official training set.
The official-to-all-training distance is 0.41270; the older 0.5688 internal distance
is superseded. Chemical and potency-distribution mismatch remains. An official
submission supplies one calibration point; ranking correlation requires multiple
comparable models. Local improvement does not establish official-test improvement.

Follow `AGENTS.md`: prediction-only submissions, repository and implementation
private, Open Code false and Method Report Link empty, including third-party links.
A new explicit submission instruction is required; experiment authorization does
not authorize uploading. Preserve receipts and respect cooldowns. Attribution and
reproducibility evidence stay in private artifacts.
