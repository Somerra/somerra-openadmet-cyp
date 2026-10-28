> **2026-10-28 user direction:** Scigantic model reconstruction and experiments
> are paused. Preserve data splitting independently of model work. See
> [the frozen split archive](evaluation/splits/README.md). Earlier Scigantic
> experiment authorizations below are historical and do not resume that work.

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
shared allocation ledger. Preserve fitted implementation hashes from the reference
manifest. Changed model/trainer code requires explicit `init
--check-model-compatibility`: reload the saved checkpoint and reproduce all saved
internal-test predictions within 1e-4. Record original and compatible source
identities, artifacts, recipe, bundle, runtime and error; never relabel old training
as having used current code. Final fitting accepts only the original implementation
or this exact compatibility record. The check establishes checkpoint inference
parity, not equivalence of retraining after training-semantic changes. `task.json` freezes data/evaluator hashes, runtime
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

The 3% gain is a practical-effect threshold, separate from the paired interval.
An adopted reference is the starting benchmark, not a promoted winner and not a
claim that it passed a comparison against itself. Every subsequent replacement
uses the same gate. A candidate with a 2% gain and a paired interval below zero is recorded as a
discard under this policy; its results remain available. The paired interval
conditions on fixed fitted models; it does not measure seed variability or correct
for adaptive search. It therefore does not replace the practical-effect threshold.
Changing that threshold requires a new task contract, not a mid-series edit.


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

`scored_endpoint_weight` is a separate model knob (default 1.0, positive and at
most 16). It multiplies only the first four scored-task losses, preserving
auxiliary weights, record weights and the loss denominator. A future comparison
can set it to 2.0 with everything else at the reference. Equal task normalization
is not evidence for or against this hypothesis. Doubling scored weight and halving
auxiliary weight have the same relative task balance but different overall loss
scales; do not treat them as identical optimization experiments. The new knob is
implemented but has not been evaluated.

Old manifests/checkpoints missing the knob retain its default of 1.0. The
compatibility handling changes frozen source hashes: continue existing series on
their recorded revision, and initialize a new task directory before using this
revision. Reuse the same bundle, reference artifacts and allocation ledger; retain
the old registry and trials as historical evidence. This does not reset spending,
change the split, or require a reference refit. See README's private artifact
preservation requirements before moving environments.

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

## Authorized scaffold ensemble experiment (2026-10-28)

The user requested scaffold splits and ensembling. Use the separate
`autoresearch.py scaffold prepare|run|status` workflow described in
`docs/scaffold-ensemble.md`; this is an explicit new protocol, not a modification
of the frozen molecular task above. The user now authorizes a fold ensemble for
this experiment. Retain seed 42 and the depth-four reference recipe across all five
folds. Whole scaffolds across all sources rotate through train, validation and
held-out roles; average only official blind predictions. Keep individual held-out
OOF scoring separate from ensemble evaluation. Use the remaining existing ledger,
with a 0.65-hour series cap. Prepare privately; a new instruction is needed to submit.

The subsequent review direction prioritizes a single final model with full eligible
training coverage. Retain the completed scaffold ensemble as comparison evidence;
it is not the preferred submission. See `docs/scaffold-ensemble.md` for the proposed
80/20 scaffold development split followed by a fixed-epoch all-data refit. This
proposal does not itself launch additional training or authorize a submission.
