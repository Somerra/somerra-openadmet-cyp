# CYP recipe research and submission construction

Read `AGENTS.md` first. Keep source, models, data, and detailed research records
private. Research and submission construction are separate workflows. Neither
workflow uploads to the challenge or contacts organizers.

## Research owns the strategy

The coding agent chooses what to investigate, based on results and available
internal data. There is no fixed experiment queue or required round-robin order.
`campaign.py` belongs to historical workflows; this research path does not use it.
The runner handles execution, budgets, records, and evaluation. It does not choose
hypotheses or call another LLM service.

CYP1A2, CYP2C9, CYP2D6, and CYP3A4 each have an editable recipe at
`research/recipes/<CYP>/recipe.py` and an independent history, incumbent, W&B group,
lock, and budget under `runs/cyp-recipes-fold0/campaigns/<CYP>/`. The four starting
recipes share a Chemprop baseline; they are expected to diverge. The common
`research/cyp_recipe.py` is a starting template, not a shared mutable dependency.

Choose architecture, representation, auxiliary heads, task weights, loss,
optimizer, schedule, regularization, and training-row subsets independently for
each CYP. All 22 auxiliary heads are available; none is mandatory. Use the
currently prepared, available data pool. New downloads or altered data/fold
assignments are a separately declared data revision, not a matched strategy gain.
Declare local pretrained inputs in `ASSETS`; retain their hashes and provenance.
Do not use challenge leaderboard feedback, borrowed calibration constants, or
third-party predictions as training labels or selection objectives.

The shared executor currently supports trainable PyTorch models with replaceable
input adapters. A new model family that needs different execution mechanics can
extend the executor; retain its source snapshot with the trial. Training code is
recorded, not hash-frozen across a research campaign. Keep evaluation, identities,
and fold roles unchanged for comparable results. Backend extensions require
smoke/replay checks before full trials.

## Fixed comparison contract

Use the saved Workbench `mt_aux8` assignments, fold 0 for live development
validation. Inherit assignments before filtering heads or training rows. Exclude
validation and official blind parents from every training source, including
auxiliary observations. Fit target scalers only on the selected training rows.
Do not remove difficult validation rows. The other three scored CYPs are not
auxiliary heads in a specialist recipe; source-specific assays remain distinct.

Select checkpoints and each CYP's incumbent by **lower primary ST-RAE**. This is
the challenge's interval-aware error, not the recipe's training loss. Better
challenge performance means a lower error. The aggregate MA-ST-RAE is the equal
mean of the four incumbents and is absent until all four exist. Neither a change
in loss scale nor an improvement on another CYP can win this CYP's campaign.

Validation is repeatedly used for strategy and checkpoint selection: these are
**development results, not an independent held-out estimate**. Treat tiny gains
as uncertain; the agent may choose stability checks or matched controls rather
than repeatedly chasing small improvements. Failed, interrupted, and smoke trials
cannot become incumbents. Historical artifacts remain unchanged.

## Editable recipe interface

Each recipe is a self-contained Python file. Keep its implementation in that file
or installed versioned libraries; declare local pretrained files through `ASSETS`.
Record `STRATEGY` with a descriptive name and the model, loss, and data choices.
`SETTINGS` contains the resolved configuration, including batch size, maximum
epochs, patience, and primary/auxiliary weights. `task_weights` can override named
heads. `SMOKE_SETTINGS` supplies small model dimensions for CPU integration tests.

Required hooks:

- `build_model(n_tasks, settings)` returns a PyTorch model emitting standardized
  predictions, with the focal CYP first.
- `compute_loss(prediction, target, weights, context)` returns a finite scalar.
  Loss is fully editable: MAE, MSE, Huber, interval-aware objectives, or mixtures.
  Mask missing labels before nonlinear operations. Context supplies train-fitted
  mean/scale, raw focal confidence bounds for these rows, task names, phase,
  epoch, and settings. Bounds can be missing on auxiliary-only rows.
- `build_optimizer(model, settings)` and
  `build_scheduler(optimizer, settings, steps_per_epoch)` define optimization;
  the scheduler may be `None`, otherwise it steps after every optimizer update.

Optional hooks:

- `select_tasks(cyp, available)` keeps the focal head first and selects any subset
  of the available auxiliaries.
- `select_training_rows(frame, tasks, settings)` returns a boolean mask over the
  available training rows. It cannot change labels or add rows.
- `build_dataset(frame, standardized_targets, settings)` and
  `batch(dataset, indices, device)` replace the Chemprop representation. Batch
  returns model inputs and targets in exactly the requested row order. Inputs
  must use molecular features, never labels or confidence bounds. Store any
  learned feature transform in model state, fitted on training data only; do
  not fit it again on validation or blind data.

`train/loss` records the actual custom batch objective, averaged with observed
label counts as weights; `val/loss` evaluates that objective on the complete
validation predictions. Batch-dependent objectives need this distinction when
interpreting curves. `diagnostic_mae/*` remains a separate, standardized MAE
breakdown, not a claimed decomposition of an arbitrary custom loss. Compare loss
curves within the same recipe; use ST-RAE to compare different objectives.

## Agent experiment loop

1. Inspect this CYP's history, incumbent strategy, train/validation curves,
   failures, and remaining budget. Choose a hypothesis and record why it might
   improve the primary score. The agent chooses the next CYP and experiment.
2. Edit that CYP's recipe. Use a matched control when needed. Run a smoke test for
   a new loss, representation, or executor before spending the campaign budget.
3. Run one full trial only within the authorized allocation. Every attempt retains
   its recipe, runtime sources, hypothesis, strategy, settings, task/row choices,
   hashes, curves, and result. W&B receives private strategy metadata and metrics,
   never source or model uploads.
4. Inspect training/validation loss, ST-RAE, MAE, R², Spearman's rho, Kendall's tau,
   and task diagnostics. The runner selects by primary ST-RAE. Record the agent's
   conclusion and next direction with `note`, including failures and inconclusive
   outcomes. Keep promising alternatives; the best single-fold score need not be
   the recipe eventually selected for submission.
5. Continue while authorized budget remains. Do not reset spending by silently
   creating a new campaign or changing its frozen budget. Full trials remain
   disabled in the checked-in configuration. This task authorizes setup only.

```bash
bash scripts/run_cyp_research.sh prepare
bash scripts/run_cyp_research.sh status
bash scripts/run_cyp_research.sh trial --cyp CYP1A2 --smoke \
  --hypothesis "Verify the current recipe, loss, and telemetry"
bash scripts/run_cyp_research.sh note --trial /absolute/path/to/trial \
  --conclusion "Observed result and interpretation" \
  --next-hypothesis "The next strategy to test"
```

The same commands are available through `python autoresearch.py recipes ...`;
bare `prepare`, `trial`, and `status` also use CYP recipes. Historical workflows
remain explicit subcommands, including `molecular`, `briford`, and `legacy`.
The shell launcher selects `runs/neural-env` and the task-local W&B installation.

`configs/cyp-autoresearch.json` declares a **zero budget for each CYP**. Preparation
freezes the data, evaluator, and allocations in a new root. CPU smoke tests use
three epochs and tiny cohorts, cannot be selected, and do not consume a scientific
campaign allocation. Full fits require CUDA. The private W&B destination is
`jhtao-none/cyp-autoresearch`; offline and disabled modes retain local evidence.

Each trial retains `strategy.json`, `recipe.py`, `runtime/`, `job.json`,
`training-rows.csv`, `settings.json`, `scaler.json`, `events.jsonl`, checkpoints,
validation predictions, `result.json`, and append-only `notes.jsonl` when the agent
records its interpretation. Results link the strategy and recipe hashes. W&B
records strategy metadata, loss curves, all five challenge metrics, and selected
checkpoint summaries. Undefined correlations are null locally and omitted remotely.

## Separate submission path

Use `bash scripts/run_cyp_submission.sh plan|fit|build|status` to select the
existing environment, or `python submissions.py` / `openadmet-cyp-build` in an
activated environment.
`plan` requires **four explicit completed, non-smoke trial selections**; it does
not follow a moving incumbent. It freezes copies of their recipes, strategy
records and results, plus source/evaluator/runtime hashes and build policy.
It does not train. The default build budget is zero.

```bash
bash scripts/run_cyp_submission.sh plan --output runs/chosen-cyp-build \
  --select CYP1A2=/absolute/path/to/cyp1a2-trial \
  --select CYP2C9=/absolute/path/to/cyp2c9-trial \
  --select CYP2D6=/absolute/path/to/cyp2d6-trial \
  --select CYP3A4=/absolute/path/to/cyp3a4-trial
bash scripts/run_cyp_submission.sh status --root runs/chosen-cyp-build
```

When separately authorized, `plan --budget-gpu-hours N` allocates N/20
hours per CYP/fold fit, including retries; `--fit-seconds` caps individual attempts.
Process startup/shutdown can add a small amount of wall-clock overhead.
`fit` replays the four frozen recipes on all five **preserved mt_aux8 folds**,
using seed 42+fold, train-only target scalers, and primary-ST-RAE checkpoint
selection. It performs no strategy search. This is a declared specialist build,
not an exact reproduction of the historical four-configuration Workbench ensemble.

`build` requires all 20 complete fits, rechecks their saved validation predictions,
constructs complete checkpoint-selected OOF evidence, and equally averages the
five fold models for each endpoint on the official blind identities. It validates
a finite 750-row prediction CSV and saves a private build report. The initial
build uses raw predictions without calibration. OOF scores are development only.

This path never uploads: `open_source_code=false`, `model_report_link=""`, and
`submitted=false`. Challenge submission remains a distinct, explicitly authorized
action. Source, checkpoints, internal reports and attribution remain private.
