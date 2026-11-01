# CYP recipe research and submission construction

Read `AGENTS.md` first. Keep source, models, data, and detailed research records
private. Research and submission construction are separate workflows. Neither
workflow uploads to the challenge or contacts organizers.

## Research owns the strategy

The coding agent chooses what to investigate, based on results and available
internal data. Maintain an adaptive per-CYP experiment queue; there is no fixed sweep or required round-robin order.
`campaign.py` belongs to historical workflows; this research path does not use it.
The runner handles execution, budgets, records, and evaluation. It does not choose
hypotheses or call another LLM service.

CYP1A2, CYP2C9, CYP2D6, and CYP3A4 each have an editable recipe at
`research/recipes/<CYP>/recipe.py` and an independent history, incumbent, W&B group,
lock, and budget under `runs/cyp-recipes-val0-test2/campaigns/<CYP>/`. The four starting
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

## Scope of research conclusions

Every strategy result is specific to its recorded data, saved split, checkpoint
selection, and search history. The [first campaign learnings](../docs/autoresearch-learnings-20261101.md)
record 156 scored trials and selected-recipe repeat variability on train folds
1/3/4, validation 0, and adaptive scoring fold 2. These are candidate strategies,
not universal CYP defaults or independent estimates. When changing the split or
data, run fresh matched controls and label the new result separately. Report all
repeats alongside the best observed score; do not transfer an incumbent score to
a new evaluation contract or infer a causal effect from an adaptively selected winner.

## Fixed comparison contract

Use the saved Workbench `mt_aux8` assignments: **train on folds 1, 3, 4**;
**validate on fold 0**; **test on fold 2**. Inherit assignments before filtering
heads or training rows. Exclude validation, internal-test and official blind
parents from every training source, including auxiliary observations. Fold 2 was chosen because its parents are disjoint from
validation fold 0; saved folds 1 and 4 each share a parent with fold 0. Fail
preparation if either held-out cohort overlaps the other or the official blind
parents; never silently drop held-out rows to repair that overlap. Fit target
scalers only on the selected training rows.
Do not remove difficult validation rows. The other three scored CYPs are not
auxiliary heads in a specialist recipe; source-specific assays remain distinct.

Select checkpoints within each run by **lower validation primary ST-RAE**.
Score that fixed checkpoint on internal test, then select each CYP's strategy
incumbent by **lower internal-test primary ST-RAE**. This is
the challenge's interval-aware error, not the recipe's training loss. Better
challenge performance means a lower error. The aggregate MA-ST-RAE is the equal
mean of the four incumbents' internal-test scores and is absent until all four
exist. Keep validation and internal-test metrics explicitly separated. Neither a change
in loss scale nor an improvement on another CYP can win this CYP's campaign.

Validation selects checkpoints; internal test is repeatedly used to score and
select strategies. Both yield **development results**. Adaptive strategy search
on internal test means its score is not an independent or unbiased estimate. Treat tiny gains
as uncertain; the agent may choose stability checks or matched controls rather
than repeatedly chasing small improvements. A graceful time-cap stop finalizes the
best fully validated checkpoint, records `stop_reason=time_cap` and any partial
optimizer steps, and can be selected. A cap reached before any validated epoch,
a hard process timeout, a failure, or a smoke run cannot become an incumbent.
Historical artifacts remain unchanged.

## Queue and experiment handoff

On each completed experiment, verify its artifacts, score its fixed
validation-selected checkpoint on internal test, and record the interpretation.
Launch the next eligible experiment within the authorized allocation without
waiting for every concurrent experiment to finish. A monitor alerts the coordinator;
the coordinator chooses and launches the next hypothesis.

At every launch, refresh the remaining per-CYP queue using completed results:
add ideas, reprioritize promising hypotheses, and retire weak or redundant ones.
Retain queue revisions, evidence, parent trials, and reasons for changes. Report
technical blockers explicitly; a saved queue is not an active scheduler.

Treat representation as a first-class axis: molecular graphs,
fingerprints/descriptors, available pretrained embeddings such as Mol-JEPA, and
combinations. Verify asset provenance and adapter smoke/replay checks before use.
First profile batch size with the model and learning-rate settings held fixed;
choose the largest batch that improves measured throughput under actual
concurrency. Then freeze batch size and tune learning rate separately per CYP.
Repeat throughput profiling when a new representation changes memory or compute.
Record changed axes explicitly so batch and learning-rate effects are not confounded.

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

Training metrics use the predictions collected before each training update, so
there is no extra training-set inference pass. They reflect a changing model in
training mode, not a final-epoch evaluation. MAE, R² and rank metrics use all
observed primary training labels. Training ST-RAE uses only rows with recorded
intervals; `train/ST-RAE_n` and `train/missing_intervals` expose that cohort. It is
null when the interval baseline is undefined. Validation retains the strict full
cohort and interval checks used for selection. Loss-context bounds and scales are
cached once and indexed in the same row order as recipe batches.

## Agent experiment loop

Work in a dedicated `codex/` research branch or worktree. Preserve unrelated
changes. Read this program, the selected CYP recipe, and the shared executor and
evaluator before editing. The first authorized full trial for each CYP should
run its starting recipe unchanged to establish a baseline. CPU smoke scores are
not baselines. On resume, reuse the recorded baseline and history; do not restart
spending or rerun an unchanged baseline without a reason.

1. Inspect this CYP's history, incumbent strategy, train/validation curves,
   failures, and remaining budget. Choose a hypothesis and record why it might
   improve the primary score. The agent chooses the next CYP and experiment.
2. Start from that CYP's incumbent recipe snapshot by default. Record the parent
   trial ID in the hypothesis (or state that this is the baseline). After a failed
   or non-improving experiment, restore only that CYP's working recipe from the
   incumbent before trying a new idea. If no baseline has succeeded, fix or retry
   the baseline first and record any required correction. An intentional branch from another saved
   strategy is allowed; name that parent and explain why. The runner preserves
   snapshots but does not restore working files automatically. Never reset the
   whole repository or delete trial evidence to discard an experiment. If the
   experiment also changed shared runtime code, retain or revert those specific
   edits deliberately and record the effect on other CYP comparisons.
   Edit that CYP's recipe. Use a matched control when needed. Run a smoke test for
   a new loss, representation, or executor before spending the campaign budget.
3. Run one full trial only within the authorized allocation. Every attempt retains
   its recipe, runtime sources, hypothesis, strategy, settings, task/row choices,
   hashes, curves, and result. W&B receives private strategy metadata and metrics,
   never source or model uploads.
4. Inspect training/validation loss, ST-RAE, MAE, R², Spearman's rho, Kendall's tau,
   and task diagnostics. After fixing the validation-selected checkpoint, score
   it on internal test and rank strategies by primary internal-test ST-RAE under
   the clarified policy below. Record the agent's
   conclusion and next direction with `note`, including failures and inconclusive
   outcomes. In the note, mark a promoted full trial as `keep`, a completed
   non-improving full trial as `discard`, and a failed/timeout attempt as `crash`;
   record interruptions and smoke checks separately. These are interpretations
   of the registry's status and promotion fields, not a second selection rule.
   Inspect `worker.log` on failure, fix a clear bug or abandon the hypothesis,
   and retain every attempt and its compute charge. Prefer simpler strategies
   when evidence is comparable; the score incumbent remains strictly defined,
   and an explicit later submission selection can choose a simpler alternative.
   Keep promising alternatives; the best single-fold score need not be
   the recipe eventually selected for submission.
5. Continue choosing and running experiments while authorized budget remains,
   without asking for permission between trials. Stop when interrupted, blocked,
   or out of allocation. Do not reset spending by silently
   creating a new campaign or changing its frozen budget. Full trials remain
   disabled in the checked-in configuration. Use a separately authorized campaign
   allocation; never infer permission for unbounded trials from this program.

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
only the explicit `recipes` prefix selects CYP recipes through that entrypoint.
Bare commands, empty argv and `--help` retain their historical behavior. Historical
workflows are also available through `molecular`, `briford`, and `legacy`.
The shell launcher selects `runs/neural-env` and the task-local W&B installation.

`configs/cyp-autoresearch.json` declares a **zero budget for each CYP**. Preparation
freezes the data, evaluator, and allocations in a new root. This revision uses
`runs/cyp-recipes-val0-test2`; never relabel an old fold-0-only campaign or edit its
manifest in place. The manifest records fold roles and hashes `internal-test.csv`
as well as train/validation data. Training only loads train and validation tables. CPU smoke tests use
three epochs and tiny cohorts, cannot be selected, and do not consume a scientific
campaign allocation. Full fits require CUDA. The private W&B destination is
`jhtao-none/cyp-autoresearch`; offline and disabled modes retain local evidence.
If W&B privacy cannot be verified, no online run is initialized. Privacy, auth,
or connection failures fall back to the local event journal and retain a
`tracking-error.json` diagnostic; they do not discard scientific work.

Each trial retains `strategy.json`, `recipe.py`, `runtime/`, `job.json`,
`training-rows.csv`, `settings.json`, `scaler.json`, `events.jsonl`, checkpoints,
validation predictions, `result.json`, and append-only `notes.jsonl` when the agent
records its interpretation. Results link the strategy and recipe hashes. W&B
records strategy metadata, loss curves, all five challenge metrics, and selected
checkpoint summaries. Undefined correlations are null locally and omitted remotely.

## Internal test scores strategies (policy clarified 2026-11-01)

| Partition | Role | Selection metric |
| --- | --- | --- |
| Training: folds 1, 3, 4 | Fit model weights and preprocessing | Editable training objective |
| Validation: fold 0 | Monitor training, early stopping, choose the checkpoint within a run | Primary validation ST-RAE, minimize |
| Internal test: fold 2 | Score each fixed checkpoint and rank strategies across runs | Primary internal-test ST-RAE, minimize |

After a full trial completes, freeze its validation-selected checkpoint and score
that checkpoint on the entire internal-test cohort. Score each eligible strategy,
not just the strategy with the lowest validation score. Internal-test feedback
may guide the agent's next hypothesis and determines each CYP's strategy incumbent.
This step does not close the campaign; continue within the authorized allocation.

Do not compare multiple epochs on internal test or choose a different checkpoint
after seeing its score. Do not use internal-test labels to fit model weights,
scalers, learned representations or calibration parameters. Strategy search may
compare choices fitted on eligible training data; keep data preparation and the
scoring function fixed. Smoke, failed or unscored trials cannot win the strategy
comparison. Retain recipe/runtime/checkpoint hashes, selected epoch, validation
metrics, internal-test predictions and all five challenge metrics per trial,
with distinct validation and internal-test fields in local records and private W&B.

Because internal test guides repeated strategy selection, call these **adaptive
development scores**, never an unbiased or independent held-out estimate. Disclose
historical use of the molecules as well. The official competition test remains
the final external evaluation; its labels and leaderboard feedback do not enter
autoresearch. The internal score describes the three-fold-trained strategy trial,
not the later five-fold submission ensemble.

Use `python -m somerra_openadmet.cyp_strategy trial|score|status` for adaptive
research. `trial` runs a GPU fit then scores its fixed validation checkpoint;
`score` also accepts existing registered full trials. Per-trial `internal-test/`
artifacts freeze the checkpoint before scoring. `strategy-registry.json` ranks
strategies by internal-test ST-RAE without closing the campaign. The original
`registry.json`, legacy `status`, and legacy W&B `best/*`/`campaign/*` summaries
remain validation-only diagnostics to preserve baseline runtime replay hashes;
they are not the strategy-selection authority. Historical closed campaigns stay
closed and historical trial records are not relabeled.

## Alignment with Karpathy's autoresearch

Reviewed against upstream [README](https://github.com/karpathy/autoresearch/blob/master/README.md)
and [program](https://github.com/karpathy/autoresearch/blob/master/program.md) on
2026-10-31 (Git blob IDs `953ea55d5599c45c1be7dad93ec03e47dfa7df9d` and
`dea9bcc0174f1502d0ba64000b94b81ba605855b`, respectively). This is an adaptation
of the agent-directed experiment loop, with these explicit correspondences:

| Upstream role | CYP adaptation |
| --- | --- |
| Human-maintained `program.md` directs the agent | Root `program.md` routes here; the agent chooses experiments within the user's scope. |
| Editable `train.py` | One self-contained recipe per CYP; architecture, representation, loss and optimization are editable. Shared runtime extensions require recorded provenance and smoke/replay checks. |
| Fixed preparation and evaluation | Saved Workbench assignments, prepared data hashes and fixed evaluator hashes; validation ST-RAE selects the checkpoint; internal-test ST-RAE ranks strategies under the clarified policy (runtime update pending). |
| Baseline, experiment, keep/discard, repeat | Establish a full baseline per CYP, inspect every result, retain or restore the working recipe, and record the next hypothesis. |
| Git commits plus `results.tsv` | Recipe/runtime snapshots and hashes, per-CYP `registry.json`, trial `result.json` and `notes.jsonl`; local records remain authoritative if W&B fails. No additional TSV is required. |
| Run until interrupted | Continue within the authorized per-CYP allocation. Checked-in zero allocations disable full trials; user-authorized campaign configurations supply budgets. |

The compute protocol is deliberately different. Upstream measures 300 seconds
of training excluding startup/compilation, then evaluates the final model.
Our configured full-trial cap is 1,800 seconds, clipped to remaining allocation;
worker preparation and epoch evaluations consume that cap, with 15 seconds
reserved for finalization. Parent process timeouts include a small grace period,
and actual process time is charged. Epoch limits and early stopping can finish
a trial sooner; the best fully validated checkpoint supplies its score. Thus
these are **capped-cost development trials, not equal-training-time experiments**.
For matched comparisons, inspect the recorded cap, elapsed time, completed epochs,
stop reason and hardware; flag a budget-clipped trial. Do not describe scores from
different compute conditions as a controlled architecture comparison. Changing
the timing or checkpoint-selection protocol is a declared experimental revision.

Four independent objectives, private W&B diagnostics and a separate frozen-recipe
submission builder are challenge-specific additions. Research uses available internal data; our additional test fold scores strategies after validation checkpoint selection
and is adaptively reused during research. Upstream alignment does not authorize access to official
competition test labels or erase historical use of the internal holdout.

## Separate submission path

Use `bash scripts/run_cyp_submission.sh plan|fit|build|status` to select the
existing environment, or `python submissions.py` / `openadmet-cyp-build` in an
activated environment.
`plan` requires **four explicit completed, non-smoke trial selections**; it does
not follow a moving incumbent. It freezes copies of their recipes, strategy
records and results, plus source/evaluator/runtime hashes and build policy.
Declared pretrained assets are verified against the selected strategy before
launch and each fit must retain the exact selected asset hash mapping.
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
Failures and interrupted reservations remain charged. If that allocation is
exhausted, an explicitly authorized retry can add compute without changing the
frozen recipe or erasing charges:

```bash
bash scripts/run_cyp_submission.sh allocate --root runs/chosen-cyp-build \
  --cyp CYP1A2 --fold 0 --additional-gpu-hours 0.5 \
  --reason "Explicitly authorized retry allocation"
```

This command only appends a plan-bound allocation record; it does not launch.
Do not run it without an authorized budget increase. `status` reports allocation
adjustments and the existing per-fit registries retain all charged attempts.
`fit` clears the research-only test role and replays the four frozen recipes on
all five **preserved mt_aux8 folds**, training on four and validating on one,
using seed 42+fold, train-only target scalers, and primary-ST-RAE checkpoint
selection. It performs no strategy search. This is a declared specialist build,
not an exact reproduction of the historical four-configuration Workbench ensemble.

`build` requires all 20 complete fits, rechecks their saved validation predictions,
constructs complete checkpoint-selected OOF evidence, and equally averages the
five fold models for each endpoint on the official blind identities. It validates
a finite 750-row prediction CSV and saves a private build report. The initial
build uses raw predictions without calibration. OOF scores are development only.
Replay defaults to the device type and float32 policy recorded by each fit.
CUDA fits therefore require CUDA for the default build; there is no silent CPU
fallback. `build --device cpu` explicitly requests a cross-device parity check,
with the same strict prediction and score tolerances. A mismatch stops the build;
it is not accepted by widening tolerances. GPU replay remains to be exercised on
a CUDA host.

This path never uploads: `open_source_code=false`, `model_report_link=""`, and
`submitted=false`. Challenge submission remains a distinct, explicitly authorized
action. Source, checkpoints, internal reports and attribution remain private.
