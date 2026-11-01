# Default regression workflow: Workbench scaffold folds

Adopted on 2026-10-29, until the user changes direction. Future regression
submission candidates use the preserved Workbench scaffold assignments. The main
experimental variables are model architecture, task-loss weights, and the weights
used to combine models. Calibration learned from our own OOF predictions and
official training labels is a separate, declared experiment axis. Prefer developing
CYP-specific models with auxiliary prediction tasks over relying exclusively on
one model for all four endpoints.

This default follows our [October 28 reproduction](briford-run-20261028.md), which
scored 0.4349 official MA-ST-RAE. That result motivates the choice; it does not
isolate a causal benefit from the split strategy. The historical Workbench recipe
is the reference here, not the changing upstream default. The user separately authorized historical Workbench calibration for the next
candidate on 2026-11-01; see
[calibration experiments and eligibility](#calibration-experiments-and-eligibility).

## What is fixed, and what can change

| Keep fixed for a matched comparison | Experiment with |
|---|---|
| Source snapshots, row identities, labels, and source-specific task definitions | Molecular encoder and prediction-head architecture |
| Saved scaffold/fold assignments for the selected data bundle | CYP-specific versus shared prediction heads |
| Four training folds and one checkpoint-selection/OOF fold | Relative weights of scored and auxiliary losses |
| Training-only fitted target scalers; masked missing labels | Weights combining completed model configurations |
| Scoring on official direct-inhibition labels, with the shared evaluator | Regularization and training policy, as separately declared changes |
| Official blind identities and prediction-only submission policy | Auxiliary-task selection within a declared, fixed bundle |
| Raw predictions in pIC50 units retained as the baseline | Own-data calibration as a separate axis; historical Workbench calibration for the next candidate |

Changing a data source or regenerating folds is a separate protocol change, not
an architecture improvement. Record any training-policy change explicitly; a new
encoder may need a different optimizer, but changing both does not isolate the
encoder's effect. Do not silently compare the old Mol-JEPA Butina study or the
60/20/20 molecular task against these folds as if they were matched experiments.

## Data bundles and tasks

These are four model views of the source measurements, not four independent
datasets. Source provenance and joins are described in
[briford reproduction](briford-reproduction.md#data-and-preserved-identity).

| Saved bundle | Records | Outputs | Role |
|---|---:|---:|---|
| `union18` | 31,670 | 18 | Four official pIC50 tasks, four official single-concentration log2fc tasks, five ChEMBL CYP potency tasks, five Veith response tasks |
| `mt_aux8` | 4,905 | 8 | Four official pIC50 tasks and four official log2fc tasks |
| `d6_isoform6` | 23,264 | 6 | CYP2D6 official pIC50, log2fc, TDI-condition pIC50, Emax, ChEMBL potency, and Veith response |
| `d6_single1` | 1,493 | 1 | CYP2D6 official pIC50 only |

Only the four official direct-inhibition pIC50 endpoints determine the regression
score. Auxiliary measurements keep separate output heads; they are not substitute
labels for official potency. A specialist can predict one scored endpoint and
several auxiliaries. Specialization does not mean single-task training.

The original preparation drops rows with no labels for a bundle, then constructs
its folds. It retains one official blind connectivity with a Veith auxiliary
measurement for historical parity, recorded in the preparation manifest; no
official blind potency labels are available or used. Preserve and report this
exception when comparing against that reference rather than silently changing it.

## Exact split construction and reuse

The implementation is `data_sources/workbench.py::scaffold_folds`. It is checked
against Workbench revision `8c72c7559471febaf0962fd551b4eaab8db3a136` during
preparation. The union18 task definition comes from revision
`971f6a7f24d86e59bab219347c3e14c006ae4e35`.

1. Read the bundle's ordered, prepared SMILES rows.
2. Extract each RDKit Bemis–Murcko scaffold and convert it to canonical SMILES.
   Invalid structures fail; there is no random-split fallback. Empty scaffolds
   also form a group under this implementation.
3. Number scaffold groups in order of first appearance.
4. Shuffle the group IDs with NumPy `default_rng(42)`.
5. Visit each whole group and assign it to the fold with the fewest rows so far.
   Ties go to the lowest fold index. There are five folds, numbered 0–4.

This balances row counts approximately, not labels or molecular similarity to the
official blind set. Row order, bundle membership, and RDKit behavior can affect
the result. A seed alone is not sufficient to preserve the split.

The default RDKit scaffold SMILES retains stereochemical distinctions. Preflight
of the [matched representation study](workbench-representations.md) found two
connectivities spanning folds in `mt_aux8` (four rows). Exact saved-scaffold
disjointness does not guarantee stereo-insensitive parent disjointness. The study
records these inherited exceptions and keeps the original assignments for parity.

**Reuse saved assignments, not just the splitter.** The reference artifacts are:

```text
/mnt/data/somerra-openadmet-runs/briford-20261028/
  data/{union18,mt_aux8,d6_isoform6,d6_single1}.csv
  data/preparation.json
  contract.json
  fits/<member>/<fold>/
```

Each CSV records `molecule_name`, `smiles`, `connectivity`, `scaffold`, `fold`, and
task columns. A new experiment should record the source manifest and CSV hashes,
then verify identities and assignments against them. Never overwrite the reference.

The portable [artifact hash record](results/workbench-protocol-artifacts.json)
pins SHA-256 values for the four prepared CSVs, preparation metadata, data manifest,
and amended training contract using paths relative to the run directory. It also
records row/fold counts and the splitter revisions/runtime versions. The absolute
directory above is only a locator: restored artifacts must match these hashes
before reuse. These are the completed run's bytes, not hashes of a new preparation
from mutable public sources.

Each bundle has its own partition: fold 0 in union18 is not necessarily fold 0 in
mt_aux8. Within a bundle, every model family uses the same assignments, including
all auxiliary observations for each row. Do not split an auxiliary table
independently and accidentally train on a held-out molecule's other assays.

For a new specialist derived from an existing bundle, inherit that bundle's fold
column before masking unused outputs or dropping rows without selected labels.
Do not rerun scaffold assignment after filtering. Record the parent bundle, row
mask, task mask, and per-fold label counts; reject an untrainable head. A matched
Chemprop control must use the same filtered rows and inherited folds. Extending
specialists to other CYPs is new work, not part of the historical 20-model run.

## Training, validation, and test roles

For fold k, train on folds other than k (approximately 80%) and use fold k
(approximately 20%) for checkpoint selection. Repeat for all five k values.
The reference initializes fits with seeds 42+k, fits target normalization on the
training rows, and applies that scaler to validation. Its weighted MAE
`val_loss` selects the best checkpoint: maximum 400 epochs, patience 40.
The selection loss includes the selected auxiliary tasks; it is not MA-ST-RAE.

Predict each held-out row using only its corresponding selected checkpoint.
Concatenating these predictions produces one OOF prediction per eligible row per
configuration. Report MA-ST-RAE, each endpoint's ST-RAE, label counts, and per-fold
results using official direct-inhibition labels. Retain curves and selected epochs.

The same held-out fold selects the checkpoint and supplies OOF scores. Consequently
OOF is **development evaluation**, not an untouched internal test. Checkpoint
selection, architecture/task-weight search, and OOF tuning of combinations or
calibration all add adaptivity. Submission construction retains four training
folds per fit to match Workbench training coverage. The CYP recipe research
revision below uses three training folds and a separate internal test. This is a deliberate
comparability tradeoff, not a claim that checkpoint-selected OOF is unbiased or
that its ranking will transfer to the leaderboard. No independent internal
held-out estimate exists for this checkpoint-selected OOF workflow.

Every comparison must distinguish these results, even when some are unavailable:

| Result | What must be reported |
|---|---|
| Raw checkpoint-selected OOF | Fixed combination weights; per-fold/per-endpoint scores and checkpoint-selection rule; development only |
| OOF-tuned combination or calibration | Separate score, fitting rows, search scope and selected parameters; reused development data, not a held-out gain |
| Independent held-out estimate | `Not available` unless a separately recorded design excludes scoring rows from training and all selection/fitting steps |
| Official score | Exact submission receipt and pipeline; `Not submitted` for a candidate without its own official evaluation |

The three-train/one-validation/one-score rotation (score k, select on (k+1)%5)
is an optional, separately declared follow-up using the same scaffold assignments.
It changes training coverage from about 80% to 60% and requires matched control
fits. It is not silently imposed by this documentation. Such OOF scores avoid
checkpoint selection on the scoring fold, but repeated architecture or weight
search on them still makes them development results. An independent estimate
requires the entire selection process to exclude the scoring data.

The 750 supplied blind molecules remain the official external test; their labels
are unavailable locally. Only the challenge supplies an official score. Neither
the observed local-to-official error gap nor the historical ranking reversal
establishes that this split strategy or any particular correction will fix transfer.

When comparing candidates, pair predictions by molecule and endpoint within the
same parent bundle and fold. Across bundles, join OOF predictions by identity,
not fold number; there is no shared global held-out fold. OOF averaging combines
one held-out prediction per configuration, whereas blind inference averages all
five fold models per configuration. Those are different evaluation objects.

## Two different kinds of weights

**Task-loss weights** affect training. In the historical union18 and mt_aux8
recipes, each scored task has weight proportional to inverse label count,
normalized so the four scored weights average one. Each auxiliary has weight
0.3. These fixed weights are calculated during preparation from the full member
table's label counts, including held-out label availability; target scalers are
still fitted on training rows only. CYP2D6 specialist weights are 1 for its scored
task and 0.3 per auxiliary. Preserve those recorded weights when reproducing the
historical fits; do not relabel their provenance.

For **new specialists and reweighting experiments**, compute any count-derived
weights from that fit's training rows only, after the declared row/task mask.
Normalize inverse counts over the selected scored tasks to mean one; a single
scored task therefore has weight 1. Keep auxiliary weight 0.3 initially and reject
zero-label training heads. Record each fold's counts and resulting weight vector.
Apply the same rule to the matched Chemprop control. Moving from full-table to
training-fold counts is a declared change from the historical baseline, not a
free architecture gain. User-chosen fixed weights do not require label counts.

Changing a loss weight can also change gradient scale and which checkpoint wins.
Record the masking rule, target scales, reduction denominator, and validation
selection loss along with the weight vector. A weight of 0.3 is not a claim that
auxiliaries contribute 30% of the total loss: label frequencies and errors matter.

**Model-combination weights** affect inference. The reference first averages five
fold predictions equally within each configuration, then combines configurations:

| Endpoint | union18 | mt_aux8 | d6_isoform6 | d6_single1 |
|---|---:|---:|---:|---:|
| CYP1A2, CYP2C9, CYP3A4 | 0.5 | 0.5 | 0 | 0 |
| CYP2D6 | 0.25 | 0.25 | 0.25 | 0.25 |

Thus those first three endpoints use ten networks each; CYP2D6 uses twenty.
Five fold fits are not five repeat seeds on identical training data. There is no
additional all-data refit in this protocol. Use these combination weights as the
baseline, while treating alternative endpoint-specific model weights as an
explicit experiment. Weights selected on OOF must be reported as tuned on that
development set, not independently validated gains.

## Calibration experiments and eligibility

**Current direction (2026-11-01): prepare the next regression candidate with
historical Workbench calibration.** The user explicitly lifted the project hold
on borrowed constants. Apply the transform separately per CYP after fold/model
averaging: `max(1, target_mean + (p - mean(p)) * target_sd / sample_sd(p))`.
Use historical population moments and the model's OOF Pearson correlation times
1.32 / 1.23 / 1.66 / 1.07 for CYP1A2 / CYP2C9 / CYP2D6 / CYP3A4, respectively.
Retain the CYP2D6 target mean 3.57 / SD 0.90 override. Do not silently clamp invalid
correlation extrapolations. Cohort prediction moments use unlabeled predictions;
this is a transductive, leaderboard-informed transform, not training-only calibration.

Retain raw predictions after the declared averaging, exact input/transform hashes,
and calibrated output separately. Use a new destination; never overwrite prior
artifacts or infer upload authorization from a manifest. The historical
`autoresearch.py briford predict` exporter already implements this formula.
This direction authorizes preparation, not a new submission or training campaign.
Prediction-only metadata, cooldowns, and receipt preservation remain mandatory.

The completed frozen-representation study remains uncalibrated. Its raw comparison
answers which frozen representation worked under the shared training recipe; do
not relabel its results or require retroactive calibration to complete that study.
Training-target normalization and conversion back to pIC50 remain necessary.
Own-data calibration is a separate experiment axis, not a mandatory promotion step.

For an own-data calibration experiment, hold base predictions and model-combination
weights fixed and compare raw predictions against an endpoint-specific calibration
fitted only from our own OOF predictions
and observed official training labels (including moments calculated from those
labels). Declare the transform family, fitting objective, regularization, fitting
rows, parameters, and any clipping rule before scoring it. Do not use borrowed
OOF multipliers, population moments, or the CYP2D6 mean/SD override. Auxiliary
assay values are not official-label population moments.

Fitting a calibrator on all OOF rows and scoring those same rows is an in-sample
postprocessing result; label it as such. Excluding a scoring fold from calibrator
fitting avoids direct reuse at that layer, but does not make existing base OOF
predictions independent: the base models for other folds can have trained on the
scoring fold's labels, and its own checkpoint was selected on that fold. A clean
held-out calibration comparison requires the complete base-model/selection/
calibration pipeline to exclude the scoring data. Do not call a cheap re-fit over
the existing OOF table an independent estimate. No such estimator is implemented
by this protocol; implementing and validating a calibrator remains separate work.

Record any own-data calibration as a separate variant alongside raw output.
Compare raw and declared calibrated variants on the same rows. Preserve historical
results with their original model, population, and transform provenance.
The 0.6284 raw OOF score and 0.4349 calibrated official score use different
populations and pipelines. Their difference does not measure calibration's
contribution, and transferring another model's local-to-official gap cannot
establish a roughly 0.3 calibration benefit. A matched raw-versus-calibrated
evaluation on the same scoring set is needed.

**Eligibility and authorization are distinct.** The question raised in
[PR #12](https://github.com/Somerra/somerra-openadmet-cyp/pull/12#discussion_r4406685923)
has no recorded organizer clarification. The user's 2026-11-01 direction lifts
our project hold; neither that direction nor historical portal acceptance proves
organizer-specific clearance. Keep this limitation and borrowed provenance explicit.
No organizer contact or new upload is authorized by this protocol. The separate
hold on third-party Scigantic reference-prediction reuse is unchanged.

## Next Mol-JEPA comparison

Use these bundles and inherited folds, with auxiliary outputs and masked losses,
to compare Mol-JEPA against matched Chemprop controls. Start with the existing
CYP2D6 specialist bundle: it already supplies a concrete specialist reference and
six tasks. Use the new training-fold count rule and default fold-combination
weights initially, and score raw predictions first. Then
compare architecture and task weighting separately before extending specialists
to other CYPs or changing the final model combination.

The previous Mol-JEPA submission used five Butina folds, four scored tasks only,
one configuration, and fixed 300-epoch fits. It does not implement this protocol.
The current representation trainer needs a bundle/fold adapter, auxiliary-head
support, and validation-selected checkpointing before it qualifies as a matched
Workbench experiment. No new fit or submission is launched by this documentation.

## Entrypoint and evidence

The CYP recipe workflow now uses `autoresearch.py recipes prepare|trial|test|status|note`
for agent-directed strategy search. Each CYP has an independent recipe,
budget and history; architecture, task selection, loss and training policy can
change. The current research revision uses saved `mt_aux8` folds 1, 3, and 4 for training,
fold 0 for training monitoring and checkpoint selection, and fold 2 for internal-test
strategy scoring and selection (policy clarified 2026-11-01).
Both held-out folds and official blind parents are excluded from every training
source. The original fold IDs are preserved; this is a new research contract in
`runs/cyp-recipes-val0-test2`, leaving the old four-training-fold artifacts intact.

After each eligible trial, score its fixed validation-selected checkpoint on
internal test. Lower primary internal-test ST-RAE ranks strategies independently
for each CYP; its feedback may guide subsequent strategy experiments. Validation
never replaces internal test as the strategy-ranking score. Internal test never
selects checkpoints or fits model weights, scalers or calibration parameters.

Repeated strategy search makes internal-test scores adaptive development results,
not an independent or unbiased estimate. Prior experiments may already have used
these molecules; disclose that history. The official competition test remains
the final external evaluation, with no official labels or leaderboard feedback
used during autoresearch.

Use `python -m somerra_openadmet.cyp_strategy trial|score|status` for adaptive
internal-test strategy selection. The original training registry and legacy
campaign telemetry remain validation-only diagnostics for replay compatibility.
The new strategy ledger does not alter or relabel historical artifacts.

`submissions.py plan|fit|build|status` is the separate construction path: freeze four
explicit selected recipes, replay each across all five preserved `mt_aux8` folds,
then average fold predictions per CYP. Builds explicitly clear the research test
role and train on four folds per fit; the research test report describes the
three-fold-trained model, not the subsequent five-fold submission ensemble.
Checkpoint selection uses primary validation ST-RAE.
The research and build allocations default to zero; no fit or upload is launched
by setup. See the [recipe contract and runbook](../research/CYP_PROGRAM.md).

`autoresearch.py` remains the research entrypoint. The existing implementation of
the historical reference is `autoresearch.py briford prepare|run|status|predict`;
see [execution commands](briford-reproduction.md#commands-and-execution).
It is currently specific to the four Chemprop configurations, not a generic
specialist/Mol-JEPA runner. Future model adapters must share the saved data, folds,
and evaluator rather than creating their own evaluation protocol.

Use `autoresearch.py molecular plan|run|reproduce|status|prepare` to explicitly
select the older frozen molecular task. Bare `plan` and `run` remain backward
compatible but print a warning to stderr identifying the historical 60/20/20
protocol and directing users here. The warning does not redirect a saved task or
change its recipe. The Workbench `briford` path does not emit this warning.
Existing studies remain reproducible under their recorded protocol.

Before comparing a new run, retain its data/fold hashes, task/record weights,
training recipe, implementation revision, per-fold checkpoints and curves, OOF
predictions, blind predictions, combination weights, and calibration parameters.
Verify checkpoint reload parity, complete identities, finite predictions, and the
750-row submission schema. Continue using the existing allocation ledger for its
authorized campaign; a protocol change does not reset spending.

Submission is separate and requires an explicit instruction. Follow `AGENTS.md`:
prediction CSV only, Open Code false, empty report link, no code/checkpoints or
links, cooldown respected, receipt preserved. Historical one-time upload
authorizations are not reusable.
