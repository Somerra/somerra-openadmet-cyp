# First CYP autoresearch campaign: split-specific learnings

These results apply to campaign `57698048316a` (`cyp-recipes-v3`): saved Workbench
`mt_aux8` assignments, training folds **1, 3, 4**, validation fold **0**, and
internal-test fold **2**. They identify promising recipes under that data,
selection process, and compute allocation. They do not establish the best model,
loss, or hyperparameters for a CYP family generally, another split, or the official
competition population.

This is a record of the October 31–November 1 campaign and the policies adopted
from it. It does not launch research, replace saved folds, or authorize an upload.
The [evidence snapshot](results/autoresearch-learnings-20261101.json) includes exact
split/source/evaluator hashes, selected recipes, repeat results, and artifact hashes.

## What was measured

The campaign scored **156 completed trials**, including repeated recipes: 43 for
CYP1A2, 33 for CYP2C9, 38 for CYP2D6, and 42 for CYP3A4. These are repeated uses of
the same development split, not 156 independent evaluations or distinct strategies.
Training included 30,945 rows with auxiliary observations; validation and internal
test each contained 981 official rows with sparse endpoint labels. Internal-test
label counts were 293 / 249 / 295 / 486 for 1A2 / 2C9 / 2D6 / 3A4.

Validation ST-RAE selected the checkpoint within each trial. Internal-test ST-RAE
then ranked fixed checkpoints across strategies, separately for each CYP, and fed
back into the queue. Lower is better. The four-CYP macro is equally weighted;
training loss values are not comparable selection scores across different losses.

Evaluation and blind molecular parents were excluded from all training sources.
External auxiliary scaffolds can still overlap held-out scaffolds; parent separation
is not a guarantee of scaffold separation across every source. Prior experiments
had also used these molecules. Repeated selection, checkpoint selection, and prior
exposure make these development results, not unbiased held-out estimates.

## Selected recipes and repeat variability

All selected recipes used a focal CYP plus 22 auxiliary outputs, with focal weight
3.0 and auxiliary weight 0.3. This shared outcome does not isolate an auxiliary-task
benefit or establish these weights as optimal. The graph recipes used width 700.

| CYP | First scored baseline ST-RAE | Selected best observed ST-RAE | Selected recipe | Batch / peak LR |
|---|---:|---:|---|---|
| 1A2 | 0.8810 | 0.7639 | Depth-5 graph; interval-aware focal loss, auxiliary MAE; Adam weight decay 0.0001 | 256 / 0.001 |
| 2C9 | 0.6908 | 0.6039 | Depth-7 graph; weighted MAE | 128 / 0.001 |
| 2D6 | 0.9458 | 0.8482 | Full fine-tuning of the pinned CheMeleon encoder; weighted MAE | 512 / 0.001 |
| 3A4 | 0.4931 | 0.4399 | Depth-5 graph; weighted MAE | 1024 / 0.002 |

The baseline-to-selected differences combine adaptive changes and selection over
many trials; they are not controlled effect sizes for the individual recipe choices.
A recipe repeat reuses the same split. It measures run/checkpoint variability,
not robustness to new chemistry or a different fold assignment.

| CYP | Exact selected-recipe runs | Repeat mean ST-RAE | Observed repeat range |
|---|---:|---:|---:|
| 1A2 | 2 | 0.7710 | 0.7639–0.7780 |
| 2C9 | 3 | 0.6235 | 0.6039–0.6572 |
| 2D6 | 3 | 0.8591 | 0.8482–0.8685 |
| 3A4 | 3 | 0.4454 | 0.4399–0.4500 |

The operational ledger selected the lowest eligible observed score, not the best
repeat mean. In particular, the CYP2C9 range warns against treating its 0.6039 run
as a stable expected result. The macro of four best observed scores, **0.6640**,
is likewise an adaptively selected development score, not a performance forecast.

## Learnings to carry forward as hypotheses

- **CYP strategies can diverge.** The selected loss, graph depth, representation,
  and batch size differed across targets. Keep independent CYP queues and controls;
  do not turn these four winners into universal CYP defaults.
- **Representation deserves direct experiments.** CheMeleon fine-tuning was selected
  for 2D6 on this split, while graph recipes were selected for the other three.
  This does not establish CheMeleon superiority on 2D6 generally or rule out
  Mol-JEPA, fingerprints, descriptors, or combinations. Availability, adaptation,
  search coverage, and training policy differ between representation experiments.
- **Loss is an experiment axis.** The 1A2 interval-aware recipe is a candidate to
  retest. The other selected recipes used MAE. Their outcomes do not prove a
  universal interval-loss benefit or that MAE is optimal for the other CYPs.
- **A low single-run score can be unstable.** Retain all repeats and failed trials;
  report means and ranges alongside the operational best. Predeclare any future
  repeat-based promotion rule rather than changing it after seeing which recipe wins.
- **Throughput and score require separate evidence.** The selected batches span
  128–1024. Those values are recipe- and hardware-specific; a batch fitting in memory
  is not evidence of higher throughput or equal optimization quality.

For a new split or data revision, reuse recipes as starting hypotheses, obtain fresh
matched controls, and report that evaluation separately. Do not inherit an incumbent
score, describe a gain as replicated, or pool scores across incompatible contracts.

## Research, final construction, and official performance are distinct

The four selected recipes were frozen and refit across the five preserved folds:
20 fits, training on four folds and selecting a checkpoint on the remaining fold.
Official predictions averaged the five models per CYP. This changes both training
coverage and construction relative to the three-fold research runs.

| Measurement | MA-ST-RAE | Interpretation |
|---|---:|---|
| Best observed strategies on research fold 2 | 0.6640 | Adaptive strategy-selection result |
| Final recipes' pooled five-fold OOF | 0.6191 | Checkpoint-selected development result; not five-fold ensemble test performance |
| Raw 750-row official submission | 0.7151 | Recorded official result for the complete raw pipeline |
| Historical calibrated Workbench submission | 0.4349 | Different model pipeline; contextual comparator |

The raw submission was accepted on November 1 at 12:47 UTC. Recorded official
MAE / R² / Spearman / Kendall were **0.8974 / 0.2154 / 0.7405 / 0.5642**.
The primary ST-RAE and MAE depend on prediction values, not just ranking. Good rank
agreement therefore does not guarantee good challenge error. Differences between
the rows above do not isolate calibration, architecture, split, or ensemble effects.
Official observations are retrospective evidence, not objectives fed into the
internal recipe-search loop. A leaderboard-informed postprocessing decision must
be explicitly distinguished from internal-data-only strategy research.

The campaign summary predates the upload and retains its original preparation-only
status. The later receipt establishes raw-submission acceptance; do not rewrite
historical artifacts to conceal this chronology.

## Calibration: separate results from the current decision

Two own-data transforms were fitted using other folds' OOF predictions and official
training labels, then applied to the excluded fold. With base recipes fixed:

| Variant | Development macro ST-RAE |
|---|---:|
| Raw | 0.61911 |
| Own-data cohort rescaling | 0.63975 |
| Own-data fixed affine | 0.63624 |

Both worsened pooled ST-RAE for every CYP in this experiment. These results apply
to these transforms, base predictions, and scaffold-fold population. They do not
show that calibration generally fails, nor test the borrowed historical Workbench
moments and overrides. Excluding the scoring fold from calibrator fitting does
not undo base-model training, checkpoint selection, or earlier adaptive exposure;
these remain development results.

The user's November 1 decision is to prepare the next candidate with historical
Workbench calibration after fold/model averaging. Preserve raw predictions and
all transform/input hashes. Use the historical population moments, OOF correlation
multipliers, CYP2D6 mean 3.57 / SD 0.90 override, and floor 1.0. This is
leaderboard-informed calibration, not calibration learned solely from our labels.
Its use of unlabeled cohort prediction moments is transductive. See the
[current calibration protocol](workbench-protocol.md#calibration-experiments-and-eligibility).

That candidate is prepared and validated, not submitted as of this evidence snapshot.
As of this November 1 report, we had no matched official
raw-versus-historical-calibrated score for this model. **November 2 follow-up:**
the same raw predictions scored 0.4612 after historical Workbench calibration,
versus 0.7151 before calibration. See the [calibration policy and matched evidence](calibration-policy.md).
Do not attribute the full 0.7151 versus 0.4349 historical gap to calibration.
The project hold was lifted by the user; organizer-specific clearance is not asserted.
Own-data calibration remains a separate experiment axis.

## Policies for subsequent campaigns

1. **Preserve the comparison contract.** Identify source and assignment hashes,
   training/validation/scoring roles, evaluator version, endpoint coverage, and
   calibration variant with every promoted result. State prior exposure and
   repeated score use. Workbench assignments remain the default for matched work.
2. **Keep roles distinct.** Fit weights and scalers on training data; select
   checkpoints on validation; rank strategies on internal test. Do not fit
   calibration on strategy-scoring labels. An expansion evaluation needs fresh
   fits and a separate calibration partition, as described in
   [PR #28](https://github.com/Somerra/somerra-openadmet-cyp/pull/28).
3. **Let research choose the strategy.** Maintain separate CYP queues, editable
   losses, representations, architectures, and task/model weights. At each launch,
   refresh the queue from completed evidence; after completion, verify artifacts
   and launch the next eligible trial within the allocation. Keep the hypothesis,
   parent recipe, changed axes, outcome, and queue revision.
4. **Profile batch size before learning rate.** Hold model and LR fixed during
   batch profiling; select the largest batch that improves measured throughput
   under actual concurrency. Freeze it before LR tuning. Reprofile for changed
   representations; do not infer a batch effect from a simultaneous LR change.
5. **Retain interpretable tracking.** Record actual train/validation losses and
   ST-RAE, MAE, R², Spearman, and Kendall with cohort counts where defined.
   Keep the internal-test strategy ledger separate from legacy validation
   telemetry. Archive recipe, assets, checkpoints, source versions, and predictions.
6. **Require explicit execution bounds.** Check `nvidia-smi` and project-environment
   CUDA access before full fits; never silently fall back to CPU. Recheck GPU memory
   and other jobs before concurrency. Respect budget/deadline gates and preserve
   spending. Store new artifacts under `/mnt/data/somerra-openadmet-runs/`.
7. **Freeze recipes before building submissions.** Construction is a separate path.
   Retain raw and calibrated variants, validate the 750-row schema, and preserve
   receipts. Research or preparation does not authorize an upload. Keep the
   repository private, Open Code false, and Method Report Link empty.

The expansion proxy in PR #28 has median nearest-training Tanimoto **0.465**, versus
**0.587** for official test and **0.419** for the Workbench research test under the
recorded audit. Different reference sizes contribute. It is a more distant-chemistry
stress test, not yet a matched official surrogate. Neither these campaign results
nor calibration behavior have been replicated on that new split. Similarity matching
alone would not establish matching assay or potency distributions.
