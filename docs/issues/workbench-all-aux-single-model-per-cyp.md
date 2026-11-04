# One Workbench model per CYP with all auxiliary tasks

Tracking: [issue #23](https://github.com/Somerra/somerra-openadmet-cyp/issues/23). Proposal recorded 2026-10-30; numeric settings remain proposed until the experiment is frozen.

## Objective

Keep the Workbench Chemprop training recipe, but build **one model per scored CYP**, with broad auxiliary supervision and a larger loss coefficient for that CYP's primary potency target. The candidate consists of **four retained checkpoints total**: CYP1A2, CYP2C9, CYP2D6, and CYP3A4. Each endpoint uses only its designated model's primary output. No five-fold ensemble, multiple recipe members per CYP, shared-model blend, or calibration.

This issue records the plan; it does not launch training or allocate additional compute. It is distinct from #22's within-CYP log2FC on/off ablation and does not reopen #21's Mol-JEPA plan.

## Scoring and competition rules checked on 2026-10-30

The regression track scores direct-inhibition pIC50 for the four CYPs. Its primary metric is **MA-ST-RAE**, the arithmetic mean of their four ST-RAE scores; lower is better. Each CYP contributes one quarter of the macro score, regardless of its observed-label count. Auxiliary assays do not directly contribute.

For each observation, soft-threshold error is the distance outside the fitted credible interval: zero inside it, otherwise distance to its nearest edge. Our shared evaluator divides summed endpoint errors by the same interval-distance error for a constant predictor equal to that evaluation cohort's mean measured potency. Verify numerical agreement with the pinned official evaluator before fitting; retain raw pIC50 predictions and original bounds for evaluation. Plain point-target MAE is a training surrogate, not the leaderboard metric.

The separate TDI classification track scores CYP3A4/CYP2D6 using MCC. TDI-condition potency used as an auxiliary here is not that classification output.

External data and pretrained models are permitted. Standard leaderboard participation does not require open-source code. Regression files require all four finite potency predictions for each of the 750 supplied molecules. Using four endpoint-specific models is compatible with that prediction schema; the reviewed rules do not prescribe one shared architecture.

Sources:
- [Official challenge tutorial](https://github.com/OpenADMET/CYP-Challenge-Tutorial/blob/main/README.md), inspected blob `5a0698c7a86fe721828693f4bbcc8fc580cb67fc`.
- [Pinned competition scoring, FAQ, and submission rules](https://huggingface.co/spaces/openadmet/cyp-challenge/blob/4afcf026a24d4db6938073ac0410a60b3a271081/config.py), especially scoring and data/rules sections.
- Local source snapshots: `runs/cyp-specialist-issue-evidence-20261030/`.

## What the existing generalists actually optimize

There are two different generalist recipes; their weights must not be conflated.

| Output | Historical Workbench union18 / mt_aux8 weight | Expanded 26-target full generalist weight |
|---|---:|---:|
| CYP1A2 direct pIC50 | 1.096103 | 1.0 |
| CYP2C9 direct pIC50 | 1.204434 | 1.0 |
| CYP2D6 direct pIC50 | 1.036636 | 1.0 |
| CYP3A4 direct pIC50 | 0.662826 | 1.0 |
| Each selected auxiliary | 0.3 | 1.0 |

**Workbench:** masked weighted absolute error on per-target standardized labels. The four primary coefficients are inverse observed-label counts, normalized to average one. Counts in the frozen member table are 1,412 / 1,285 / 1,493 / 2,335 for CYP1A2 / CYP2C9 / CYP2D6 / CYP3A4. Historical weights were computed from full member-table label availability, including held-out rows; target means/scales were fitted on training rows only. Each auxiliary receives 0.3. Row weights default to one. Historical checkpoint selection uses combined weighted validation loss, including auxiliaries.

**Expanded 26-target generalist:** the frozen full run records `loss_kind=mae` and `auxiliary_weight=1.0`. Its primary and auxiliary task coefficients are therefore all one, but it additionally multiplies by the frozen bundle's nonuniform per-molecule `sample_weight`. Equal task coefficients do not mean equal aggregate task contributions. This is a separate, wider Scigantic-derived architecture, not the Workbench encoder.

Both reductions sum weighted errors over observed entries and divide by the **number of observed entries**, not the sum of task weights and not the number of targets:

`L = sum(i,t)[mask_it * row_weight_i * task_weight_t * abs(z_pred_it - z_true_it)] / sum(i,t)[mask_it]`

Here `z` is standardized with training-only means/scales. Thus one raw pIC50 unit effectively receives coefficient `task_weight / target_scale`. The primary/auxiliary ratio is a standardized-error ratio, not a ratio per raw assay unit. A target with many labels can contribute strongly despite a small coefficient; 0.3 for every auxiliary does not cap their combined share at 30%. On the full union18 table, primary labels account for about 11.8% of coefficient-weighted observed-entry mass; this is not a measured loss or gradient share.

Private evidence:
- `/mnt/data/somerra-openadmet-runs/briford-20261028/data/manifest.json`: member tasks, counts, and exact weights.
- `src/somerra_openadmet/data_sources/workbench.py::task_weights` and `briford.py::make_model`: Workbench implementation.
- `runs/regression-hillclimb-20261026-v2/final/run_manifest.json`: expanded full-run configuration.
- Matching frozen `hillclimb_model.py` SHA-256: `8083ac50da2434c69235ccd515b6733adf54881e87f20ab7657accad0c047d18`.
- Chemprop `nn/metrics.py`: observed-entry loss reduction.

## Task scope

Retain **all 22 auxiliary targets from the expanded local recipe** discussed in the PCA analysis. Each specialist has **23 outputs**: its own official direct-inhibition pIC50 plus:

- Four single-concentration log2FC targets.
- Four TDI-condition pIC50 targets.
- Five ChEMBL potency targets, including CYP2C19.
- Five PubChem maximum-response targets, including CYP2C19.
- Four PubChem qHTS potency targets.

Exclude the other three official direct-inhibition primary heads. Retain auxiliaries across all CYPs, including auxiliary-only molecules. Confidence bounds remain evaluation metadata, not output heads. Do not add Emax or classification targets.

This extends historical Workbench union18, which has four primary and 14 auxiliary heads. Preserve the Workbench model/training recipe while explicitly recording the expanded supervision. Keep qHTS despite its strong observed overlap with ChEMBL; retain the provenance caveat about possible measurement reuse.

Use the hash-verified `regression-hillclimb-20261026-data` bundle as the source of these labels, with an explicit column-name mapping. Do not treat the PCA-imputed matrix as training labels. Only observed assay values provide supervision.

## Proposed specialist loss and fixed recipe

Use Workbench's randomly initialized Chemprop BondMessagePassing encoder (width 700, depth 5, dropout 0.1), NormAggregation(100), batch normalization, and two-hidden-layer FFN (width 300, dropout 0.1). Retain batch 64, two warmup epochs, learning rates 1e-4 / 1e-3 / 1e-4, maximum 400 epochs, patience 40, and training-only per-target standardization. Pin package versions and runtime precision before fitting.

**Proposed initial coefficients: primary = 3.0; each auxiliary = 0.3.** This makes a focal observed standardized error ten times as influential as one auxiliary error, before label availability and residual magnitude. It increases primary emphasis relative to the existing specialist convention of 1.0 versus 0.3. The value 3.0 is a proposed starting setting, not an empirically established optimum or a competition requirement. Freeze it before examining evaluation outcomes; no weight grid is included in this issue.

Keep the observed-entry denominator unchanged. Use unit row weights to preserve Workbench's row-weight convention; do not silently import the separate 26-target recipe's sample weights. Log primary and auxiliary loss sums separately, counts, target scales, and their weighted contributions. Missing labels contribute no error; missing primary labels must not suppress an observed auxiliary. Reject zero-observation or constant training heads.

Select and early-stop each specialist using **its own primary validation ST-RAE only**, calculated in pIC50 units with confidence bounds. Record primary validation MAE as a secondary diagnostic. This explicitly replaces historical selection on combined auxiliary/primary validation loss. Do not change training to interval loss in this experiment.

## One-model evaluation and comparison

Use one fixed development split rather than training five models per CYP. Proposed split: inherit Workbench's official `mt_aux8` scaffold fold 0 as the scoring cohort; split the remaining official molecular parents 80/20 into training/validation with seed 42 (approximately 64/16/20 overall). Persist one common parent assignment for all four specialists and controls before selecting endpoint labels. Retain known historical-development exposure in the limitations; this is not a historically untouched test.

The expanded bundle and Workbench tables have different identities and partitions. Preparation must reconcile them explicitly, inherit official cohort membership before attaching auxiliary labels, and audit duplicate parents and the documented historical connectivity-overlap exceptions. Never inherit a different table's fold number as though it denotes the same cohort. Remove scoring/validation parents from every training source and remove all official blinded parents from training. External-only parents may enter training after these exclusions; report scaffold/similarity overlap rather than claiming the entire expanded training pool is scaffold-disjoint. Resolve any unmatched official identity before fitting.

Fit four fresh specialist models with seed 42 and retain one selected checkpoint each. Evaluate all four once on the fixed scoring cohort. These are the four-model candidate; no fold averaging or final full-data refit is part of this first experiment.

For a controlled comparison, fit one shared 26-output Workbench model on the same data split, preprocessing, architecture dimensions, scheduler, and row-weight convention. Use inverse training-only primary counts normalized to mean one, and 0.3 per auxiliary. Select that control on primary macro validation ST-RAE. This is a declared adaptation of historical Workbench. Historical 18-target Workbench and the wider 26-target model remain context, not matched controls.

Report per-CYP ST-RAE and MAE, macro MA-ST-RAE, scoring counts, paired specialist-minus-control differences, selected epochs, optimizer updates, total runtime, and model storage/inference cost. Explain that the comparison tests specialization plus primary reweighting; it does not isolate their individual effects. Four specialist encoders also have more total capacity and training cost than one generalist. Use fixed, paired molecular-parent bootstrap resamples for descriptive uncertainty; report seed variability as unmeasured.

## Deliverables and acceptance

- [ ] Explicit 23-head mapping for each CYP; four candidate checkpoints total.
- [ ] Frozen source/data hashes, split IDs, overlap exclusions, task counts, target scalers, exact loss weights/reduction, and seeds.
- [ ] Tests cover masked loss, the 10:1 primary/individual-auxiliary coefficient ratio, training-only scaling, parent separation, and focal-primary-only checkpoint selection.
- [ ] All four specialists and the matched generalist control finish; retain curves, recovery state, selected checkpoints, and aligned scoring predictions.
- [ ] Reloaded predictions agree within 1e-4 pIC50; the final endpoint assembly uses exactly one primary output from each specialist.
- [ ] A private comparison report distinguishes observed results, missing-data assumptions, prior adaptive exposure, and unequal aggregate compute.
- [ ] Profile the expanded training pool and record a ledger-backed compute cap before launching. Preserve historical charges; this issue grants no new GPU allocation.
- [ ] Keep code, models, reports, attribution, and metadata private. No submission is authorized by this issue. Any later authorized challenge submission stays prediction-only, with `open_source_code=false` and an empty `model_report_link`.

