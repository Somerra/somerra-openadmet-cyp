# Classification improvement campaign

Improve macro MCC for CYP2D6 and CYP3A4 time-dependent inhibition (TDI), prioritizing
the weak CYP2D6 endpoint. **Paused at the user's request on 2026-10-17.**
Status and results below distinguish completed work from
planned experiments. Existing runs are immutable; leaderboard feedback is not a
selection objective.

## Starting evidence

The September 30 LightGBM run uses ECFP4, ten descriptors, balanced class weights,
and five scaffold folds. CYP2D6 has 1,497 labels (324 positive); CYP3A4 has 3,584
labels (764 positive). Saved OOF ROC-AUC is 0.569738 and 0.793635, respectively.
Macro MCC is 0.197384 at 0.5 and 0.234554 after selecting thresholds on those same
OOF labels. The latter is optimistic model-selection evidence, not an unbiased
reference for promotion. Emax repeats all observed TDI labels without adding labels
or conflicts; its continuous measurements remain useful auxiliary supervision.

## Work tracker

| Stage | Status | Deliverable and acceptance criterion |
| --- | --- | --- |
| 1. Validation | Paused in implementation | Persist scaffold and fingerprint-cluster splits; nested threshold/model selection; per-endpoint diagnostics; paired group bootstrap; replayable manifests. |
| 2. CPU experiments | Planned | Matched-fold LightGBM, ExtraTrees and logistic controls, followed by a bounded 12–18 configuration screen of radius/count features, descriptors, balancing and regularization. |
| 3. Ensemble | Planned | Inner-only endpoint weights and thresholds; compare equal averaging and learned mixtures with the incumbent under identical outer folds. |
| 4. Auxiliary learning | Planned | Two-head molecular classifier, then assay-pretraining or separate regression heads; exclude all held-out molecules/groups from auxiliary fitting. |
| 5. Label audit | Partially complete | Emax label concordance checked; standardized structure overlaps, conflicting labels, missingness and decision-boundary reliability remain. |
| 6. Promotion | Planned | Repeated-split evidence, archived artifacts and validated 750-row submission. |

## Evaluation contract

Use five outer molecular-group folds and three inner group folds. Persist outer
assignments before comparing models. Missing labels remain missing. Standardized
parents group salts and stereoisomers for exclusion; original SMILES provide model
features. A common outer assignment across endpoints supports later multitask work.
Fingerprint Butina groups provide a secondary protocol; group disjointness does not
guarantee that every cross-fold pair falls below the clustering similarity cutoff.

For each outer fold, fit inner models using outer-training rows only. Choose each
endpoint's thresholds and ensemble parameters on inner OOF predictions, then refit
base models on the outer training set and score once on the outer holdout. Selecting
a winner among outer scores remains model selection; confirm finalists with three
predeclared split seeds. Do not reuse ordinary full-dataset OOF predictions as inner
training predictions: their models may have seen the outer holdout labels.

Report pooled macro/per-endpoint MCC, average precision, ROC-AUC, confusion counts,
fold results, threshold variation and positive prediction rates. Paired confidence
intervals resample molecular groups jointly across endpoints. They quantify sampling
variation conditional on fitted predictions, not all model-selection uncertainty.

The provisional promotion gate is an absolute macro MCC gain of at least 0.02 over
the matched incumbent, a paired 95% delta interval above zero, and no endpoint loss
greater than 0.02. Require consistency across split seeds and the secondary protocol.
These are decision criteria, not promised improvements.

## Experiment sequence

1. Establish the nested incumbent and matched logistic/ExtraTrees controls on scaffold
   folds; compare a simple inner-selected blend.
2. Test ECFP radius 3, count fingerprints, class weighting and stronger regularization
   in staged screens. Expand descriptors only if the initial evidence warrants it.
3. Evaluate finalists on fingerprint clusters and three split seeds. Investigate
   disagreements, neighbor similarity and class-specific errors.
4. Train a two-endpoint molecular graph classifier. Test auxiliary potency, Emax and
   single-concentration supervision separately before combining sources. Measured
   assays are training targets, never inputs required at blinded inference.
5. Consider external data only after internal auxiliary experiments, with source
   provenance and structure exclusions. Preserve official evaluation labels.
6. Freeze the selected pipeline, refit, archive probabilities and hard predictions,
   and validate the candidate. No upload is part of this implementation.

## Execution record

2026-10-17: Plan recorded and work paused before the first CPU comparison. The
local implementation adds a classification CLI, persisted group folds, inner-only
threshold and mixture selection, diagnostics, and paired group bootstrap. Six new
tests passed, including a check that changing outer holdout labels cannot change
that holdout's predictions or selected thresholds. Initial lint found two long
lines; formatting was applied, but the follow-up lint check did not run before
the pause. Full-suite validation is outstanding.

Fold preparation did not create its output directory. No classification training
is running, and no new performance result or submission has been produced.

The plan PR contains documentation only. The local evaluator imports molecular
helpers from the separate, uncommitted regression campaign, so implementation must
be isolated or integrated with that dependency before its own PR. Local work remains
in `src/somerra_openadmet/classification.py`, `tests/test_classification.py`,
`configs/classification-controls.json`, and classification additions to `cli.py`.

## Resume checkpoint

After an explicit request to resume:

1. Resolve the regression-helper dependency without bundling unrelated campaign work.
2. Finish lint and full-suite checks, then inspect the data audit and frozen folds.
3. Run the three-model nested comparison with four training threads. The existing
   configuration uses five outer folds and three inner folds; no GPU is needed.
4. Record results and confidence intervals here before choosing the next experiment.

These commands refer to the local implementation; they are not available from the
documentation-only PR by itself. Use the supported Conda environment.

```bash
openadmet-cyp classification prepare \
  --output-dir runs/classification-scaffold-s42-v1
openadmet-cyp classification evaluate \
  --config configs/classification-controls.json
openadmet-cyp classification compare \
  --run-dir runs/classification-controls-s42-v1
```
