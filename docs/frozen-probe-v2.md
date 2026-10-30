# Frozen representation probe v2

This experiment addresses the remaining head-quality comment on private PR #18.
It is separate from the immutable v1 pilot and keeps all encoders frozen.

## Declared before execution

- Reuse the prepared five-fold corpus and frozen CheMeleon mean, Mol-JEPA CLS,
  Atom-JEPA scalar, and Atom-JEPA invariant caches. Use conformer 0 only and one
  prediction per molecule. No encoder loading, adaptation, or seed ensemble.
- Outer fold 0 remains the evaluation set (983 molecules). Split its 3,922 training
  molecules using the existing group-aware 20% inner-holdout function, seed 42:
  3,128 inner-training and 794 inner-validation molecules. Persist the same IDs
  for every learner, feature set and head seed; clusters cannot cross partitions.
- Fit feature and endpoint target scalers on inner-training rows only. Keep the
  common MLP `D -> 512 -> BatchNorm -> ReLU -> Dropout(0.2) -> 4`, Adam 1e-3,
  weight decay 1e-4, batch 256, float32, and endpoint-balanced masked MAE on
  standardized targets. Reuse the shared head factory and optimization step.
- Run head seeds 42/43/44. Select the best inner-validation MA-ST-RAE checkpoint
  with a 300-epoch ceiling, patience 30 and minimum improvement 1e-5. Outer labels
  never select checkpoints or hyperparameters. Restore that checkpoint and retain
  epoch-by-epoch training loss and inner scores. Do not refit on the inner holdout.
- CPU controls use exactly the same feature cache, inner-training rows, validation
  rows, and outer-test rows. Include the inner-training endpoint means; ridge
  alpha in `[1, 10, 100]`; and LightGBM L1 with leaves in `[15, 31]`, learning rate
  0.05, at most 500 estimators, minimum child samples 20, L2 penalty 1, seed 42,
  and eight CPU threads. Ridge selects alpha by inner macro ST-RAE. LightGBM
  selects endpoint iterations by inner ST-RAE with patience 30, then leaf count
  by inner macro ST-RAE. Targets are standardized before fitting these controls
  and predictions are transformed back to pIC50. Constants are not tuned.
- Twelve MLP fits each reserve at most 0.08 GPU-hours under both the original
  four-hour pilot cap and the existing shared campaign budget. Four control sets
  run on CPU only with a half-hour wall limit each. The supervisors retain failed
  job charges. No new allocation is created.

The larger input dimensions still give larger first-layer parameter counts.
Early stopping matches the selection rule and maximum budget, not the final
number of updates. The new models train on fewer molecules than v1, so v1-v2
changes are not an isolated test of dropout or target normalization. Fold 0 has
already been inspected: these results remain exploratory and cannot establish a
general representation ranking or pretraining-architecture effect. Controls
anchor this experiment; the earlier approximately 0.74 tree result is not assumed
to use an identical split or feature protocol.

## Execution and artifacts

With the original environment active and `SOMERRA_CAMPAIGN_LEDGER` pointing at the
existing bound campaign ledger:

```bash
python scripts/run_frozen_probe.py --data-root /absolute/original-checkout/runs
```

The driver persists its design before fitting. Existing outputs require `--resume`
and unchanged inputs, split, recipe, implementation digests, and artifact checksums.
The immutable implementation hashes also bind resumed per-fit results to the code.
Artifacts go under the original pilot's `regularized-probe-v2/`: per-fit model
checkpoints, selected epochs, training curves, predictions, endpoint metrics,
split IDs, summary tables, and cumulative accounting. CPU checkpoint replay must
agree within 1e-4 pIC50; control artifacts are replayed after serialization.

Run manifests remain partial-fold artifacts (`complete=false`) and cannot pass
the existing blind-export gate. Fine-tuning and challenge submission remain deferred.

## Completed results — 2026-10-30

All twelve MLP fits and four CPU control sets completed. Lower MA-ST-RAE is better.
MLP values are mean ± sample SD across head seeds 42/43/44, not seed-ensemble
scores or cross-fold uncertainty. Controls are single deterministic fits after
inner selection. Every row uses exactly the same partitions and one evaluation
prediction per molecule.

| Frozen features | MLP | Ridge | LightGBM |
|---|---:|---:|---:|
| CheMeleon mean | 0.875013 ± 0.016195 | 1.198042 | 0.793051 |
| Mol-JEPA CLS | 0.839971 ± 0.021192 | 0.881572 | 0.806545 |
| Atom-JEPA scalar | 0.884304 ± 0.014105 | 0.837321 | 0.814071 |
| Atom-JEPA invariant | 0.875051 ± 0.010187 | 0.983976 | 0.825055 |

The shared **inner-training-mean baseline is 1.006835**. It is computed from
training labels, distinct from the evaluation-mean reference in the ST-RAE
denominator. All v2 MLPs and LightGBM controls improve on this fitted constant.
Mol-JEPA has the lowest mean MLP score in this pilot; CheMeleon has the lowest
LightGBM score. The ordering depends on the readout, so these data do not establish
a model-independent representation ranking. Atom-JEPA does not show an advantage
over both references in the matched, one-conformer MLP comparison.

### Checkpoint selection and head limitations

| Frozen features | Selected epochs, seeds 42/43/44 | Stopped at epochs |
|---|---|---|
| CheMeleon mean | 18 / 7 / 8 | 48 / 37 / 38 |
| Mol-JEPA CLS | 7 / 10 / 6 | 37 / 40 / 36 |
| Atom-JEPA scalar | 31 / 19 / 61 | 61 / 49 / 91 |
| Atom-JEPA invariant | 6 / 3 / 9 | 36 / 33 / 39 |

In all twelve runs, training loss fell after the selected epoch while the final
inner-validation score worsened. This is direct evidence of overfitting later in
these training trajectories, and the saved best checkpoint avoids using those
later weights. The v2 MLP scores improve numerically over all corresponding v1
standardized-feature, single-conformer results. Several things changed together
(target scaling, dropout, checkpoint selection, and training-set size), so this
does not isolate which change caused the improvement or diagnose v1 causally.

LightGBM outperforms the MLP for every cache here. The stronger MLP is therefore
still an imperfect probe of available information. All ridge controls selected
alpha 100, the upper boundary of the declared grid; this grid does not establish
optimal ridge performance. All tree controls selected 15 leaves. Their selected
endpoint iterations, in the repository's official endpoint order, are:

- CheMeleon: 176 / 61 / 172 / 222.
- Mol-JEPA: 235 / 78 / 133 / 216.
- Atom-JEPA scalar: 163 / 108 / 186 / 327.
- Atom-JEPA invariant: 65 / 45 / 157 / 170.

No settings were changed after viewing outer scores. The earlier approximately
0.74 tree result was not reproduced under this exact protocol; the same-cache,
same-split CheMeleon control here is 0.793051. Broader conclusions still require
independent folds and an appropriate uncertainty analysis.

### Cost and verification

- New full-process GPU charge: **0.062196 hours** for twelve fits.
- Cumulative original pilot + conformer follow-up + v2: **0.441994 / 4 hours**.
- Four control sets used CPU only. Encoders stayed frozen; no fine-tuning, blind
  prediction export, model promotion, or challenge submission occurred.
- Full test suite: 261 passed, 10 GPU/private-checkpoint-dependent skips. Clean
  CPU pilot suite: 39 passed, zero skips. Tests cover cluster isolation, train-only
  scaling, best-checkpoint restoration, outer-label independence, current-input
  resume rejection, serialized control replay, and dual-ledger GPU supervision.

Private `validation.json` records independent artifact/score verification and
installed versions, including PyTorch 2.13.0, scikit-learn 1.9.1 and LightGBM 4.7.0.
Per-fit source digests bind the artifacts to the executed implementation. All
checkpoints, predictions, and detailed learning curves stay in ignored local run
artifacts; this private repository contains the aggregate results and protocol.
