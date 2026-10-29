# Matched frozen representations: pilot and conformer follow-up

All results below use the same persisted outer fold 0, the four official CYP
pIC50 endpoints, and head seeds 42/43/44. They are exploratory single-fold results,
not five-fold OOF results or leaderboard scores. Lower MA-ST-RAE is better. Values
are the mean and sample standard deviation of individual-seed scores.

## Original pilot

| Frozen features | Dimensions | Raw MA-ST-RAE | Standardized MA-ST-RAE |
|---|---:|---:|---:|
| CheMeleon mean | 2048 | 1.035717 ± 0.012692 | 1.115816 ± 0.021415 |
| Mol-JEPA CLS | 512 | 1.014414 ± 0.005668 | 1.005796 ± 0.016864 |
| Atom-JEPA invariant, one conformer | 768 | 1.298492 ± 0.127612 | 1.128870 ± 0.009236 |
| Atom-JEPA scalar, one conformer | 256 | 1.172374 ± 0.054827 | 1.096170 ± 0.006767 |

The scalar standardized cell was completed in the follow-up. The original pilot
had 21 fits; the follow-up added 15 fits and reused three invariant heads.

Every head uses input -> 512 -> BatchNorm -> ReLU -> 4, Adam 1e-3, weight decay
1e-4, batch 256, 300 epochs, float32, endpoint-balanced masked MAE in raw pIC50,
and the final checkpoint. The number of first-layer parameters differs with
embedding width. Standardization uses outer-training rows only.

## Conformer sensitivity

Both views are standardized. The scaler is held fixed across conformer counts,
using the original conformer on outer-training molecules. With multiple training
conformers, each molecule still occurs once per epoch: one conformer is sampled
uniformly using a separate RNG. Every fit performs 4,800 optimizer updates with
the same molecular batch order for each paired head seed. Evaluation averages
predictions, not embeddings.

| Training conformers | Evaluation conformers | Scalar MA-ST-RAE | Invariant MA-ST-RAE |
|---:|---:|---:|---:|
| 1 | 1 | 1.096170 ± 0.006767 | 1.128870 ± 0.009236 |
| 1 | 2 | 1.065495 ± 0.006238 | 1.075863 ± 0.017370 |
| 1 | 10 | 1.035640 ± 0.012062 | 1.034538 ± 0.019666 |
| 2 | 1 | 1.074787 ± 0.002878 | 1.077034 ± 0.014400 |
| 2 | 2 | 1.056704 ± 0.011575 | 1.046216 ± 0.015544 |
| 2 | 10 | 1.037229 ± 0.013390 | 1.014460 ± 0.017183 |
| 10 | 1 | 1.007057 ± 0.014735 | 1.011773 ± 0.026734 |
| 10 | 2 | 1.000138 ± 0.017049 | 0.994990 ± 0.028616 |
| 10 | 10 | 0.989837 ± 0.011788 | 0.981689 ± 0.024071 |

Ten-conformer training and evaluation improved invariant features by 0.147181
relative to their standardized single-conformer baseline (exploratory 95% paired
cluster-bootstrap interval for candidate minus reference: [-0.202241, -0.091232]).
Scalar features improved by 0.106332, interval [-0.158609, -0.051934].
The scalar-minus-invariant difference for ten conformers has interval
[-0.028898, 0.047142], so these data do not distinguish the two readouts clearly.
Intervals use 1,000 cluster resamples and do not correct for multiple comparisons.

The ten-conformer Atom-JEPA scores are numerically below the fixed CheMeleon and
Mol-JEPA references. They use additional 3D views, and the follow-up was motivated
by inspection of this same fold. Additional-fold confirmation is needed before
claiming a general ranking or promoting a model. Encoders also differ in
pretraining data and modalities; the pilot does not isolate a causal effect of
pretraining architecture or establish absence of pretraining data overlap.

## Cost and retained evidence

- Original pilot: 0.109387 GPU-hours, including failed extraction attempts and diagnostics.
- Follow-up: 0.270411 GPU-hours.
- Cumulative: **0.379798 / 4 GPU-hours**; no fine-tuning or challenge submission.
- Extra geometries: 44,145 views for 4,905 labeled molecules; 264 finite MMFF
  nonconvergences retained, zero random-coordinate retries, zero dropped rows.
- Encoder weights/buffers were unchanged. Cached/online extraction and CPU
  checkpoint replay passed. The inherited approximately 0.015% relative-L2
  rotation residual and its pre-score tolerance decision are documented in the
  [protocol](atom-jepa-frozen-pilot.md).

Private artifacts remain under `runs/regression-frozen-representation-pilot-v1`:
`report.md`, `summary.csv`, per-fit checkpoints/predictions, and
`conformer-sensitivity-v1/{report.md,summary.csv,paired_comparisons.json,budget_summary.json}`.
The follow-up's `evaluation/` holds all per-endpoint scores, paired predictions,
per-conformer predictions, and variability measurements. These artifacts are
intentionally ignored by Git; the PR contains this aggregate summary only.

Historical execution validation passed 55 pilot tests and 20 follow-up tests
(overlapping suites). The readiness review later passed 107 tests on its original
working tree and verified all 36 fitted checkpoints against current data/cache
hashes. PR hardening adds current-input resume checks, reproducible setup, and CPU
CI without changing or rerunning those experiment results. Its separate validation
counts and exact commit are recorded in the PR.

The PR covers this frozen pilot only. It does not claim completion of broader
fine-tuning, fusion, full-CV, or model-promotion workstreams.
