# CheMeleon five-fold controls — completed 2026-10-29

All six follow-up arms completed the requested five outer folds. Cached CheMeleon
features with LightGBM performed best among the new arms (MA-ST-RAE **0.775495**).
Regularized full tuning scored **0.825758**, and the regularized frozen MLP scored
**0.846595**. The saved CPU incumbent remains stronger at **0.743511**. No model
was promoted or submitted. This report and all source attribution remain private.

## Evaluation and interpretation

The follow-up retains the first pilot's exact prepared table and folds: 4,905 rows,
4,901 parents, 4,565 ECFP4/2048 Butina groups (distance threshold 0.4), and 6,525
observed labels across four official inhibition endpoints. Each outer fold is held
out once; the other four folds train the model. There is **no inner split, recipe
selection, or early stopping**. Neural epochs are fixed. Missing labels are masked.
Feature scaling and target normalization use only the current training partition.

All recipes were fixed in `protocol.json` before this follow-up's scores were read.
They were motivated by the first pilot and review feedback, so the combined study is
exploratory. Multiple settings changed together; improvements cannot be attributed
to normalization, dropout, batch size, or learning rate individually. The fixed
Ridge setting performed poorly and is not an optimized linear baseline.

The newer main-branch scaffold protocol is a separate study. Mol-JEPA should use
these same explicit Butina folds for this comparison, or both methods should be
rerun together under the new protocol. The incumbent uses saved nested predictions
on matching rows/folds, whereas these new recipes are outer-only. Paired scoring
is valid on those predictions but does not make model-selection procedures identical.

## Results

Lower MA-ST-RAE is better. Endpoint columns are ST-RAE; macro is their mean.

| Arm | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | Macro |
|---|---:|---:|---:|---:|---:|
| Constant training mean | 1.001481 | 1.001717 | 1.000543 | 1.000158 | 1.000975 |
| Cached features + Ridge | 2.866217 | 2.550275 | 4.247779 | 1.420822 | 2.771273 |
| Cached features + LightGBM | 0.862859 | 0.716888 | 0.942840 | 0.579395 | **0.775495** |
| Frozen reference MLP | 1.099031 | 0.937535 | 1.266501 | 0.667708 | 0.992694 |
| Frozen regularized MLP | 0.970142 | 0.755303 | 1.103717 | 0.557216 | 0.846595 |
| Full regularized tuning | 0.963424 | 0.713233 | 1.083947 | 0.542428 | 0.825758 |

For context, the [first pilot](chemeleon-quickpass-20261028.md) scored 1.066875
(frozen, throughput-scaled large batch) and 1.525201 (full tuning). The saved CPU
incumbent scores 0.743511. LightGBM demonstrates useful signal in the cached
representations; its success and the constant sanity check do not establish that
all possible cache or model issues have been excluded. CYP2D6 remains difficult
for the neural recipes, with ST-RAE above 1.

The paired comparison verifies matching IDs, labels, folds, and groups and uses
1,000 cluster-bootstrap replicates. Candidate-minus-incumbent macro differences:

| Arm | Difference | 95% bootstrap interval |
|---|---:|---:|
| LightGBM | +0.031984 | [+0.016364, +0.048152] |
| Frozen reference | +0.249182 | [+0.219921, +0.279461] |
| Frozen regularized | +0.103083 | [+0.081992, +0.124481] |
| Full regularized | +0.082247 | [+0.061360, +0.102910] |

Regularized frozen versus frozen reference has a difference interval of
[-0.165952, -0.123911]; full regularized versus frozen reference has
[-0.190383, -0.141006]. These intervals describe this fixed sample/protocol and
are not corrected for exploratory comparisons or a substitute for nested confirmation.

The historical approximately 0.739 reference included fingerprints and a nested
ensemble. It is not an expected score for a pure CheMeleon-feature tree model.
Workbench's eight-target pilot and Scigantic's multi-source/scaffold ensemble remain
recipe context, not matched scores; see the [reference audit](chemeleon-implementation.md).

## Fixed recipes

All CheMeleon arms use the same general pretrained depth-6, width-2048 encoder,
mean aggregation, and four official targets. Cached controls have no fingerprints.
The neural head is 2048 → 512, BatchNorm, ReLU, optional dropout, then four outputs.

| Neural arm | Optimizer | LR | Weight decay | Batch | Epochs | Target loss | Dropout | Training precision |
|---|---|---:|---:|---:|---:|---|---:|---|
| Frozen reference | Adam | 1e-3 | 1e-4 | 256 | 300 | Raw-pIC50 masked MAE | 0 | FP32 |
| Frozen regularized | Adam | 3e-4 | 1e-3 | 256 | 100 | Standardized masked MAE | 0.2 | FP32 |
| Full regularized | AdamW | 5e-5 | 1e-3 | 256 | 60 | Standardized masked MAE | 0.2 | BF16 |

These follow-up recipes use no warmup. Target means/scales are fit independently
within each training fold, stored in checkpoints, and inverted for prediction.
Evaluation and replay use raw pIC50 in FP32. BF16 passes the finite-gradient/parity
probe. The smaller batches deliberately test a different optimization regime after
the large-batch pilot underperformed; filling GPU memory is not itself an objective.

CPU controls: training-fold endpoint means; StandardScaler plus Ridge(alpha=10,
LSQR, tolerance 1e-4); and LightGBM with 300 trees, LR 0.03, 15 leaves, minimum
child 40, L2 5, row/column sampling 0.8, seed 42 + fold, eight threads. CPU times
were approximately 0.04 s, 1.94 s, and 50.20 s respectively (excluding setup).

## Recovery, memory, and shared budget

The first attempts at both frozen recipes stopped after the first training epoch,
with **zero completed outer folds**. The cost estimator extrapolated first-epoch
CUDA startup across all remaining epochs/folds. The fix estimates from warmed-up
epochs; the supervisor's hard full-process deadline remains enforced. Both arms
were restarted from scratch with identical learning settings and folds. Only timing
projection and per-attempt wall-time caps changed. Failed outputs remain intact.

The original follow-up allocation was 0.12 GPU-hours (caps 0.02/0.02/0.08). Recovery
used caps of 0.03 hours each, after checking that prior actual spending plus recovery
caps still fit within the same 0.12-hour envelope and shared 12-hour campaign.
The reusable launcher now uses 0.03/0.03/0.06 caps, still totaling 0.12 hours.

| GPU process | Charged hours | Charged seconds |
|---|---:|---:|
| Frozen reference initial timing stop | 0.004696 | 16.90 |
| Frozen regularized initial timing stop | 0.004765 | 17.16 |
| Full regularized, completed | 0.032626 | 117.46 |
| Frozen reference recovery, completed | 0.029484 | 106.14 |
| Frozen regularized recovery, completed | 0.019033 | 68.52 |
| **Follow-up total, including failures** | **0.090604** | **326.18** |

The first pilot's measured legacy charge (0.030888 hours) was reconciled into the
persistent shared campaign before launching these controls. Its older timing scope
is preserved rather than retroactively claiming full-process measurement. The new
coordinator reserves before launch, supervises the full worker lifetime, waits for
exit, then settles the shared ledger. Workers do not release reservations themselves.
Environment-selected absolute campaign paths take precedence; no per-config legacy
ledger, budget reset, or new allocation is used.

At completion, campaign use is **11.954367 / 12.000000 hours**, no reservations
remain, and **0.045633 hours (2.74 minutes)** remain. Follow-up charging includes
startup, checkpoint replay, and the failed attempts, not just training loops.

On the H100 NVL, frozen reference/regularized peak allocated memory was about
0.124 GiB (0.146 GiB reserved); full tuning peaked at 2.626 GiB allocated and
7.852 GiB reserved. These are cumulative process peaks. The 80% allocator limit
and 10 GiB free-headroom policy remained enabled; no out-of-memory failures occurred.
All 15 completed neural fold checkpoints replayed on CPU within **3.34e-6 pIC50**
(tolerance 1e-4). Saved Ridge/LightGBM models also passed prediction replay.

## Private evidence and reproducibility

Artifact root: `runs/regression-chemeleon-controls-20261029-v1/`.

- `completion.json` is the combined final result index with `complete: true` and
  paths to all six completed arms. Original `status.json`/`summary.json` retain
  the earlier failed attempts; do not use them as the final combined status.
- `protocol.json` SHA-256:
  `3c6c158861d466f498315116c373a72a050959efeabd8d87cc7f031ca337bc25`.
- Prepared training/split table SHA-256:
  `efc567b18f9dba44d91c645aa26ff0c8d48d9bf887e1972b873a7095d8ebd971`.
- Foundation checkpoint SHA-256:
  `c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651`.
- Reused cache: `runs/regression-chemeleon-quickpass-20261028-v1/cache/`,
  with ID/SMILES/checkpoint identity checks. No new extraction was needed.
- `source/`, `recovery-source/`, copied launcher and `recover_frozen.py` preserve
  the executed code. Recovery introduced the timing fix; full tuning used the
  earlier estimator. Run manifests hash source, OOF predictions, and checkpoints.
- Per-arm configs, epoch logs, OOF tables, checkpoints, CPU replay errors, paired
  comparison JSON, and adjacent `.accounting.json` files remain local and private.
  The initial pilot's `campaign-reconciliation.json` records the budget migration.

Validation on the updated PR: 190 tests passed, two optional installed-checkpoint
tests skipped in the isolated worktree; full Ruff lint and formatting checks passed.
The real GPU experiments separately exercised the installed foundation checkpoint.
Nested confirmation, Mol-JEPA, fusion, promotion, and blind export/submission are
outside this completed outer-only pass. Issue #13 remains open for the deferred work.
