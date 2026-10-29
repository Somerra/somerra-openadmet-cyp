# CheMeleon fixed outer-fold quick pass — 2026-10-28

Both models completed all five outer folds. The large-batch recipes were fast but
did not improve the saved CPU incumbent. These are exploratory screening results,
not a promoted model or an estimate of the best achievable CheMeleon performance.

| Endpoint | Frozen features + MLP | Full fine-tuning | Existing CPU incumbent |
|---|---:|---:|---:|
| CYP1A2 | 1.159591 | 1.529480 | 0.832707 |
| CYP2C9 | 1.035802 | 1.783903 | 0.683655 |
| CYP2D6 | 1.378723 | 1.927139 | 0.915334 |
| CYP3A4 | 0.693385 | 0.860281 | 0.542348 |
| Macro MA-ST-RAE | **1.066875** | **1.525201** | **0.743511** |

Lower is better. Full-minus-frozen macro difference is approximately +0.4583;
the paired molecular-cluster bootstrap 95% interval is [+0.4139, +0.5030]. Neither
arm passes the existing numerical promotion gates. The CPU incumbent's predictions
use the same rows, targets, and outer folds but its own nested ensemble selection.

## Frozen protocol

- Official four-target dataset: 4,905 rows, 6,525 observed endpoint labels,
  4,565 molecular groups. All five fixed ECFP4/Butina group folds were evaluated.
- No inner nesting, inner validation holdout, outer early stopping, or score-based
  hyperparameter search. Mol-JEPA onboarding should follow the same procedure;
  full nested confirmation remains deferred.
- Both recipes were frozen before any outer evaluation. Batch selection used only
  throughput, with synthetic zero targets and discarded probe weights. Learning
  rates used capped square-root scaling from the declared reference rates.
- Frozen arm: batch 2048, Adam LR 0.0028284271, 300 epochs, 600 successful updates
  per fold. Full tuning: batch 1024, AdamW LR 0.0004, 60 epochs, 240 updates per fold.
  Both use weight decay 1e-4 and 5% linear warmup, then constant LR.
- Both arms use the same 512-unit BatchNorm/ReLU MLP, raw-pIC50 endpoint-balanced
  masked MAE, and the original CheMeleon foundation checkpoint (depth 6, width 2048).
- BF16 forward/backward checks passed in every fold. Persisted evaluation uses
  float32. Every fold replayed on CPU; maximum discrepancy was 2.87e-6 pIC50,
  below the predeclared 1e-4 tolerance.

## H100 measurements

Hardware: one H100 NVL, 95,830 MiB total. The allocator allowance was at most 80%
of device memory with 10 GiB free headroom; no parallel model jobs were launched.

| Stage | Charged elapsed seconds | Peak allocated | Peak reserved |
|---|---:|---:|---:|
| Representation extraction | 2.45 | 4.01 GiB | 5.65 GiB |
| Training throughput probes | 5.93 | 11.09 GiB | 11.79 GiB |
| Frozen MLP, all five folds | 9.40 | 0.15 GiB | 0.17 GiB |
| Full tuning, all five folds | 93.42 | 9.76 GiB | 45.33 GiB |

Total charged time: **0.03089 GPU-hours (1.85 minutes)** against a one-hour cap.
Charges conservatively include CPU work inside each reservation; setup outside
reservations adds wall time. Extraction selected batch 512 and processed about
11,681 molecules/second during the measured extraction loop. Reserved memory
includes the PyTorch caching allocator and differs from live tensor allocation.

The fixed training objectives ended low (batch MAE roughly 0.085–0.125), but held-out
errors were large. These results do not isolate whether batch size, learning rate,
training duration, head design, or other recipe choices caused the poor generalization.
No extra tuning against these outer predictions was performed.

## Private reproducibility evidence

Local artifact root: `runs/regression-chemeleon-quickpass-20261028-v1`.
It contains the protocol, selected recipes, source snapshot, Conda package records,
input and checkpoint hashes, immutable representation cache, per-epoch training
curves, fold checkpoints, OOF tables, CPU replay errors, and paired comparisons.

Prepared training SHA-256:
`efc567b18f9dba44d91c645aa26ff0c8d48d9bf887e1972b873a7095d8ebd971`.
Protocol SHA-256:
`ee57d575678ee0bc6fed4b4de73e25c72ef924401556fd92931a8b6a61240a1a`.

The run used the existing experiment checkout at `714601c` plus the snapshotted
implementation. The PR ports that implementation onto current main, preserving its
CLI and extracting the fold-selection helper that current main no longer exports.
Tests on the PR branch check this integration; the run artifacts remain the exact
execution provenance rather than a claim that the experiment ran from the PR commit.

No blind prediction file was exported or submitted by this quick pass. Code,
checkpoints, attribution, and reports remain private; any future submission remains
prediction-only with `open_source_code=false` and `model_report_link=""`.
