# CheMeleon implementation draft (issue #13)

The new commands implement the four-target standalone comparison: immutable foundation
representations, a frozen MLP, full encoder tuning, bounded H100 recipe selection, and
local prediction export. Historical neural commands and artifacts retain their behavior.
No training campaign or challenge submission is implied by this draft.

## Authorized outer-fold quick pass

The user subsequently requested a quick pass with **no inner nesting or inner
holdout selection**, and the same procedure for Mol-JEPA onboarding. Run:

```bash
python scripts/run_chemeleon_quickpass.py --output-dir runs/regression-chemeleon-quickpass-v1
```

The launcher freezes both recipes before scoring any holdout. Throughput profiling
uses synthetic zero labels and discards probe weights. It chooses the smallest batch
within 5% of peak throughput and applies the predeclared capped square-root learning
rate scaling with 5% warmup. It then rotates through all five outer folds at fixed
300/60-epoch horizons for frozen/full tuning. No inner split, early stopping, or
validation-score-based hyperparameter selection occurs. Its four sequential stage
caps total one GPU-hour. Read `status.json`, `recipes.json`, and the final `report.md`
in the run directory; the comparison remains screening evidence, not promotion.

The H100-select configurations below remain available for later inner validation,
but are deliberately not used by this launcher. The launcher records source,
environment, input hashes, exact folds, and prediction-only policy metadata privately.

## Review findings and choices

- The verified foundation checkpoint has **depth 6**, width 2048, dropout 0, and the
  SHA-256 pinned in issue #13. Read its architecture from the checkpoint. Scigantic's
  arm-C configuration declares depth 4, so it is a separate recipe, not our encoder
  definition. Our mean aggregation also differs from its NormAggregation.
- The new trainer uses raw-pIC50 endpoint-balanced masked MAE and a 512-unit MLP with
  BatchNorm/ReLU. The historical runner uses a 256-unit head, normalized targets and
  Huber loss, so its scores do not isolate a representation change.
- BF16 is a training option with forward-parity and finite-gradient checks. Persisted
  evaluation predictions use float32; every completed fold must replay on CPU within
  1e-4 pIC50. Record failures instead of weakening the tolerance silently.
- Frozen features are joined by IDs and SMILES hashes, never row position. A cache can
  cover both train and blind molecules because the fixed encoder sees no challenge
  labels. Its row table contains no fold assignments; each training manifest binds the
  exact label/split checksum. No fingerprint features enter the new frozen MLP.
- The shared interfaces currently implement the CheMeleon adapter only. Mol-JEPA and
  fusion remain issues #14 and #15; unsupported encoders are not silently substituted.

## Commands

Use the existing neural Conda environment, with this package installed editable.
Paths in configurations resolve from the repository root. Each output directory must
be new. Completed caches are read via their atomic manifest; interrupted runs retain
an incomplete run manifest and any completed epoch/fold evidence.

```bash
openadmet-cyp regression cache-representations --config configs/representations-chemeleon-cache.json
openadmet-cyp regression screen-representations --config configs/representations-chemeleon-frozen.json
openadmet-cyp regression screen-representations --config configs/representations-chemeleon-full.json
openadmet-cyp regression screen-representations --config configs/representations-chemeleon-frozen-h100.json
openadmet-cyp regression screen-representations --config configs/representations-chemeleon-full-h100.json
```

Reference defaults are 300 epochs/batch 256/Adam 1e-3 for the frozen arm and 60
epochs/batch 64/AdamW 1e-4 for full tuning. Both use weight decay 1e-4. Optional
`epochs`, `batch_size`, `learning_rate`, and `weight_decay` overrides are recorded.
`selected_folds` supports bounded smoke/recovery experiments; a subset never receives
complete-run metrics or permission to export blind predictions.

The H100 variants profile batches using only the inner training partition, check BF16,
and choose the smallest batch within 5% of peak throughput. They compare the reference
recipe with the bounded learning-rate candidates from issue #13 on an inner 20% group
holdout for each outer fold, then reset weights and refit on all outer-training rows.
Their epoch curves contain inner validation only; outer labels never select settings.
Profiling uses long molecules and two warmup/five timed updates; memory measurements
are conservatively labeled cumulative run peaks. Training logs effective batches,
updates, exposures, elapsed time, and loader time. Timing-based projections stop work
when the remaining allocation is insufficient. A singleton final batch is merged, so
its actual size can be nominal batch size plus one.

Initial configuration reservations sum to **11.5 GPU-hours**, leaving 0.5 hours within
the proposed 12-hour issue ceiling for smoke checks/recovery. Reconcile historical
GPU-hour versus process-hour entries before launching; the ledger's existing 72-hour
ceiling remains in force. The supplied set is not an additional independent budget.
CPU export reserves no GPU time but obeys its configured wall-time limit. Do not
run copies with new output names to evade cumulative campaign accounting.

CUDA runs default to an 80% allocator ceiling and at least 10 GiB free headroom,
reduced for current free memory. This controls this process only, not a reservation
against other users. Old commands retain their historical 25% cap. Extraction's
`batch_size: "auto"` profiles inference separately across 64/128/256/512/1024; an
integer overrides profiling. It never reuses a training-selected batch.

After all five folds complete, use the existing paired comparison command, for example:

```bash
openadmet-cyp regression compare --run-dir runs/regression-chemeleon-full-reference-v1 --reference-run runs/regression-chemeleon-frozen-reference-v1 --reference y_pred
openadmet-cyp regression predict-representations --config configs/representations-chemeleon-predict.json
```

The existing comparison computes paired cluster-bootstrap intervals and numerical
gates. Broad model selection still requires predeclared nested/repeated/scaffold
confirmation before promotion. Export averages the five saved fold models, validates
the official 750-row schema, and records prediction-only policy metadata; it never
uploads. Its default CPU device makes replay/export independent of GPU availability.

## Reference audit update — 2026-10-28

The earlier `baseline-research.md` and `next-submission.md` descriptions of Scigantic as
report-only are historical snapshots. At revision
`e70a2920d611684c9aaff07400d951e2e79dd5f6`, the repository includes training scripts,
tables, member predictions, and ensemble reconstruction. Foundation checkpoint
weights are obtained separately. The two checkpoints (foundation versus organizer
CYP-adapted) must remain distinguished.

| Reference | Labels / evaluation | Relation to this comparison |
|---|---|---|
| Local Workbench-derived R25 mt_aux8 pilot | Eight organizer targets; one outer fold | Context only; extra supervision and incomplete five-fold coverage |
| Scigantic arm C / R25 | Multi-source targets; scaffold folds; specialized weighting, scheduler and ensemble calibration | Recipe reference, not a matched local score |
| Existing CPU incumbent | Saved five-fold nested group evaluation | Paired comparison only when row, label and fold manifests match |
| New CheMeleon arms | Four official targets; common five group folds | Primary baseline for Mol-JEPA |

[Workbench pinned source](https://github.com/SuperCowPowers/workbench/tree/205029d30942f062548806f3ff8d0536bbb24db2/ml_pipelines/OpenADMET/cyp)
and [Scigantic pinned training source](https://github.com/Scigantic/open-admet-challenge-cyp/tree/e70a2920d611684c9aaff07400d951e2e79dd5f6/training)
remain private-report attribution only. Do not copy their external labels, calibration,
blind prediction values, or disclosed blind-overlap exception into this primary study.

## Still required before closing issue #13

Run measured H100 extraction/training profiles and complete the declared five-fold
matrix; produce the private paired comparison report and blind replay/export evidence.
Pin an exact solved Conda environment alongside those run artifacts (the draft records
installed versions and source hashes). Confirm the cumulative ledger before reserving
the campaign. These experimental deliverables are not satisfied by unit tests or by
the presence of runnable configurations.
