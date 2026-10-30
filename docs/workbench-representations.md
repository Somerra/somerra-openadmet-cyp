# Matched frozen representations on Workbench auxiliary-task folds

This study compares CheMeleon mean features (2048 dimensions), Mol-JEPA CLS
features (512), and Atom-JEPA scalar features (256). Encoders remain frozen;
Atom-JEPA uses the single cached seed-42 conformer. This is a comparison of
available representations under a common head, not a causal isolation of
pretraining architecture. Representation widths imply different parameter counts.

The [completed five-fold results](workbench-representations-results-20261030.md)
include all 15 fits, paired uncertainty, and the uncalibrated Chemprop context.

The user requested all three representations on Workbench folds with identical
auxiliary supervision. Use the preserved `mt_aux8` bundle: 4,905 molecular rows,
four official direct-inhibition pIC50 tasks, and four single-concentration log2fc
auxiliaries. All three verified caches already cover these exact IDs and SMILES.
The broader 23,264-row CYP2D6 six-task specialist bundle is a separate experiment
requiring additional feature extraction. This study is not a submission fit.

Preflight found two historical cross-fold connectivities (four records):
`LOUPRKONTZGTKE` spans folds 0/1 and `MJIHNNLFOKEZEW` spans folds 0/4. Workbench's
isomeric scaffold strings distinguish their stereochemistry. These exact
hash-pinned assignments are preserved and the records are included in the design
manifest; any additional connectivity overlap is rejected. The folds are disjoint
by the saved scaffold strings, not strictly by stereo-insensitive connectivity.
All three representations inherit this same evaluation limitation.

## Fixed comparison

- Reuse the exact five `mt_aux8` scaffold assignments pinned in
  [the artifact record](results/workbench-protocol-artifacts.json). No new split.
- Train on four folds and select on the fifth; one seed `42 + fold` per fit.
  All three encoders run each fold, giving 15 fits and complete OOF predictions.
- Use one shared head: input -> 512 -> BatchNorm -> ReLU -> Dropout(0.2) -> 8.
  Adam, learning rate 0.001, weight decay 0.0001, batch 256, float32.
- Fit feature and target population means/standard deviations on training rows
  only, with minimum scale 1e-6. Convert outputs back to each task's native units.
- Inverse training-label counts set the four primary weights, normalized to mean
  one; each auxiliary has weight 0.3. Missing labels contribute no loss. Divide
  the weighted absolute standardized error by the number of observed labels,
  not by the sum of weights. All encoders use identical targets and weight vectors.
- Select minimum weighted standardized validation MAE over all eight tasks, with
  maximum 400 epochs and patience 40. Save the best checkpoint and complete curve.
  This common head recipe differs from the historical end-to-end Chemprop recipe.
- Use the existing official-style ST-RAE scorer on the four official tasks only.
  Report each fold, pooled OOF, and per-endpoint scores. These are checkpoint-
  selected development results, not independent internal-test or official scores.
- No calibration, output floor, conformer averaging, or repeat-seed ensemble.
  Training-target normalization and its inverse are retained.

## Entrypoint and artifacts

```bash
PYTHONPATH=src python autoresearch.py workbench-probe plan --config /path/to/config.json
PYTHONPATH=src python autoresearch.py workbench-probe run --config /path/to/config.json
PYTHONPATH=src python autoresearch.py workbench-probe status --config /path/to/config.json
```

The config supplies protocol `workbench-mt-aux8-frozen-v1`, the pinned data directory,
three encoder cache locations (and Mol-JEPA's model directory for permitted-use
evidence), a new output directory, CUDA, the existing campaign ledger, and a finite
`max_gpu_hours`. `plan` validates data/cache identities without fitting or reserving
GPU time. `run` is supervised against the shared ledger and refuses existing output
directories. Never reset spending or overwrite a partial run to retry it.

This series is capped at 0.75 GPU-hours of the existing 20-hour allocation, with
17.789850 hours previously charged at preflight. No new allocation is created.
The supervisor charges full child lifetime and kills an over-budget worker.

The run preserves its design, input and implementation hashes, per-fold training
curves and batch-order digests, fitted weights/counts, checkpoints, predictions,
and selected epochs. CPU checkpoint replay must match GPU predictions within
1e-4 for all eight tasks. Inputs are reverified before accepting completed results.
The runner neither predicts the official blind set nor uploads anything.
