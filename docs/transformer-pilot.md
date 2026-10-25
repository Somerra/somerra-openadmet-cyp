# Matched SMILES-transformer pilot

Run from the repository root in the existing neural environment:

```bash
runs/neural-env/bin/python -m somerra_openadmet.transformer_pilot \
  --config configs/regression-smiles-pilot.json
```

The isolated entry point leaves the existing neural screen and checkpoint formats unchanged.
The output directory must not exist. The config specifies prepared official data, official
auxiliary data, output directory, and the shared ledger. `SOMERRA_GPU_LEDGER`, when set,
overrides the ledger path. The pilot reserves one hour, with at most five minutes for GPU
smoke checks and 27.5 minutes per model, including validation. It retains the best completed
inner-validation checkpoint if training hits its allocation. Final outer evaluation runs on CPU.

The reference is a local adaptation of the R25 `mt_aux8` member at upstream commit
`e70a2920d611684c9aaff07400d951e2e79dd5f6`, not the full ensemble or a reproduction of its
published score. It uses the eight organizer-only targets and published task weights from
`training/workbench_recipe/mt_aux8_member_spec.json`. The local prepared corpus retains
4,905 rows; auxiliary labels join by standardized parent. Only the four single-concentration
log2FC heads are added. No external labels or TDI-condition targets are used.

The random graph reference has 1,395,708 parameters. The random SMILES transformer has
1,398,248 parameters (0.18% more): three pre-norm blocks, width 192, six heads, feed-forward
width 800, GELU, dropout 0.1, sinusoidal positions, and masked mean pooling. Parameter
matching includes all trainable encoder and output parameters. Canonical isomeric SMILES
use a fixed ASCII vocabulary, zero padding, and dynamic batch padding without truncation.

Both models use seed 42, batch size up to 64, Adam, Chemprop's Noam-like schedule
(2 warmup epochs; 1e-4 / 1e-3 / 1e-4; 400-epoch horizon), FP16 mixed precision, and
training-only target normalization. Checkpoint selection minimizes weighted masked MAE
on a deterministic 20% inner cluster holdout, with patience 40. Outer fold 0 is evaluated
only after training. This deliberately replaces the upstream scaffold split with the
repository's molecular-cluster split. The same shuffled batches are used by both models.

Artifacts include input/source/split hashes, architecture and task metadata, checkpoints,
per-epoch validation loss and MA-ST-RAE against steps and elapsed time, throughput, peak
GPU memory, and final outer predictions. Compare learning curves only over their common
step/time range. A single fold and seed provide exploratory evidence, not a promotion gate
or a general conclusion about transformer sample efficiency.

Tests cover tokenizer round-trip, padding invariance, finite masked gradients, parameter
matching, checkpoint round-trip for both models, split isolation, and fit-only normalization.
The new modules require the optional neural environment; the ordinary CPU CLI does not.

Graph-transformer follow-up: private issue #6. Neither this experiment nor that issue
publishes code, checkpoints, reports, or challenge submissions.

The first attempt was stopped by an overly strict finite-gradient check during normal
FP16 loss-scale adjustment. Its artifacts remain in `runs/regression-smiles-pilot-v1`.
The corrected runner lets GradScaler skip overflowing updates and lower its scale;
only successful updates advance the learning-rate schedule and optimizer-step count.
Persistent overflow below unit scale still fails. Curves record skipped updates.

The recovery config `configs/regression-smiles-pilot-recovery.json` writes a new directory
and lists the prior attempt. The runner reconciles that attempt's finalized ledger charge
with its manifest and subtracts it from the one-hour allocation before reserving more time.

## Completed pilot

`runs/regression-smiles-pilot-v2/report.md` records the completed comparison. Outer
MA-ST-RAE was 0.748153 for the graph reference and 0.904496 for the SMILES transformer
(20.9% worse). Transformer throughput was 70.0% of reference throughput. Both models
stopped early; the total charged time including recovery was 0.016460 GPU-hours.
These results do not support a learning-efficiency gain for this configuration.

Regenerate the private report and PNG/SVG learning curves with:

```bash
runs/neural-env/bin/python scripts/report_transformer_pilot.py \
  runs/regression-smiles-pilot-v2
```

## Follow-up with the GPU free

The follow-up keeps the small model size and original split fixed. It performs five
predeclared trials using only inner-validation scores: a fresh reference, a cached-input
SMILES control, a regularized SMILES recipe at batch sizes 64 and 256, and the batch-256
recipe with eight deterministic training-only SMILES enumerations. Token tensors stay
on the GPU; validation always uses canonical SMILES. Tests verify enumeration identity,
stereochemistry, reproducibility, and exclusion of validation rows from augmentation.

The regularized recipe uses AdamW (weight decay 0.01), peak learning rate 0.0003,
five-epoch warmup and cosine decay over 400 epochs, BF16, patience 60 and minimum
100 epochs. Several settings change together, so this is a recipe comparison rather
than attribution to a single optimizer change. Batch-size comparisons have different
numbers of optimizer updates per epoch. The shared trainer retains the initial pilot's
defaults when no override is supplied and records all overrides in checkpoints.

```bash
runs/neural-env/bin/python -m somerra_openadmet.transformer_followup \
  --config configs/regression-transformer-followup.json
runs/neural-env/bin/python scripts/report_transformer_followup.py \
  runs/regression-transformer-followup-v1
```

The runner reserves at most 45 minutes and reconciles prior completed pilot charges
against the shared ledger before enforcing a cumulative one-hour cap. It writes new
artifacts and does not re-evaluate the outer holdout. Results are adaptive inner-split
model-selection evidence and cannot independently establish a generalization gain.

The completed follow-up selected the fresh graph reference (inner weighted MAE 0.567233).
The best SMILES recipe by the predeclared weighted-MAE criterion was the augmented
batch-256 model (0.645886 versus 0.659339 for the cached control). Batch-256 transformer
throughput increased to roughly 21,000 molecules/second from roughly 8,000, with different
optimizer-update counts per epoch. This improved throughput and modestly improved the
transformer validation objective, but did not close the gap to the graph reference.
The private report is `runs/regression-transformer-followup-v1/report.md`.
