# briford reproduction from source measurements

This workflow targets the historical four-component Workbench submission reporting
MA-ST-RAE 0.4378. It trains fresh models; no published predictions, checkpoints, or
Scigantic-derived data tables are used. Scigantic model work remains paused.

The union18 data/task definition is pinned to Workbench commit
`971f6a7f24d86e59bab219347c3e14c006ae4e35`. Training, specialist definitions,
scaffold assignment and calibration follow
`8c72c7559471febaf0962fd551b4eaab8db3a136`. Current Workbench defaults describe a
later, different system. These modules adapt MIT-licensed Workbench behavior;
see `licenses/Workbench-MIT.txt`.

## Data and preserved identity

Read the public `workbench-public-data` measurement tables for official inhibition,
single-concentration, TDI and emax, ChEMBL CYP potency, Veith response, and blind
structures. Preserve raw bytes and hashes. Official scored labels and confidence
bounds must match our official challenge snapshot before preparation succeeds.

The original inference-cache join preserves the input official SMILES: standardized
SMILES from descriptor generation do not replace that column. Descriptors themselves
are unused by this SMILES-only recipe. Public structures arrive already processed
by Workbench's source loaders. Connectivity joins use the original recipe's InChIKey
first block and stable first-record deduplication; Veith responses are clipped to
[-150, 50]. Sources retain separate target columns.

| Member | Trainable records | Observed labels | Targets |
|---|---:|---:|---:|
| union18 | 31,670 | 161,400 | 18 |
| mt_aux8 | 4,905 | 24,025 | 8 |
| d6_isoform6 | 23,264 | 33,407 | 6 |
| d6_single1 | 1,493 | 1,493 | 1 |

The specialist sees direct CYP2D6 potency, log2fc, TDI potency, emax, ChEMBL potency
and Veith response. Each member drops rows without any of its selected targets.
One blind connectivity (`RWZAIHHFBKOVIK`) appears in the public Veith table with
auxiliary response only. It is retained for historical recipe parity and recorded
in the manifest. No official blind scored labels are available or used. The
challenge's published FAQ permits external data. This source-specific reproduction
is distinct from our other protocols, which exclude all exact blind parents.

## Splits and fitting

Each member gets its own five scaffold folds on its own trainable rows. Preserve
row order: Workbench numbers scaffold groups by first appearance, then visits groups
in a seed-42 permutation, assigning each to the least-loaded fold. Preparation
checks every assignment against the pinned upstream splitter. These are neither
our earlier balanced scaffold folds nor Scigantic's CheMeleon assignments.

Each fold network trains on four folds and selects its checkpoint on the fifth.
That fifth fold is also its OOF prediction set, so its reported score is development
evidence, not an untouched test. The 750 official blind molecules remain external.

Use random-initialized Chemprop BondMessagePassing, width 700, depth 5, dropout 0.1,
NormAggregation(100), batch normalization, two-layer FFN width 300, weighted MAE,
batch 64, 400 maximum epochs, patience 40, minimum validation-loss checkpoint,
learning rates 1e-4/1e-3/1e-4 and two warmup epochs. Match mixed-precision training,
FP32 prediction, and seeds 42–46. Primary task weights are inverse label counts
normalized to mean one; auxiliary weights are 0.3 times that mean. Specialist
weights are one for the scored task and 0.3 for auxiliaries.

Five fold predictions are averaged within each recipe. CYP1A2/CYP2C9/CYP3A4 then
average union18 and mt_aux8 equally. CYP2D6 averages all four recipes equally.
There is no additional all-data refit: “full data” means the ensemble covers the
whole training pool, not that each constituent trained on every record.

## Calibration and verification

Apply the pinned `--oof --strae` rule, not today's evolved ensemble/calibration:
compute ensemble OOF Pearson on official labeled rows; multiply it by the published
endpoint factor; set prediction center and sample SD using published population
moments. CYP2D6 instead uses center 3.57 and sample SD 0.90. Floor all outputs at 1.0.
Preserve raw predictions and the exact factors used. The constants use historical
leaderboard feedback; local OOF metrics are not independent validation of them.

Do not infer training reproduction from matching a CSV. Acceptance here requires
20 newly trained networks, immutable source/fold provenance, checkpoint reload
parity, complete finite OOF/blind predictions, official schema/identity validation,
and a new official score. Numerical score similarity is evidence, not proof of
identical historical training artifacts.

## Commands and execution

```bash
PYTHONPATH=src python autoresearch.py briford prepare --output RUN \
  --source PINNED_WORKBENCH_CHECKOUT --official-raw OFFICIAL_RAW
PYTHONPATH=src python autoresearch.py briford run --output RUN
PYTHONPATH=src python autoresearch.py briford status --output RUN
PYTHONPATH=src python autoresearch.py briford predict --output RUN
```

Use the CUDA environment for run. The user allocated a new four-GPU-hour block;
existing historical spending is not reset. The serial supervisor charges worker
wall time conservatively, including featurization and inference. It records every
attempt and stops on failure or allocation exhaustion without submitting a partial
ensemble. Completed folds are verified and reused on restart. An interrupted
supervisor's outstanding reservation is charged conservatively; do not reset it.

A persistent service runs independently of the chat. Predictions remain private.
The user authorized one verification submission after completion: Open Code false,
no report or code links, minimal metadata, receipt retained, cooldown respected.
The model runner itself never uploads; submission follow-through is a separate step.

The separate, explicitly authorized completion watcher can be inspected without
sending anything:

```bash
PYTHONPATH=src python -m somerra_openadmet.briford_submission --output RUN
```

Adding `--execute` waits for complete training, validates all 20 fold artifacts,
respects the portal's current cooldown, makes the single prediction-only upload,
and records the resulting score comparison. `followthrough.json` reports progress;
`predictions/submission_receipt.json` preserves acceptance. An uncertain acceptance
or an existing non-accepted receipt requires reconciliation, never a blind retry.

The 2026-10-28 execution uses private artifacts under
`/mnt/data/somerra-openadmet-runs/briford-20261028`, with systemd user services
`cyp-briford-20261028` and `cyp-briford-submit-20261028`. Model training and submission
follow-through are independently supervised; stopping submission does not discard
training outputs. The initial CPU parity audit matched every initialized state-dict
tensor to the pinned upstream model builder for a common seed and scaler.

Prediction export uses deterministic CUDA reductions while retaining the upstream
matrix precision setting. With ordinary CUDA reductions, repeated inference from
the same trained checkpoint differed by up to 0.00544 on the mixed assay output
scales; deterministic inference matched exactly, including after checkpoint reload.
This setting applies only to prediction, not training. A hashed training-completion
record allows an interrupted export to recover without fitting the model again.
The initial union18 fold's training completed at epoch 134 (best epoch 94). Its
original contract and an explicit inference-only contract amendment are preserved
in the private run directory; the existing checkpoint is reused and all runtime
charges remain in the ledger.

For concurrent folds on one GPU, use the dedicated supervisor after the serial
supervisor has released `run.lock` and its active worker has been settled:

```bash
python -m somerra_openadmet.briford_parallel --output RUN --workers 2
```

Both supervisors use the same frozen data, folds, training worker, checkpoint
verification, and prediction export. The parallel supervisor runs bounded waves
and reserves a separate share of the remaining allocation for each worker. It
conservatively charges the **sum of worker wall times**, even though workers share
one GPU, and records its own source hash and concurrency in `series.json`. A worker
failure stops further dispatch; successful artifacts in that wave remain reusable.
The submission watcher continues to require all 20 verified models.
