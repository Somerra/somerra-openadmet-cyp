# Scaffold folds and a five-model reference ensemble

The user authorized testing scaffold-separated splits and ensembling on 2026-10-28.
This is a new evaluation protocol; the existing molecular-matching task, checkpoints,
and registry remain historical evidence. Our matched depth-four submission scored
0.7952 officially, versus 0.8032 for depth six, reversing their validation ranking.
Use the depth-four recipe unchanged for this first comparison.

## Frozen experiment

Use the same pinned 26-task source pool, sample weights, molecular parent standardization,
loss, model configuration, and training-only target scaling as the reference. Compute
non-chiral RDKit Bemis–Murcko scaffolds on standardized parents across the entire pool.
Identical scaffolds, including the empty scaffold for acyclic molecules, stay together
across official and auxiliary sources. Exclude exact official-blind parents everywhere;
blind scaffolds are not removed, and blind labels are unavailable.

Assign whole scaffolds deterministically to five groups with seed 42, balancing pool
record counts, official record counts, and availability of each scored endpoint. Target
values, model predictions, and official-test similarity do not enter assignment. Require
at least 100 labels per scored endpoint in each fold; fail rather than split scaffolds.
This adopts Scigantic's rotation, not its exact fold assignments or full stacking recipe.
Scaffold separation is stricter chemically; better leaderboard alignment remains a hypothesis.

For held-out fold k, validate on (k+1) modulo 5 and train on the other three folds.
All source records follow their scaffold assignment, including auxiliary-only molecules.
Train five depth-four models for 100 epochs, batch 128, scheduler horizon 200, seed 42
for each run. Only validation selects the best checkpoint. Preserve curves, selected
checkpoints, reload parity, validation predictions, and held-out predictions.

The first series has a 0.65 GPU-hour ceiling within the existing shared 12-hour ledger;
each fold is capped at 0.13 hours. The supervisor stops on a failed fold, stalled worker,
source drift, or exhausted allocation. It does not reset historical spending. A systemd
service survives chat disconnection and imposes an additional wall-time limit. CPU
prediction export follows training. All completed artifacts are retained on interruption.

## Evaluation and interpretation

Concatenate exactly one held-out prediction per official record to compute pooled OOF
MA-ST-RAE, using the existing challenge scorer. Also retain each fold's score. This OOF
result measures single models on held-out scaffolds; it is NOT the five-model ensemble's
internal score. Never average all five models on labeled training molecules: some of
those models trained on each molecule. Scores on our old molecular holdout and these
new scaffold folds have different denominators/populations and are not paired gains.

Predict all 750 official blind molecules with each selected checkpoint, then average
predictions with five fixed weights of 0.2. Verify model/data/checkpoint provenance,
finite outputs, and exact molecule identity/order. Reject missing or mismatched folds.
Write a validated prediction CSV and private provenance manifest; never upload from
this workflow. A later authorized leaderboard submission can compare this combined
change against the matched single-model reference. It will not separately identify
the effects of scaffold splitting, smaller per-model training pools, and averaging.

No learned ensemble weights, endpoint-specific corrections, extra model recipes,
additional seeds, or refit on validation data are included in this first experiment.

## Commands

Run from this branch with `PYTHONPATH=src` and the existing GPU Python environment:

```bash
python autoresearch.py scaffold prepare --bundle BASE --model MODEL_JSON --output NEW
python autoresearch.py scaffold run --output NEW --raw RAW --ledger SHARED_LEDGER --hours 0.65
python autoresearch.py scaffold status --output NEW
```

The output root contains `plan.json`, frozen `folds.csv`, and `model.json`. Each
`fold-K/` contains its data manifest, fit log, checkpoints, predictions, and evaluation.
`series.json` records execution progress and spending. Successful completion creates
`ensemble/activity_submission.csv`, `ensemble/oof.csv`, and `ensemble/manifest.json`.
All predictions stay private. Open Code remains false and report links remain empty.
