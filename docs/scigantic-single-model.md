# R25-inspired single model

This workflow implements a single predictor using selected components of Scigantic
R25. It is not a reproduction of R25's 0.4436 submission, which combines several
recipes, five-fold averaging, and endpoint-specific corrections. Our previous
Scigantic-derived depth-four baseline was randomly initialized and scored 0.7952.

## Recipe

Use CheMeleon's general pretrained encoder (the initialization used by Scigantic's
arm C), adapted to the arm-B/C architecture: hidden size 2048, depth 4, dropout 0.2,
FFN width 1024 and two layers, normalized aggregation and batch normalization.
The source encoder was pretrained at depth 6; its recurrent message-passing tensors
transfer exactly to depth 4, as in Scigantic's runner. All four encoder tensors must
match in name, shape and finite values. No partial loading is permitted.

Retain the pinned 26-task pool, including four TDI heads (the data extension used by
Scigantic's g26 arm B), source record weights, weighted MAE, training-only target
scaling, batch size 128, seed 42, and 200-epoch learning-rate schedule. Combining
arm-C initialization with the 26-task pool is our single-model adaptation, not an
upstream named R25 member. Reuse the existing trainer, checkpoint parity check,
challenge metrics and prediction validation.

The checkpoint is downloaded from the versioned public source:
https://zenodo.org/records/15460715/files/chemeleon_mp.pt
SHA-256: `c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651`.
The published MD5 was also verified during acquisition. Load tensors with
`weights_only=True`. Record source identity and transfer details in every run.
Inference uses the final checkpoint alone and does not require the source encoder.
Keep attribution in private artifacts; do not put links into challenge submissions.

Upstream reference: `Scigantic/open-admet-challenge-cyp`, revision
`e70a2920d611684c9aaff07400d951e2e79dd5f6`, particularly
`training/chemeleon_arms/sagemaker_runner.py` and `training/README.md`.

## Selection and final fit

Reuse the existing five-way scaffold assignment, fixing fold 0 as validation and
folds 1–4 as training. It is a convention, not the fold with the best observed score.
Whole scaffolds follow the same role across official and auxiliary records.
This produces 26,327 selection-training records, including 3,925 official rows,
and 980 official validation rows. No separate internal test is introduced.
These molecules have appeared in previous experiments; validation is development
feedback, not a historically untouched estimate.

Run selection for 100 epochs and use its best validation MA-ST-RAE checkpoint to
choose the epoch count. Freeze a decision manifest. Then initialize a fresh model
from the same foundation encoder and fit all 32,909 eligible records, including all
4,905 official labeled rows, for exactly that count. Refit scalers on this full
training pool. Do not warm-start the final stage from the selected checkpoint, and
do not use validation to choose a new final checkpoint. The final export contains
one model's predictions, not an ensemble. There is no independent local score for
the all-data final model. Official blind-parent exclusions remain in force.

The first execution has a 0.32 GPU-hour cap within the existing shared 12-hour
allocation. Reserve half for selection and the actual remaining allowance for the
final fit. Charge complete subprocess lifetimes, including failures. Stop on
incomplete training, deadline, stale progress, changed code/data/encoder, or
provenance mismatch. A persistent systemd supervisor provides an additional wall
limit. No automatic retry/resume or additional experiment is appended.

Calibration and moment restoration remain disabled. They are not implied by
pretrained initialization and require separately assessed fitting/evaluation before
use. No third-party frozen predictions or organizer-finetuned checkpoint is used.
The prepared CSV remains private and is not automatically submitted.

## Usage

```bash
PYTHONPATH=src python autoresearch.py single prepare \
  --source-fold EXISTING_SCAFFOLD_SERIES/fold-0/data \
  --encoder VERIFIED_CHEMELEON_PT --output NEW_RUN
PYTHONPATH=src python autoresearch.py single run \
  --output NEW_RUN --raw RAW --ledger SHARED_LEDGER --hours 0.32
PYTHONPATH=src python autoresearch.py single status --output NEW_RUN
```

The output contains the frozen data/model/encoder plan, separate `selection/` and
`final/` fit artifacts, `selection-decision.json`, and `predictions/` with the
validated CSV and its checkpoint provenance. `run.json` tracks stages; the shared
ledger is authoritative for all charges, including failed stages.
