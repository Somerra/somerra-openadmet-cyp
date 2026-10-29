# Workbench reproduction: October 28, 2026 submission

Our freshly trained reproduction of the historical briford / SuperCowPowers
Workbench recipe scored **0.4349 official MA-ST-RAE**, with recorded rank **24**.
The portal accepted the prediction CSV on October 28 at **5:50:08 p.m. Eastern**
(21:50:08 UTC). Lower MA-ST-RAE is better.

| Official submission | MA-ST-RAE | Difference from this run |
|---|---:|---:|
| This Workbench reproduction | **0.4349** | — |
| Historical briford recipe targeted | 0.4378 | This run is lower by 0.0029 |
| Scigantic R25 reference | 0.4436 | This run is lower by 0.0087 (~2%) |
| Our preceding depth-four submission | 0.7952 | This run is lower by 0.3603 |

These are comparisons of scored submissions, not controlled ablations. They do
not isolate the effects of data, ensembling, architecture, or calibration. Rank
24 is the saved post-submission observation, not a claim about the current board.
Other recorded official metrics: MAE **0.6294**, R² **0.6244**, Spearman **0.7552**,
and Kendall **0.5785**. Endpoint-specific official scores were not captured.

## What produced the predictions

All 20 networks were trained from random initialization using Chemprop: depth 5,
width 700, two-layer regression head of width 300, dropout 0.1, batch size 64,
weighted MAE, mixed precision, maximum 400 epochs, and early-stopping patience 40.
No Scigantic predictions or pretrained checkpoints were used.

| Configuration | Molecular records before splitting | Tasks | Fits |
|---|---:|---|---:|
| `union18` | 31,670 | Four official inhibition, four single-concentration, five ChEMBL potency, five Veith response | 5 |
| `mt_aux8` | 4,905 | Four official inhibition and four single-concentration | 5 |
| `d6_isoform6` | 23,264 | CYP2D6 inhibition, single-concentration, TDI-condition potency, Emax, ChEMBL potency, Veith response | 5 |
| `d6_single1` | 1,493 | Official CYP2D6 inhibition only | 5 |

Each configuration builds its own five scaffold folds. A network trains on four
folds and selects its checkpoint on the fifth, with seeds 42–46 across fits.
That validation fold also supplies its OOF predictions; there is no separate
untouched internal test in this recipe. The official 750-molecule blind set is
used for predictions, and its scored labels remain unavailable locally.
One blind molecule has an external Veith auxiliary response retained for recipe
parity; no official blind potency label is used in training.

For each endpoint, first average the five fold models within a configuration.
For CYP1A2, CYP2C9, and CYP3A4, average `union18` and `mt_aux8` equally (ten networks).
For CYP2D6, average all four configurations equally (20 networks). Only each
endpoint's direct-inhibition output enters the submitted prediction.

The historical calibration then sets each endpoint's mean and sample standard
deviation using published population moments and adjusted OOF correlation.
CYP2D6 instead uses mean 3.57 and SD 0.90. All predictions are floored at pIC50 1.
These constants incorporate historical leaderboard feedback. The official result
therefore measures the complete trained ensemble **plus calibration**; this run
does not establish how much either contributes independently. There is no final
refit on all training rows.

## Training and internal evaluation

All fits completed with verified checkpoint reloads and finite OOF/blind predictions.
The table lists epochs in fold order 0–4; epoch numbers are one-based.

| Configuration | Best epochs | Stopping epochs |
|---|---|---|
| `union18` | 94, 93, 67, 43, 17 | 134, 133, 107, 83, 57 |
| `mt_aux8` | 13, 11, 71, 35, 23 | 53, 51, 111, 75, 63 |
| `d6_isoform6` | 44, 12, 30, 42, 38 | 84, 52, 70, 82, 78 |
| `d6_single1` | 18, 12, 15, 18, 16 | 58, 52, 55, 58, 56 |

Raw, uncalibrated ensemble OOF MA-ST-RAE was **0.6284**: CYP1A2 **0.7365**,
CYP2C9 **0.5479**, CYP2D6 **0.8133**, CYP3A4 **0.4158**. These are development
scores on checkpoint-selection folds and a different population, not a local
measurement of the official calibrated submission score.

The H100 run switched from serial fitting to two concurrent folds. The ledger
charged **3,386.19 seconds (56.44 minutes)** of a four-hour allocation, conservatively
summing worker wall time, including setup, inference, and recorded recovery work.
This is not physical GPU busy time or end-to-end elapsed time.

An inference repeatability check initially stopped export. Deterministic CUDA
inference fixed the mismatch; the first fold's existing best checkpoint was
reused, with the original contract and amendment preserved. A later submission
schema check rejected identifier-column order locally; it was corrected without
changing the prediction file. The portal accepted the first upload attempt.

## Receipt and reproducibility evidence

Only the validated **750-row prediction CSV** was uploaded. The portal confirms
**Open Code: No** and **Model Report Link: Not submitted**. No code, checkpoints,
code links, or report links were published through the submission.

Submission SHA-256:
`0e763564e98d5cfa681e964bc2f702ca79ef9e2bcfe2d9459019c8cb303d26a4`.

The [evidence snapshot](results/briford-20261028.json) records the accepted receipt
fields, leaderboard observation, all 20 fit summaries, calibration values, runtime
versions, and private artifact hashes. Full source data, checkpoints, predictions,
receipt, and logs remain under
`/mnt/data/somerra-openadmet-runs/briford-20261028`.
See the [reproduction recipe](briford-reproduction.md) for pinned upstream revisions,
data joins, exact training settings, and execution commands.
