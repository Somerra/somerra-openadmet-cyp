# Baseline results

Run date: 2026-09-30. These are chemistry-aware local validation estimates, not blind
leaderboard scores.

## Shared configuration

- official dataset revision: `146282d2ce1532f1803679a834a9bec9dac1e482`;
- features: ECFP4 (radius 2, 2,048 bits) plus ten RDKit descriptors;
- estimator: per-endpoint LightGBM, 500 trees;
- validation: five scaffold-disjoint folds;
- seed: 42;
- model thread budget: 8.

## Direct inhibition

Command:

```bash
conda run -n somerra-openadmet-cyp openadmet-cyp baseline activity --model lightgbm --n-splits 5 \
  --seed 42 --radius 2 --n-bits 2048 --n-jobs 8 \
  --data-dir data/raw --output-dir runs/baseline-20260930-activity-lgbm-v1
```

| Endpoint | OOF ST-RAE |
| --- | ---: |
| CYP1A2 | 0.8941 |
| CYP2C9 | 0.7391 |
| CYP2D6 | 1.0055 |
| CYP3A4 | 0.5483 |
| **Macro average** | **0.7968** |

Submission candidate:
`runs/baseline-20260930-activity-lgbm-v1/activity_submission.csv`

SHA-256:
`2c39f973516c670ce6a82b4933760852a38f7d4c534c95d04be0e252e0e0c485`

## Time-dependent inhibition

Command:

```bash
conda run -n somerra-openadmet-cyp openadmet-cyp baseline tdi --model lightgbm --n-splits 5 \
  --seed 42 --radius 2 --n-bits 2048 --n-jobs 8 \
  --data-dir data/raw --output-dir runs/baseline-20260930-tdi-lgbm-v1
```

| Endpoint | OOF MCC at 0.5 | OOF-selected threshold | OOF MCC at selected threshold |
| --- | ---: | ---: | ---: |
| CYP2D6 | 0.0473 | 0.15 | 0.1011 |
| CYP3A4 | 0.3475 | 0.15 | 0.3680 |
| **Macro average** | **0.1974** | — | **0.2346** |

The selected-threshold result is a model-selection estimate, not an unbiased performance
claim. Thresholds were derived only from training-set OOF predictions.

Submission candidate:
`runs/baseline-20260930-tdi-lgbm-v1/tdi_submission.csv`

SHA-256:
`ad39b2696d018d969519cc167d72c81e92323cc909fb222f972434325a1dbb3e`

## Validation and upload state

Both candidates contain exactly 750 rows and pass both this repository's strict
validator and the current official tutorial validators at tutorial revision
`832ae6d6447b07c2c71059de34d07f1130388247`.

The live portal accepts regression and classification as independent tracks, enforces a
12-hour per-user/per-track cooldown, and requires a real Hugging Face username plus
leaderboard/privacy metadata. These files have **not** been uploaded. Generated run
artifacts are intentionally gitignored; their commands, metrics, and hashes are recorded
here so they can be reproduced without publishing blinded predictions.
