# Next activity submission candidate

Run date: 2026-10-02. This document records local model-selection evidence, not a blind
challenge score. No challenge upload was made.

## Live contract checked before modeling

- challenge Space revision: `e7c370a19b111df8df54ef4fb9c0a090428e6635`;
- official dataset revision: `146282d2ce1532f1803679a834a9bec9dac1e482`;
- tutorial/validator revision: `832ae6d6447b07c2c71059de34d07f1130388247`;
- activity and TDI tracks open in phase 2, with 750 blinded compounds and a 12-hour
  per-user/per-track submission cooldown.

## What transfers from Scigantic

The public
[`Scigantic/open-admet-challenge-cyp`](https://github.com/Scigantic/open-admet-challenge-cyp)
repository is a two-file method report, not an executable model release. It describes a
12-model CheMeleon/Chemprop ensemble trained with external ChEMBL and Veith CYP labels,
structure exclusion, endpoint-specific metric-aligned stacking, and output rescaling.
It contains no code, data manifest, split assignments, OOF predictions, checkpoints, or
ensemble weights, so its reported result cannot be reproduced from the repository.

The immediately testable idea is therefore **representation-diverse, endpoint-specific,
metric-aligned ensembling**. External-label multitask neural models remain a later phase
because they first require exact source provenance and blinded-structure exclusion.

## Candidate construction

Four five-fold scaffold-disjoint OOF runs use identical folds and official input hashes:

| Base learner | ECFP radius | OOF MA-ST-RAE |
|---|---:|---:|
| LightGBM | 2 | 0.796772 |
| LightGBM | 3 | 0.798704 |
| ExtraTrees | 2 | 0.846760 |
| Ridge control | 2 | 1.774016 |

For each endpoint and held-out fold, non-negative weights summing to one are optimized on
the other four folds against exact ST-RAE. The reported ensemble score is computed only
from those held-out predictions. Final test weights are then fit on all OOF rows. This
separates the honest cross-fitted estimate from the all-OOF model-selection fit.

## Local evidence

| Endpoint | Best single OOF ST-RAE | Cross-fitted ensemble | Change |
|---|---:|---:|---:|
| CYP1A2 | 0.860644 | 0.829589 | -0.031055 |
| CYP2C9 | 0.736682 | 0.704590 | -0.032093 |
| CYP2D6 | 0.929940 | 0.916541 | -0.013398 |
| CYP3A4 | 0.548345 | 0.535879 | -0.012466 |
| **Macro** | **0.796772** | **0.746650** | **-0.050122** |

This is a 6.3% reduction relative to the incumbent LightGBM OOF MA-ST-RAE. Repeating the
weight search with seed 31415 produced 0.746634, a difference of 0.000016. Test
predictions from the two searches correlate above 0.999999 for every endpoint.

Final all-OOF weights vary by endpoint as expected: ExtraTrees dominates CYP1A2 and
CYP2D6, while the two LightGBM variants dominate CYP2C9 and CYP3A4. The weak Ridge model
receives zero weight on CYP1A2/CYP2D6 and small calibration-like weights on CYP2C9/CYP3A4.

## Candidate artifact

- path: `runs/candidate-20261002-activity-metric-ensemble-v1/activity_submission.csv`;
- SHA-256: `549c54797ad919fb2e3122e6a7fbe8f9642b6fd3290d29761379ebe0adf6f496`;
- rows: 750;
- repository validator: pass, no warnings;
- official pinned tutorial validator: pass with exact expected molecule IDs.

Generated submissions and model artifacts remain gitignored. Their immutable hashes,
commands, OOF metrics, fold-specific weights, and input hashes are retained in the run
directory. Upload still requires an explicit participant identity and authorization.
