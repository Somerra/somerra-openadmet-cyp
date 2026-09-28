# Baseline research

Research snapshot: 2026-09-28. This document distinguishes reproducible code from
method reports and leaderboard anecdotes.

## Official sources

### Dataset

- Source: [OpenADMET CYP challenge dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)
- Pinned Git revision: `146282d2ce1532f1803679a834a9bec9dac1e482`
- Activity regression: 4,905 compounds with sparse pIC50 values and credible intervals.
- TDI: 6,145 compounds with CYP2D6/CYP3A4 labels and paired assay measurements.
- Test: 750 compounds.
- Auxiliary data: 17,504 single-concentration measurements and 6,145 Emax rows.

The downloader deliberately excludes `cyp-challenge-TEST-BLINDED_structures.csv`.

### Tutorial and evaluator

- Source: [OpenADMET/CYP-Challenge-Tutorial](https://github.com/OpenADMET/CYP-Challenge-Tutorial)
- Reviewed revision: `832ae6d6447b07c2c71059de34d07f1130388247`
- Official introductory model: RDKit descriptors plus one LightGBM model per endpoint.
- Official warning: random splits are optimistic; cluster-based validation is preferred.
- Regression metric: prediction error is distance to the nearest credible-interval
  boundary, normalized by the same error for a constant mean predictor.
- TDI metric: endpoint MCC, macro-averaged across CYP2D6 and CYP3A4.

Our metric formulas and submission contracts are tested against these definitions; no
challenge submission API is included.

## Public approaches reviewed

### `adlvdl/cyp_competition`

- [Repository](https://github.com/adlvdl/cyp_competition)
- Reproducible code and dated data snapshots.
- Uses ECFP4, LightGBM, nested scaffold CV, calibration, ensembles, and paired tests.
- Reports simple ECFP4/LightGBM TDI MCC around 0.259 for CYP3A4 and 0.116 for CYP2D6
  under its nested scaffold protocol; three of four simple regression endpoints were
  near or worse than a constant before calibration.
- Useful conclusion: fingerprint/tree models are controls, not the likely ceiling.

### `avnikonenko/openadmet_challenge_cyp_2026`

- [Repository](https://github.com/avnikonenko/openadmet_challenge_cyp_2026)
- Compares ECFP6 + descriptors + LightGBM with directed message-passing models,
  Chemprop, transfer from single-concentration data, and ensembles.
- Uses saved ECFP cluster folds as its primary validation design.
- Useful conclusion: cluster folds and masked multitask learning are the natural next
  step once the CPU baselines are stable.

### `Scigantic/open-admet-challenge-cyp`

- [Method report](https://github.com/Scigantic/open-admet-challenge-cyp)
- Reported 11th of 219 on the full interim regression board (MA-ST-RAE 0.4382).
- Describes multi-source, multi-assay learning and a 12-model CheMeleon/Chemprop
  ensemble over roughly 32,000 compounds and 26 endpoints.
- It is a blueprint, not runnable code: no training implementation, checkpoints,
  exclusion file, or predictions were published at review time.

### `lachrymator/openadmet-cyp-challenge-public`

- [Method report](https://github.com/lachrymator/openadmet-cyp-challenge-public)
- Describes representation-diverse per-endpoint stacking and a strong interim TDI
  result (rank 6 of 114, within the organizers' top statistical tier).
- Useful conclusion: TDI benefits from diverse representations and endpoint-specific
  thresholds; the report is not a drop-in implementation.

## Decisions encoded here

1. **Scaffold-disjoint CV by default.** Random splits are too easy for analogue series.
2. **Per-endpoint models first.** Sparse labels differ substantially across CYPs.
3. **Official metrics from day one.** R2 and ROC-AUC are diagnostics, not rank metrics.
4. **Class balancing plus threshold selection for TDI.** A 0.5 cutoff is arbitrary under
   roughly 20% prevalence and MCC scoring.
5. **Full provenance.** Every run records configuration, OOF predictions, hashes, and
   validator output.
6. **No structure work yet.** It is excluded from download, code, and roadmap milestones.
