# Competitive roadmap

The repository starts with trustworthy controls, then adds complexity only when it wins
under chemistry-aware validation.

## Phase 0 — repository baseline (implemented)

- pinned official data and checksums;
- schema audit and strict submission validation;
- ECFP4 + RDKit descriptor baselines;
- Ridge/logistic, ExtraTrees, and LightGBM;
- scaffold-disjoint OOF predictions;
- official ST-RAE and MCC;
- TDI threshold selection and serialized models.
- cross-fitted, endpoint-specific ST-RAE ensemble weights.

## Phase 1 — validation hardening

- freeze one scaffold/ECFP-cluster selection split before model search;
- add repeated cluster CV and confidence intervals;
- compare models with paired bootstrap or repeated-fold tests;
- quantify performance by potency bin, credible-interval width, and nearest-neighbor
  similarity;
- add exact structure-overlap checks before ingesting external labels.

## Phase 2 — activity-first competitive models

- masked multitask Chemprop across the four direct-inhibition heads;
- shared encoder with separate assay/source heads for challenge, ChEMBL, PubChem Veith
  qHTS, and single-concentration/Emax readouts;
- CYP2C19 as an auxiliary representation task, never as a scored output;
- censoring and uncertainty-aware losses;
- fold-safe calibration and representation-diverse ensembles.

## Phase 3 — TDI differentiation

- shared molecular encoder for direct inhibition and TDI;
- direct/TDI potency-shift and Emax heads as auxiliary supervision;
- endpoint-specific threshold selection inside nested CV;
- explicit uncertainty and abstention diagnostics, even though final submissions require
  hard booleans.

## Promotion gates

A candidate advances only if it:

1. beats the incumbent macro metric on the frozen selection protocol;
2. does not rely on random-split gains;
3. preserves test blindness and source provenance;
4. passes local submission validation;
5. produces a replayable configuration and OOF predictions.

Leaderboard feedback is a sparse external check, not a hyperparameter optimizer wearing a
fake moustache.
