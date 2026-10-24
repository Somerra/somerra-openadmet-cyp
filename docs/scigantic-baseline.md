# Scigantic regression baseline and experiment plan

We have reproduced the published R25 predictions. We have not retrained R25 or
established its performance on our evaluation splits. Those are separate milestones.

## Verified reference

Source: [Scigantic/open-admet-challenge-cyp](https://github.com/Scigantic/open-admet-challenge-cyp/tree/e70a2920d611684c9aaff07400d951e2e79dd5f6),
pinned to `e70a2920d611684c9aaff07400d951e2e79dd5f6` (MIT-0).

Our CPU reconstruction on 2026-10-24 covered 750 molecules and all 3,000 predictions.
Maximum absolute disagreement with the published R25 CSV was **0.00000626091 pIC50**,
below the upstream tolerance of 0.0001. Bytes differ; the upstream README documents
Powell/BLAS numerical sensitivity. Runtime for the recorded verification and rescoring
was 8.61 seconds, with no GPU training.

- Submitted SHA256: `8fae881bfc3a343873d39dd1fc5dfd880f0de88a5da33da7c05232cb7896167f`
- Rebuilt SHA256: `bc5b8065790ce5e47e087c7a0f0a9ab5e9e0904e1c524a481d24f13491a3134d`
- Environment: NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1.
- Local evidence: `runs/scigantic-r25-baseline-v2/manifest.json` and `rebuild.log`.
- The first adapter attempt failed a schema check because the published CSV includes
  SMILES; its failed manifest is retained in `runs/scigantic-r25-baseline/`.

The author reports R25 leaderboard MA-ST-RAE **0.4436**. We verified the predictions,
not that score: blind labels are unavailable. Our historical local CPU score of
0.743511 uses different splits and cannot establish the size of an R25 improvement.
The published blend-fit OOF scores also reuse rows used to optimize the combiner.

Reproduce from the repository root with the project Python environment:

```bash
git clone https://github.com/Scigantic/open-admet-challenge-cyp.git runs/scigantic-reference/source
git -C runs/scigantic-reference/source checkout e70a2920d611684c9aaff07400d951e2e79dd5f6
python autoresearch.py reference \
  --source-dir runs/scigantic-reference/source \
  --output-dir runs/scigantic-r25-baseline
```

Use a new output directory on every run. The command checks the pinned revision and
script hash, rejects tracked modifications, copies the required inputs, and runs the
reviewed upstream builder with a 300-second timeout and one BLAS thread. It preserves
input hashes, package versions, agreement checks, logs and failure status. Frozen
members are rescored through our `metrics.soft_threshold_rae`. This command neither
imports the editable recipe nor trains models, promotes a candidate, or uploads a CSV.
It records `score_kind=frozen_submission_reproduction`, with no local research score.
Downloaded data and predictions stay ignored; external training loaders remain deferred.

## What we can learn from this submission

The actual member table is more informative than the high-level narrative: most arms
use randomly initialized Chemprop networks; arm C and ivC2048 use the CheMeleon
foundation initialization, while arm D uses an organizer CYP-finetuned checkpoint.
R25 combines twelve members plus a TDI-head member used for CYP1A2. It uses endpoint-
specific, regularized interval-loss blending and restores prediction moments to an
incumbent. Reproduction depends on that frozen incumbent as well as the members.

The 22-head training table combines official direct labels and single-concentration
responses with ChEMBL potency and PubChem screening responses/AC50. The 26-head table
adds official TDI-condition potency. The organizer-only `mt_aux8` member provides a
useful model recipe to adapt before reopening external-data work.

Our shared metric reproduces these frozen member scores on the **upstream reference
rows**, rounded to four decimals. These are diagnostics, not our baseline leaderboard.

| Member | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
| --- | ---: | ---: | ---: | ---: |
| Organizer-only mt_aux8 | 0.8007 | 0.5934 | 0.9019 | 0.4821 |
| Arm B, MAE + inverse variance | 0.8246 | 0.5841 | 0.8739 | 0.5922 |
| Arm B, MAE + uniform | 0.8604 | 0.6273 | 0.8922 | 0.4883 |
| Arm B, interval + uniform | 0.8304 | 0.5889 | 0.8966 | 0.4421 |
| CheMeleon, interval + uniform | 0.7885 | 0.6071 | 0.8883 | 0.4540 |

These columns cover 1,390 / 1,275 / 1,441 / 2,326 observations: 6,432 total, 93 fewer
than the official 6,525-label corpus. The adapter does not silently score a member on
even fewer rows: incomplete member coverage yields a null score. Different model
families and source combinations in this table are not controlled source ablations.

## Reproduction limits to resolve before training comparisons

1. **Coverage and splits.** Our audit identified 22 / 10 / 52 / 9 missing observations
   by endpoint; all 93 have finite official bounds. Shared labels match exactly.
   Their IDs and the hashed official input are recorded in
   `runs/scigantic-reference/coverage-audit.json` and `missing-reference-labels.csv`.
   The historical reason for their omission remains unverified. Upstream scaffold
   assignments differ from our molecular-cluster folds,
   and different upstream members derive folds separately. Frozen predictions cannot
   simply be relabeled with our fold numbers or used for a fully nested ensemble claim.
2. **Incomplete history.** `training/build_r08_baseline.py` references absent
   `analysis/` inputs and `family_panel`; some member specs contain historical paths.
   R25's added `g26_tdi_armB` has blind predictions but no published OOF predictions.
   We can reproduce the frozen submission, not the entire training lineage as shipped.
3. **Checkpoint provenance.** Audit the organizer-finetuned checkpoint's exposure to
   official labels before using it for local validation. Start with random initialization;
   use a separately audited foundation checkpoint for the initialization experiment.
4. **External rows.** Upstream discloses one blind molecule in the `union18` table with
   auxiliary response labels. Preserve frozen artifacts for reconstruction, but apply
   our own parent/analogue exclusions to any future retraining. Do not infer that
   external data helps from its presence in a successful ensemble.
5. **Selection.** The frozen blend's OOF fit scores are optimistic. Fit blending,
   calibration and regularization only inside each outer training partition. A
   cross-fit combiner on existing OOF predictions alone does not recreate fully nested
   base-model training.

## Trainable baseline to establish next

Adapt the organizer-only `mt_aux8` recipe into our shared evaluation backend. Keep one
OpenADMET data bundle, and let the model select four direct targets plus the four
single-concentration log2FC heads. The current auxiliary preparation retains separate
concentrations and other assay types. Our audit confirmed a single concentration,
4.95049505e-5 M, with all 4,375 values per enzyme matching the upstream eight-head table
to floating-point tolerance. Select those four heads explicitly; do not substitute all
our auxiliary heads. Parent aggregation remains our fixed policy. This requires a
reviewed loader selection and model-adapter change,
not just a new JSON recipe: our current neural backend hard-codes different architecture,
aggregation and loss settings.

Start from random Chemprop BondMessagePassing, hidden width 700, depth 5, dropout 0.1,
NormAggregation (norm 100), batch normalization, and a two-layer FFN of width 300.
Use batch size 64 and MAE with uniform sample weights; carry the frozen published task
weights in the recipe with their provenance. Upstream uses up to 400 epochs, patience
40 and a 1e-4 → 1e-3 → 1e-4 learning-rate schedule. Treat that as the recipe to adapt,
not a promise that our hardware budget permits the full run.

Keep our corrected parent representation, official label/bound snapshot, five outer
cluster folds, full 6,525-label coverage where finite bounds permit, and existing
MA-ST-RAE implementation. Use an inner cluster holdout for early stopping; the outer
fold must never select an epoch. Record per-fold scores, epochs, optimizer steps and
runtime, as well as the pooled endpoint scores. This will be an **adapted local
baseline**, not an exact reproduction of the author's scaffold-CV result.

Before that full run, use one predeclared training-only pilot to check finite masked
loss/gradients, speed, memory and checkpoint round-trip. It has no comparable research
score. Estimate a finite five-fold allocation from the pilot. Keep the same epoch/step
policy and resource limits for the baseline and each paired experiment. The observed
GPU was an H100 NVL with about 85 GB of 96 GB occupied; check capacity again before
launching. Share the existing absolute `SOMERRA_GPU_LEDGER` across worktrees.

## Experiment order

Freeze the baseline recipe and evaluation before selecting experiments. Each row is
a separate decision; this is a proposed queue, not authorization for an automatic sweep.

| Priority | Axis | Comparison | Hold fixed / decision |
| --- | --- | --- | --- |
| 1 | Model | MAE vs interval hinge, first with uniform sample weights | Data, initialization, architecture, folds and training schedule; verify masked finite gradients |
| 2 | Model | Complete the objective × weighting 2×2 with inverse-variance weights | Same four cells on all five folds; test the interaction instead of changing loss and weights together |
| 3 | Data | Direct-only vs +single-concentration vs +TDI vs +Emax | Fixed winning model recipe; select named assay heads from the same official bundle, report step counts |
| 4 | Model | Random vs audited CheMeleon initialization | Match width/depth first; changing width and initialization together is not a pretraining ablation |
| 5 | Model | Small complementary ensemble, then regularized interval-aware blend | Fully nested base fits and inner-only weight/calibration selection; compare against equal blend |
| 6 | Data | Revisit ChEMBL and PubChem separately | Implement provenance/exclusion checks through issues #3/#4; one source at a time, fixed model and schedule |

The upstream [2×2 result](https://github.com/Scigantic/open-admet-challenge-cyp/blob/e70a2920d611684c9aaff07400d951e2e79dd5f6/training/chemeleon_arms/INTERVAL_2X2_RESULT.md)
reported that interval loss improved MA-ST-RAE by 0.0275 with uniform sample weights
but worsened it with inverse-variance weights. This is a hypothesis worth testing in
our setup, not a guaranteed gain. The grouping report includes a divergent fold and
partial coverage, so it is not sufficient evidence to prioritize all external heads.

Require complete matched OOF predictions and paired cluster-bootstrap uncertainty.
Retain our existing promotion gate: at least 3% macro improvement, upper 95% delta
bound below zero, and no endpoint regression greater than 0.02. Track training cost
and per-endpoint behavior, especially CYP2D6. No leaderboard upload is part of this plan.
