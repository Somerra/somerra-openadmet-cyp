# Standing challenge submission policy

The user requires prediction-only submissions for this challenge.

- Keep this repository and our implementation private.
- Do not emit or upload code, model checkpoints, code links, or report links through
  challenge submissions, leaderboard metadata, public reports, or organizer messages.
- Set `open_source_code` / the portal's Open Code checkbox to false.
- Leave `model_report_link` / Method Report Link empty. This includes third-party
  reference repositories and attribution links, not just our own repository.
- Keep source attribution and reproducibility evidence internally in the private
  repository and local run artifacts.
- If a submission requirement conflicts with this policy, stop and explain the
  conflict rather than publishing anything or weakening the policy.
- Respect the portal's cooldown and preserve submission receipts. A new submission
  does not imply that earlier metadata has been deleted.

This policy was explicitly established by the user on 2026-10-24 and remains in
force unless the user changes it.

## Default regression research protocol (2026-10-29)

Until the user changes direction, use the preserved Workbench scaffold folds for
future submission candidates. Follow [the default protocol](docs/workbench-protocol.md)
for bundle-specific assignments, training/validation roles, OOF evaluation, and
ensemble construction. Reuse saved row assignments; do not substitute Butina,
60/20/20 molecular, or Scigantic CheMeleon splits for a matched comparison.

Focus experiments on architecture and task/model weights. Work towards CYP-specific
models with auxiliary tasks, including Mol-JEPA specialists. A specialist may have
multiple auxiliary outputs. Keep data preparation and evaluation shared across
model families. Preserve historical artifacts and contracts. This direction does
not authorize an upload or change the prediction-only policy above.

Follow the [regression calibration policy](docs/calibration-policy.md), documented
on 2026-11-02 after a matched official comparison (raw 0.7151; calibrated 0.4612).
Prepare historical Workbench calibration alongside an immutable raw baseline for
future regression submission candidates. Apply it once after model averaging;
reuse saved predictions and OOF evidence without retraining for calibration alone.
Record any departure as a separate experiment. Preserve raw research comparisons;
this default does not replace raw strategy scoring or authorize another upload.

Calibration policy updated by explicit user direction on 2026-11-01:
“Use the historical Workbench constants; change the standing hold.” The prior
project hold on those constants is lifted. Prepare the next regression candidate
with the historical Workbench formula, including borrowed population moments,
OOF correlation multipliers, the CYP2D6 mean 3.57 / SD 0.90 override, and floor 1.0.
Retain raw predictions and exact transform/input provenance as separate artifacts.
This is leaderboard-informed calibration, not calibration learned solely from our
training labels; user authorization does not establish organizer-specific clearance.
Own-data calibration remains a separate experiment axis. Report checkpoint-selected
OOF and OOF-tuned results as development evidence, not independent held-out estimates.
Do not contact organizers without authorization. This policy change authorizes
preparation, not a new upload; retain prediction-only metadata and portal cooldown.

The user also authorized a separate hit-expansion-style evaluation. Preserve all
saved Workbench folds and historical contracts; do not replace them. Seed-hit
selection uses only training seed-pool labels. Candidate analog selection uses
structures, not held-out assay labels. Exclude evaluation/blind parents from all
training sources; keep validation/checkpoint selection, calibration fitting and
strategy-scoring roles distinct. Report sparse assay coverage and prior adaptive
exposure, and require fresh fits for the new split.

## Third-party reference predictions

Scigantic reference (R25) is a reconstruction of third-party published predictions,
not a model trained by this project. Keep it attributed as an external reference.
Its resubmission is on hold pending clarification of prediction-reuse eligibility
and attribution requirements. A validated file or prior acceptance is not clearance
to resubmit. Do not contact organizers without user authorization. If required
attribution conflicts with the submission policy above, stop and explain the conflict.


## CYP autoresearch selection policy (2026-11-01)

The user clarified the two-level selection policy. This supersedes earlier CYP
recipe instructions that used validation to rank strategies and reserved internal
test for a single report after closing the campaign.

- Preserve saved Workbench assignments: train on folds 1, 3, and 4; validate on
  fold 0; score strategies on internal-test fold 2. Exclude both held-out parent
  sets and official blind parents from every training source, including auxiliaries.
- Training fits model weights and preprocessing. Validation monitors training,
  early stopping, and checkpoint selection within each run, using primary ST-RAE.
- Freeze the validation-selected checkpoint for each completed trial, then score
  that checkpoint on internal test. Lower primary internal-test ST-RAE ranks
  strategies and selects the incumbent independently for each of the four CYPs.
  Internal-test feedback may guide subsequent architecture, loss, task-weight,
  batch-size, learning-rate, and other strategy experiments within the allocation.
- Never use internal test to choose an epoch/checkpoint within a trial, fit model
  weights or scalers, or directly fit calibration parameters. Preserve separate
  validation and internal-test metrics and the exact recipe/checkpoint provenance.
- Repeated strategy selection makes internal-test scores adaptive development
  results, not an unbiased or independent held-out estimate. Report prior use of
  these molecules. Official competition test labels/scores do not enter research;
  the official competition test remains the final external evaluation.
- Freeze explicitly selected recipes before five-fold submission construction.
  Keep prediction-only submission policy and authorization boundaries unchanged.

Use `python -m somerra_openadmet.cyp_strategy trial|score|status` for adaptive
research. Per-trial internal-test reports rank strategies in
`strategy-registry.json` without closing the campaign. The original training
registry and legacy W&B best/campaign summaries remain validation diagnostics
for replay compatibility, not strategy-selection authority. Historical closed
campaigns and trial records remain unchanged.

## Scope of autoresearch evidence

Treat selected recipes as hypotheses specific to their recorded split, data,
evaluator, and adaptive search history, not universal CYP defaults. Report repeat
variability alongside the best observed score. New splits require fresh matched
controls; do not transfer incumbent scores or pool incompatible evaluations.
Keep research scoring, checkpoint-selected OOF, and official submission results
separate. See [the first campaign learnings](docs/autoresearch-learnings-20261101.md)
for the evidence and operating policies.
