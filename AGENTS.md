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
