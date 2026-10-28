# Matched reference submission

Measure the depth-four reference on the official regression task, matched to the
submitted depth-six candidate. Use the same expanded training pool (original train
plus validation and eligible auxiliary records), former internal test for checkpoint
selection, seed 42, batch 128, 100 epochs, 200-epoch scheduler horizon, scaler fitting,
record weights and prediction export. Only message-passing depth changes from 6 to 4.
Select each model's own best validation checkpoint; do not refit on validation.

Verify identical derived data manifests and model/trainer source hashes before
comparing scores. Preserve the original development-task reference and its artifacts.
This run measures the reference architecture under the submission training protocol;
it does not submit the older reference trained on only 60% of the official labels.
The two official submissions will give a direct comparison for these fitted models,
not establish general correlation between internal validation and the leaderboard.

Prepare using the maintained incumbent recipe with the existing submission-selection
supervisor and shared ledger. Reserve at most 0.4 GPU-hours; use the same checkpoint
parity and 750-row CSV validation. Submission metadata must remain prediction-only:
Open Code false, no model report or code links, other optional text fields empty.
Preserve the prior depth-six score (0.8032) and submission receipt before uploading.


## Prepared result

The matched reference completed 100 epochs and selected epoch 91. Its validation
MA-ST-RAE is 0.66551114, versus 0.64404843 for depth 6. Both used the same data
manifest and model/training implementation; only message-passing depth differed.
The reference CSV has 750 validated rows, no errors and no warnings.
SHA-256: `2a4dcc0496ecafefa55aa0392c0168e5d7c23c43b876d4b61c9c0ecb5df5d880`.

Artifact directory: `/mnt/data/somerra-openadmet-runs/reference-submission-20261028`.
The reference has not been uploaded. Automatic approval review rejected the upload
because the request to measure it did not explicitly authorize submission of this
candidate. Submission approval is pending. No official score is available yet.
