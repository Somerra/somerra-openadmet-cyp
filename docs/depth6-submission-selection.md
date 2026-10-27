# Depth-six submission candidate

The user selected depth 6 after the four-hour block, overriding the conservative
promotion gate for this submission candidate. Its earlier internal-test score was
0.654596 versus 0.659996 for the reference; the difference was inconclusive under
paired resampling. The original hill-climbing task and incumbent stay unchanged.

For this preparation, combine the original training and validation partitions for
training. Use the original internal-test molecules as validation for selecting the
checkpoint. Preserve original assignments in a derived bundle; retain the original
bundle unchanged. Exclude validation and official blind-test parents across every
training source, including auxiliary tasks. Fit target scalers on eligible training
rows only. Retain analogues and existing record weights.

Train one fresh model with the saved depth-six recipe, seed 42, batch 128, 100 epochs
and scheduler horizon 200. Select the lowest validation point MA-ST-RAE checkpoint
and export its predictions directly. Do not refit using the new validation labels.
The new validation score is a model-selection score, not an independent test score.
Only the official challenge supplies the official blind-test score.

`python -m somerra_openadmet.submission_selection prepare --bundle BASE_BUNDLE
--strategy DEPTH6_MODEL_JSON --raw RAW --output NEW_OUTPUT --ledger SHARED_LEDGER
--hours 0.4` creates the derived bundle, supervises one fit with at most one recovery
inside the original allowance, and validates the prediction CSV. The persistent
service also caps total lifetime. Existing epoch/checkpoint/curve code is reused.

Export is explicit: `selected_validation=True` accepts only a completed fit on the
submission-selection protocol and loads best.ckpt. The default export path still
requires a final fit. The loader rejects final=True for this selection bundle so
validation cannot silently be folded into training. Predictions and metadata remain
private. Open Code is false and the model-report link is empty. Preparation does
not upload to the challenge.


## Prepared data audit

The derived bundle has 3,924 official training records (original train plus
validation) and 981 validation records representing 980 molecular parents (the
original internal test). Including eligible auxiliary records, fitting uses
31,928 records / 31,895 parents. The previous fitting pool had 30,948 records;
exactly 980 original validation records have been restored. New validation and
blind-test parents remain excluded across the complete training pool.
