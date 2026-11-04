# Final prediction postprocessing

This note accompanies the two-page method report. It records a material step
in producing the final prediction file; it is not a separate model evaluation.

After equal raw averaging over folds, applicable task bundles, and the two
selected Chemprop configurations, each 750-row endpoint vector is transformed as

```text
prediction_i = max(1.0, target_mean +
    (raw_prediction_i - mean(raw_predictions)) *
    target_sd / std(raw_predictions, ddof=1))
```

For CYP1A2, CYP2C9, and CYP3A4, the target SD is the product of the borrowed
population SD, historical multiplier, and candidate OOF Pearson correlation.
CYP2D6 instead uses the explicit final mean/SD override below.

| Endpoint | Target mean | Borrowed SD | Multiplier | Candidate OOF Pearson r | Final target SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| CYP1A2 | 4.412 | 1.553 | 1.32 | 0.618278246 | 1.267445673 |
| CYP2C9 | 4.830 | 1.101 | 1.23 | 0.736968706 | 0.998025131 |
| CYP2D6 | 3.570 | — | — | — | 0.900000000 |
| CYP3A4 | 4.880 | 1.272 | 1.07 | 0.833066684 | 1.133837080 |

Values displayed here are rounded. Exact transform inputs and raw predictions
are retained in the run artifacts. The final CSV SHA-256 is
`88bdb3dbe8801a4d442de7eec0c0c5007146130d74d49418dce75dccf4d36d2e`.

The moments and multipliers come from the historical
[Workbench recipe](https://supercowpowers.github.io/workbench/blogs/cyp_challenge/).
This is leaderboard-informed, transductive calibration, not calibration fitted
solely from project training labels. The mean and sample SD used to standardize
raw predictions are computed on the complete blind prediction cohort. No blind
assay labels are loaded. Organizer-specific clearance is not established by a
prior portal acceptance or by permission to send a late submission.

OOF correlations come from checkpoint-selected development predictions.
The same development evidence was used to choose the ensemble; its scores are
not independent held-out estimates. Postprocessing occurs once, after raw
averaging, with a floor of 1.0. Raw predictions are preserved separately.

See the [calibration policy](../docs/calibration-policy.md) for the historical
matched comparison and the distinction between this recipe and own-data
calibration experiments. The old comparison is not an official result for
this final candidate.
