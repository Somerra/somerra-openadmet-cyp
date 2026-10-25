# Scigantic data and auxiliary-task ablation

The 2026-10-24 campaign studies data usefulness with one fixed neural model. It
is not an ablation of the R25 ensemble or an exact retraining of its members.
The completed study has 54 configurations and 270 fold models. The first phase
used 2.7852 charged hours of its three-hour allocation; the 42-run continuation
used 10.6152 of 12 additional authorized hours. Prediction export used another
0.00113 hours. CPU preprocessing and startup are charged conservatively.

## Source and task boundaries

Official labels stay in the official bundle. The recipe selects direct inhibition
(always present), four single-concentration response heads and four TDI-condition
potency heads. Emax is available in the bundle but is not part of Scigantic's
26-head design and is excluded from this campaign.

The pinned Scigantic derivative table supplies five ChEMBL potency heads, five
PubChem maximum-response heads and four PubChem AC50 heads. CYP2C19 appears in
ChEMBL potency and PubChem maximum response, but is not a scored endpoint.
`data_sources/scigantic.py` imports auxiliary columns only, checks the upstream
revision and tracked table, standardizes parents, aggregates duplicate parent/task
labels by median, and records source hashes. It does not import upstream direct
labels, folds or uncertainty weights. Raw source acquisition remains deferred;
the published derivative's raw assay processing is not independently verified.

The import removes blinded-test parents and ECFP4 analogues at Tanimoto >= 0.6.
The shared training loader separately removes auxiliary parents and analogues of
each outer validation fold. Per-task coverage is recorded after these exclusions.
This is intentionally stricter than the upstream frozen reconstruction, which
retains one blind molecule with auxiliary observations.

## Frozen experiment

- Random width-700, depth-5 Chemprop encoder; norm aggregation; batch normalization;
  two 300-wide hidden layers; dropout 0.1.
- Four direct heads use interval MAE; all auxiliary heads use MAE on training-only
  standardized targets. Published direct-head weights remain fixed; every selected
  auxiliary head has weight 0.3. The number of output heads follows the selection.
- AdamW, learning rate 0.0003, weight decay 0.0001, batch 64, gradient clip 5.
- Fifteen epochs, seed 42 plus fold ID, all five original folds. No validation-based
  stopping or epoch selection. This shortened schedule is a screening budget,
  distinct from the prior 60-epoch loss comparison.
- The source hashes are frozen before the first trial. The same autoresearch
  reproduction entrypoint and evaluation backend run every cell.

The order is full model, direct-only, each of five auxiliary groups alone, then
five leave-one-group-out configurations (12 total). Next are 22 leave-one-head-out
configurations. The remaining 20 group combinations complete the 32-cell group
factorial, bringing the total to 54 if time permits. A failed or incomplete run
stops the runner for investigation. Partial folds are never scored as results.

## Interpretation

Compare each completed configuration with the matched full and direct-only runs
using the shared metric and paired molecular-cluster bootstrap. Individual intervals
are exploratory: they are not corrected for the many comparisons and do not capture
training-seed variability. A promising selection needs independent confirmation.

Fixed epochs mean different dataset sizes receive different optimizer-step counts;
the manifests record both. Removing a source also removes molecules with no remaining
labels. Thus these tests measure the practical value of a source/task selection,
not purely the value of extra labels at fixed molecule exposure. Separate same-parent
and matched-step experiments would be required to identify those mechanisms.

No configuration is automatically promoted or submitted. The prediction-only
submission policy remains unchanged. Trial recipes, data, checkpoints and results
remain private ignored run artifacts, not maintained recipe catalog entries.

## Evidence

Durable artifacts live under
`runs/campaign-20261024-data-ablation/` in the original workspace: `contract.json`,
`data.json`, `external/`, `recipes/`, `evaluator-hashes.json`, per-trial logs and
manifests, `results.json`, and `completion.json` once execution stops.

## Completed results

All 54 configurations finished on 2026-10-25 at 07:14 UTC: 32 group combinations
and 22 individual-head removals. Integrity checks verified 270 checkpoint hashes,
fixed evaluator/model settings, and all 6,525 official observations per configuration.
All scores below are local validation MA-ST-RAE, not official test scores.

| Configuration | Validation error |
| --- | ---: |
| Direct labels only | 0.763399 |
| All five auxiliary groups | 0.679435 |
| All except TDI | 0.666293 |
| Single-concentration + ChEMBL + PubChem response | 0.669898 |

All auxiliary groups together reduced validation error by 11.0% versus direct-only.
Across the full group factorial, exact Shapley error reductions were 0.02415 for
PubChem response, 0.02243 for ChEMBL, 0.01902 for single-concentration, 0.01260 for
AC50 and 0.00576 for TDI. These are descriptive averages over combinations, not
independently validated rankings of data sources.

Removing TDI improved the full model by 1.93%; the nominal paired delta interval
was [-0.02525, -0.00056]. This does not meet the 3% local promotion threshold and
is not corrected for selection across 54 candidates. TDI helped in 10 of 16 group
contexts. No improving single-head removal had a nominal interval excluding zero.
Same-parent, matched-step, repeat-seed and longer-training controls remain unrun.

## Official test results and investigation

The official test is the fixed 750-row blinded challenge set. Our validation folds
are locally generated Butina clusters within official training, not organizer folds.
Eligible auxiliary and external sources supplement training under the exclusions
above. The reconstructed R25 prediction ensemble scored **0.4436** on the official
test. Our without-TDI screening model, submitted at 01:27 UTC on 2026-10-25, scored
**0.7790**. R25 remains the best officially evaluated submission of these two.
A numerical gate against the local CPU reference did not establish improvement
over R25; future reporting must distinguish these references.

Investigation found no export/alignment defect: sampled CPU checkpoint predictions
matched saved OOF predictions within 4.6e-7 and sampled five-fold exports matched
within 1.9e-7. R25 is a materially different ensemble with stronger training,
pretrained members and calibration. Our CYP2D6 blind predictions had mean/SD
4.708/0.344 versus R25's 3.576/0.865. This is a discrepancy, not proof of causation.

Official train-to-test median ECFP4 nearest-neighbor similarity was 0.587 versus
0.435 for local train-to-validation; scaffold overlap was 24.5% versus 7.3%.
Using the smaller fold-training references for the blind set still gave median
0.571. Thus greater blind-test chemical novelty is not supported by these measures;
assay/label shifts remain possible. Analysis covers official direct training only.

The latest submission uploaded predictions only, with Open Code false and the
report field empty. Earlier historical submission metadata is not erased by it.
Automatic post-campaign uploads are disabled. No further submission is implied by
completion of this study. See [next steps](regression-next-steps.md).

## Private evidence

Under the ignored campaign directory: `continuation-01/completion.json` is the
final completion record; the top-level `completion.json` is historical phase one.
`integrity-audit.json`, `group-factorial.csv`, `group-shapley.csv`,
`group-interactions.csv`, `individual-head-ablation.csv` and plots contain results.
`chemical-shift-audit.json`, `official-vs-local-splits.json`, and
`leaderboard-gap-investigation.json` preserve investigation evidence.
`submission-prepared-20261025/` retains the actual CSV, receipt and leaderboard
snapshot. Frozen source snapshots, operational scripts and trial recipes remain
ignored local artifacts, not a maintained catalog of every configuration.
