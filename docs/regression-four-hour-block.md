# Four-hour regression experiment block

Run one bounded GPU block after PR #10, using the frozen molecular regression
objective and the reference scoring 0.659996 internally. This is an internal
strategy-development objective; official ranking correlation is still unverified.

Keep the data bundle, molecular exclusions, 60/20/20 assignments, seed 42, batch
128, 100 epochs, scheduler horizon 200, checkpoint selection and evaluation fixed.
Use one model per candidate. Reuse the saved reference and existing allocation
ledger. No reference refit, final fit, extra seeds, ensemble or challenge upload.

Recent task-normalization, auxiliary-zero and auxiliary-double scores were
0.739830, 0.941728 and 0.676694 respectively, versus 0.659996 for the reference.
These results motivate retaining the reference's observation normalization and
exploring explicit scored-task weight, optimization and capacity. Auxiliary-zero
retained molecules, batches and denominator; it was not a source-removal study.

## Proposed comparisons

The first 25 candidates change exactly one field from the starting incumbent:

| Field | Values |
| --- | --- |
| Scored endpoint weight | 2, 1.5, 4, 0.5 |
| Maximum learning rate | 0.0005, 0.002, 0.0003, 0.003 |
| Dropout | 0.1, 0.3, 0, 0.4 |
| Message-passing depth | 3, 5, 2, 6 |
| Message-passing width | 1024, 3072, 512 |
| Prediction-head width | 512, 2048 |
| Prediction-head layers | 1, 3 |
| Auxiliary weight | 0.75, 1.25 |

Follow-ups combine up to four distinct axes that improved validation, then refine
around the best validation recipe. If no single-axis trial improves validation,
refinement uses the reference. Finally, if time remains, interval loss with scored
weights 2 and 4 probes a predeclared interaction. Follow-up proposals do not use
internal-test scores. Deduplicate recipes; complete at most 40 candidates. Do not
rerun old losing configurations or identical reference recipes.

Internal-test comparisons still use the reviewed promotion gate: at least 3%
point improvement, no endpoint regression above 0.02, and paired 95% upper delta
below zero. Validation-driven proposals do not make adaptive internal-test
selection an unbiased final evaluation. The persistent task registry supplies the
recommended single model; if none qualify, retain the reference.

## Execution

Run `python -m somerra_openadmet.regression_block --task ABS_TASK --output NEW_DIR
--hours 4` with `PYTHONPATH=src` and the neural Python environment. Use a persistent
systemd user unit with `RuntimeMaxSec=14400` and `KillMode=control-group`; the unit
provides a hard four-hour wall-clock limit, including child processes. Run one GPU
candidate at a time. Each candidate uses the existing stall detector and one
checkpoint recovery within its original allowance. Stop after two consecutive
failures. Reserve up to 24 minutes per candidate from the shared ledger, bounded by
remaining block time; actual usage is settled on exit. Do not start a trial unless
its estimated duration plus a two-minute reporting margin fits. The limit is a
ceiling; do not spend time merely to fill it.

The coordinator stores resolved trial recipes and source identities in private run
artifacts, not the maintained config directory. It writes block.json and results.csv
after each trial, along with report.html, recommendation.json and best-model.json.
The HTML report works on mobile with a horizontally scrollable results table.
A hard service termination may leave the last block record marked running; check
systemd and the trial logs before resuming. Never reset the shared ledger.

Keep artifacts on the large local data disk: the repository disk lacks space for
40 checkpoint trees. Preserve original task/reference paths. The new block uses
the compatibility-checked task from PR #10; the reference's original fitted hashes
and explicit inference compatibility evidence remain intact.
