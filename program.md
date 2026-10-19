# Regression research protocol

Inspired by [autoresearch](https://github.com/karpathy/autoresearch): a small editable
recipe, a fixed evaluation harness, and a local keep/discard/crash record. This is
an agent protocol, not a daemon. `train.py` executes one trial; it does not edit
itself, decide promotion, or launch another trial.

## Agree the experiment first

Work on an isolated research branch. Preserve unrelated changes. Before training,
obtain a finite trial count and compute allocation, choose ONE research axis,
record the baseline run, and freeze the evaluation contract. Do not infer permission
for an unlimited loop from this file. Stop at the agreed limit or a blocker.

| Axis | Editable surface | Hold fixed |
| --- | --- | --- |
| Model usefulness | Model knobs in the `train.py` recipe block | Recipe's `data_sources`, catalog and snapshots, folds, labels, metric, evaluation code, resource limits |
| Data usefulness | Only `data_sources` in the recipe / `MODEL_OVERRIDES` | All other model/training settings, checkpoint, catalog and snapshots, folds, metric, resource limits |

Do not change both axes in the same comparison. For a data trial, use a neural
recipe that supports auxiliary targets; CPU evaluation rejects auxiliary selections.
Choose that recipe before establishing the matched baseline, not after looking at
data-trial results. Source-specific ablations can select any subset of the named
sources through the recipe's `data_sources`. One shared `configs/data.json` exposes
all source locations. Do not create per-trial catalogs or maintained model configs.

## Fixed infrastructure

`src/somerra_openadmet/data_sources/` owns acquisition/preparation and exclusion
rules. `campaign.py`, `neural.py`, `metrics.py`, and `regression_report.py` own
evaluation. These replace autoresearch's fixed `prepare.py` role; do not edit them
during a research cycle. Infrastructure changes require a separate reviewed change
and fresh matched baselines. Pin prepared snapshots and embeddings by their hashes;
do not refetch sources mid-cycle. Keep all five molecular-cluster folds, confidence
bounds, exact/analogue exclusions, inner-only ensemble selection, and all endpoints.

Unlike a five-minute language-model trial, a partial-fold chemistry score is not a
comparable result. Require full-fold completion. Freeze epoch/step policy, threads,
memory and per-trial time allocation before the cycle. Neural runs enforce their
configured GPU allowance; CPU runs currently have no built-in wall-time governor.
Use a supervised timeout when required and record timeouts as crashes, never wins.
More examples at fixed epochs also means more optimizer steps: report this caveat
for data comparisons, or explicitly design a matched-step experiment separately.

## One trial

1. Run the unchanged baseline with the same harness and environment first.
2. State one hypothesis; change only the selected axis. Run from the repository root:

   ```bash
   python train.py --output-dir runs/regression-trial-001
   ```

   The runner saves the effective recipe, catalog, and editable script in an ignored
   sibling `-inputs/` directory. Evaluators retain resolved configuration, hashes,
   predictions and metrics. Use a fresh output path for every attempt. Keep neural
   outputs directly under the shared `runs/` directory, not separate budget roots.
3. Check completion, finite scores, coverage, input hashes, and unchanged evaluator.
   Compare paired OOF predictions against the predeclared baseline with
   `regression compare --run-dir ... --reference-run ... --reference ...`.
   CPU reference column: `nested_ensemble`; neural reference column: `y_pred`.
4. Append one row to ignored `results.tsv` with columns:
   `commit`, `axis`, `run`, `reference`, `model_hash`, `data_hash`, `score`,
   `elapsed_seconds`, `status`, `description`. Status is `keep`, `discard`, or
   `crash`. Retain all run evidence locally, including failures; do not commit every
   trial recipe or log. A crash has no usable score. Never compare partial runs.
5. Keep only a justified improvement (or equal performance with less complexity)
   in the research branch. Revert only your own rejected recipe edits; never use a
   destructive repository reset. Update the maintained best recipe when warranted.
   Stop when the approved trials or compute allocation are exhausted.

## Keep is not promote

Repeated OOF comparisons are adaptive model selection, not independent validation.
Research `keep` does not change the incumbent or authorize export/upload. Promotion
still requires the campaign's 3% macro gain, paired 95% delta bound below zero,
no endpoint regression above 0.02, complete provenance, and remaining validation
gates. The frozen-feature recipe is the best observed challenger, **not promoted**.
External data has not shown a clear benefit in the matched comparison. Preserve
these distinct conclusions; availability of a source is not evidence of usefulness.
