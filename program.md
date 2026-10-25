# Autoresearch campaign protocol

Inspired by [autoresearch](https://github.com/karpathy/autoresearch): a small editable
recipe, a fixed evaluation harness, and a local keep/discard/crash record. This is
an agent protocol, not a daemon. Start with `python autoresearch.py --help`.
`autoresearch.py plan` previews a trial and `autoresearch.py run` executes one.
Neither edits the recipe, decides promotion, or launches another trial.
`train.py` is the editable recipe; the launcher, `research.py` and `evaluation.py`
stay fixed. An autoresearch agent runs the campaign by repeatedly invoking these
tools within an approved budget. This CLI does not itself implement an autonomous
hypothesis generator or keep/discard loop.

## Reproducibility is complementary

Use `autoresearch.py reproduce --model-config PATH --backend cpu|neural
--output-dir NEW_PATH` to rerun a saved recipe, with `--preview` for a no-training
inspection. This path does not import `train.py` or inherit its overrides. It uses
the exact same `evaluation.evaluate_recipe` dispatcher and backend metrics as a
research trial. Backend and input snapshot hashes still matter; a recipe alone
does not guarantee bit-for-bit results across environments.

The maintained recipe catalog records the incumbent and best observed challenger.
Agents may propose catalog updates after reviewing results; neither execution mode
automatically promotes a recipe. Existing low-level comparison and export
commands support both workflows and are not a second autonomous campaign system.

## Agree the experiment first

Work on an isolated research branch. Preserve unrelated changes. Before training,
obtain a finite trial count and compute allocation, choose ONE research axis,
record the baseline run, and freeze the evaluation contract. Do not infer permission
for an unlimited loop from this file. Stop at the agreed limit or a blocker.

| Axis | Editable surface | Hold fixed |
| --- | --- | --- |
| Model usefulness | Model knobs in the `train.py` recipe block | Recipe's `data_sources` and `auxiliary_tasks`, catalog and snapshots, folds, labels, metric, evaluation code, resource limits |
| Data usefulness | Only `data_sources` / `auxiliary_tasks` in the recipe / `MODEL_OVERRIDES` | All other model/training settings, checkpoint, catalog and snapshots, folds, metric, resource limits |

Do not change both axes in the same comparison. For a data trial, use a neural
recipe that supports auxiliary targets; CPU evaluation rejects auxiliary selections.
Choose that recipe before establishing the matched baseline, not after looking at
data-trial results. Source-specific ablations can select any subset of the named
sources through the recipe's `data_sources`. One shared `configs/data.json` exposes
all source locations. Do not create per-trial catalogs or maintained model configs.

## Training, validation, and official test

The fixed challenge test set is `cyp-challenge-TEST-BLINDED.csv`. Never replace it
with a locally generated holdout or use its withheld labels for fitting or selection.
Official test scores come from challenge evaluation; local OOF scores are validation
scores and must be labeled as such. Keep official test scores and receipts separate
from validation rankings. A local validation winner is not an official-test improvement.

Construct validation folds only within `cyp-challenge-TRAIN_inhibition.csv`, grouping
standardized molecular parents so a parent cannot cross training/validation boundaries.
Our current five Butina-cluster folds are locally constructed, not organizer-provided
folds. Future validation designs require a new frozen contract and matched baselines.

Auxiliary official data and external sources may supplement training. For each fold,
apply the existing parent-overlap and analogue exclusion rules to supplemental data
against held-out validation molecules and the fixed challenge test set. Fit target
scalers, preprocessing, calibration, and model selection using training data only,
with inner validation where needed. Supplemental labels must not turn held-out
validation or test molecules into training examples. Record source provenance and
post-exclusion coverage. After selecting a recipe, final fitting may use all official
training data plus eligible supplemental data; preserve test exclusions.

Report the validation reference and the official-test incumbent separately. In
particular, beating a local CPU or short-training reference does not establish an
improvement over the R25 submission on the official test set.

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
   python autoresearch.py plan --output-dir runs/regression-trial-001
   python autoresearch.py run --output-dir runs/regression-trial-001 \
     --hypothesis "One predeclared model or data change"
   ```

   The runner saves the effective recipe, catalog, and editable script in an ignored
   sibling `-inputs/` directory. Evaluators retain resolved configuration, hashes,
   predictions and metrics. Use a fresh output path for every attempt. Keep neural
   outputs directly under the shared `runs/` directory, not separate budget roots.
   `trial.json` in the sibling input directory records `running`, `completed`, or
   `failed`, elapsed time and the score when valid. Completion is not a keep decision.
   A hard process kill may leave `running`; inspect backend evidence before recovery.
   A preview reads the recipe/catalog only; it does not check snapshot availability,
   hardware, checkpoint compatibility or backend-specific settings. The backend
   validates inputs during execution. A hypothesis is recorded text, not an edit.
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
Pinned Scigantic auxiliary derivatives are supported for matched data studies;
fresh raw-source acquisition remains deferred to issues #3 and #4. Observed benefits
are conditional on the evaluated model and validation design; availability alone
is not evidence of usefulness.

## Submission privacy

Follow the standing policy in `AGENTS.md`: predictions only; keep code private,
Open Code false, and the Method Report Link empty, including third-party links.
Keep provenance and attribution in private records. A leaderboard submission must
never publish code or code/report links without a new explicit user instruction.
