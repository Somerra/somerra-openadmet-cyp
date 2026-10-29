# Default regression workflow: Workbench scaffold folds

Adopted on 2026-10-29, until the user changes direction. Future regression
submission candidates use the preserved Workbench scaffold assignments. The main
experimental variables are model architecture, task-loss weights, and the weights
used to combine models. Prefer developing CYP-specific models with auxiliary
prediction tasks over relying exclusively on one model for all four endpoints.

This default follows our [October 28 reproduction](briford-run-20261028.md), which
scored 0.4349 official MA-ST-RAE. That result motivates the choice; it does not
isolate a causal benefit from the split strategy. The historical Workbench recipe
is the reference here, not the changing upstream default.

## What is fixed, and what can change

| Keep fixed for a matched comparison | Experiment with |
|---|---|
| Source snapshots, row identities, labels, and source-specific task definitions | Molecular encoder and prediction-head architecture |
| Saved scaffold/fold assignments for the selected data bundle | CYP-specific versus shared prediction heads |
| Four training folds and one checkpoint-selection/OOF fold | Relative weights of scored and auxiliary losses |
| Training-only fitted target scalers; masked missing labels | Weights combining completed model configurations |
| Scoring on official direct-inhibition labels, with the shared evaluator | Regularization and training policy, as separately declared changes |
| Official blind identities and prediction-only submission policy | Auxiliary-task selection within a declared, fixed bundle |

Changing a data source or regenerating folds is a separate protocol change, not
an architecture improvement. Record any training-policy change explicitly; a new
encoder may need a different optimizer, but changing both does not isolate the
encoder's effect. Do not silently compare the old Mol-JEPA Butina study or the
60/20/20 molecular task against these folds as if they were matched experiments.

## Data bundles and tasks

These are four model views of the source measurements, not four independent
datasets. Source provenance and joins are described in
[briford reproduction](briford-reproduction.md#data-and-preserved-identity).

| Saved bundle | Records | Outputs | Role |
|---|---:|---:|---|
| `union18` | 31,670 | 18 | Four official pIC50 tasks, four official single-concentration log2fc tasks, five ChEMBL CYP potency tasks, five Veith response tasks |
| `mt_aux8` | 4,905 | 8 | Four official pIC50 tasks and four official log2fc tasks |
| `d6_isoform6` | 23,264 | 6 | CYP2D6 official pIC50, log2fc, TDI-condition pIC50, Emax, ChEMBL potency, and Veith response |
| `d6_single1` | 1,493 | 1 | CYP2D6 official pIC50 only |

Only the four official direct-inhibition pIC50 endpoints determine the regression
score. Auxiliary measurements keep separate output heads; they are not substitute
labels for official potency. A specialist can predict one scored endpoint and
several auxiliaries. Specialization does not mean single-task training.

The original preparation drops rows with no labels for a bundle, then constructs
its folds. It retains one official blind connectivity with a Veith auxiliary
measurement for historical parity, recorded in the preparation manifest; no
official blind potency labels are available or used. Preserve and report this
exception when comparing against that reference rather than silently changing it.

## Exact split construction and reuse

The implementation is `data_sources/workbench.py::scaffold_folds`. It is checked
against Workbench revision `8c72c7559471febaf0962fd551b4eaab8db3a136` during
preparation. The union18 task definition comes from revision
`971f6a7f24d86e59bab219347c3e14c006ae4e35`.

1. Read the bundle's ordered, prepared SMILES rows.
2. Extract each RDKit Bemis–Murcko scaffold and convert it to canonical SMILES.
   Invalid structures fail; there is no random-split fallback. Empty scaffolds
   also form a group under this implementation.
3. Number scaffold groups in order of first appearance.
4. Shuffle the group IDs with NumPy `default_rng(42)`.
5. Visit each whole group and assign it to the fold with the fewest rows so far.
   Ties go to the lowest fold index. There are five folds, numbered 0–4.

This balances row counts approximately, not labels or molecular similarity to the
official blind set. Row order, bundle membership, and RDKit behavior can affect
the result. A seed alone is not sufficient to preserve the split.

**Reuse saved assignments, not just the splitter.** The reference artifacts are:

```text
/mnt/data/somerra-openadmet-runs/briford-20261028/
  data/{union18,mt_aux8,d6_isoform6,d6_single1}.csv
  data/preparation.json
  contract.json
  fits/<member>/<fold>/
```

Each CSV records `molecule_name`, `smiles`, `connectivity`, `scaffold`, `fold`, and
task columns. A new experiment should record the source manifest and CSV hashes,
then verify identities and assignments against them. Never overwrite the reference.

Each bundle has its own partition: fold 0 in union18 is not necessarily fold 0 in
mt_aux8. Within a bundle, every model family uses the same assignments, including
all auxiliary observations for each row. Do not split an auxiliary table
independently and accidentally train on a held-out molecule's other assays.

For a new specialist derived from an existing bundle, inherit that bundle's fold
column before masking unused outputs or dropping rows without selected labels.
Do not rerun scaffold assignment after filtering. Record the parent bundle, row
mask, task mask, and per-fold label counts; reject an untrainable head. A matched
Chemprop control must use the same filtered rows and inherited folds. Extending
specialists to other CYPs is new work, not part of the historical 20-model run.

## Training, validation, and test roles

For fold k, train on folds other than k (approximately 80%) and use fold k
(approximately 20%) for checkpoint selection. Repeat for all five k values.
The reference initializes fits with seeds 42+k, fits target normalization on the
training rows, and applies that scaler to validation. Its weighted MAE
`val_loss` selects the best checkpoint: maximum 400 epochs, patience 40.
The selection loss includes the selected auxiliary tasks; it is not MA-ST-RAE.

Predict each held-out row using only its corresponding selected checkpoint.
Concatenating these predictions produces one OOF prediction per eligible row per
configuration. Report MA-ST-RAE, each endpoint's ST-RAE, label counts, and per-fold
results using official direct-inhibition labels. Retain curves and selected epochs.

The same held-out fold selects the checkpoint and supplies OOF scores. Consequently
OOF is **development evaluation**, not an untouched internal test. Repeated model
or weight selection on OOF adds adaptivity. No new confirmation split is required
by this default. The 750 supplied blind molecules remain the official external
test; their labels are unavailable locally. Only the challenge supplies an
official score.

When comparing candidates, pair predictions by molecule and endpoint within the
same parent bundle and fold. Across bundles, join OOF predictions by identity,
not fold number; there is no shared global held-out fold. OOF averaging combines
one held-out prediction per configuration, whereas blind inference averages all
five fold models per configuration. Those are different evaluation objects.

## Two different kinds of weights

**Task-loss weights** affect training. In the historical union18 and mt_aux8
recipes, each scored task has weight proportional to inverse label count,
normalized so the four scored weights average one. Each auxiliary has weight
0.3. These fixed weights are calculated during preparation from the full member
table's label counts, including held-out label availability; target scalers are
still fitted on training rows only. CYP2D6 specialist weights are 1 for its scored
task and 0.3 per auxiliary. Preserve these as the comparison baseline.

Changing a loss weight can also change gradient scale and which checkpoint wins.
Record the masking rule, target scales, reduction denominator, and validation
selection loss along with the weight vector. A weight of 0.3 is not a claim that
auxiliaries contribute 30% of the total loss: label frequencies and errors matter.

**Model-combination weights** affect inference. The reference first averages five
fold predictions equally within each configuration, then combines configurations:

| Endpoint | union18 | mt_aux8 | d6_isoform6 | d6_single1 |
|---|---:|---:|---:|---:|
| CYP1A2, CYP2C9, CYP3A4 | 0.5 | 0.5 | 0 | 0 |
| CYP2D6 | 0.25 | 0.25 | 0.25 | 0.25 |

Thus those first three endpoints use ten networks each; CYP2D6 uses twenty.
Five fold fits are not five repeat seeds on identical training data. There is no
additional all-data refit in this protocol. Use these combination weights as the
baseline, while treating alternative endpoint-specific model weights as an
explicit experiment. Weights selected on OOF must be reported as tuned on that
development set, not independently validated gains.

Calibration is a separate final step. Preserve raw predictions and the historical
calibration rule for matched comparisons; do not silently retune it during an
architecture or weighting experiment. Its constants incorporate historical
leaderboard feedback. The 0.4349 score measures the complete calibrated pipeline,
not the raw model alone. Details are in
[calibration and verification](briford-reproduction.md#calibration-and-verification).

## Next Mol-JEPA comparison

Use these bundles and inherited folds, with auxiliary outputs and masked losses,
to compare Mol-JEPA against matched Chemprop controls. Start with the existing
CYP2D6 specialist bundle: it already supplies a concrete specialist reference and
six tasks. Keep the default task and fold-combination weights initially. Then
compare architecture and task weighting separately before extending specialists
to other CYPs or changing the final model combination.

The previous Mol-JEPA submission used five Butina folds, four scored tasks only,
one configuration, and fixed 300-epoch fits. It does not implement this protocol.
The current representation trainer needs a bundle/fold adapter, auxiliary-head
support, and validation-selected checkpointing before it qualifies as a matched
Workbench experiment. No new fit or submission is launched by this documentation.

## Entrypoint and evidence

`autoresearch.py` remains the research entrypoint. The existing implementation of
the historical reference is `autoresearch.py briford prepare|run|status|predict`;
see [execution commands](briford-reproduction.md#commands-and-execution).
It is currently specific to the four Chemprop configurations, not a generic
specialist/Mol-JEPA runner. Future model adapters must share the saved data, folds,
and evaluator rather than creating their own evaluation protocol.

The bare `autoresearch.py run` command still routes to the older frozen molecular
task. This documentation changes the research default, not CLI dispatch or old
task contracts. Existing studies remain reproducible under their recorded protocol.

Before comparing a new run, retain its data/fold hashes, task/record weights,
training recipe, implementation revision, per-fold checkpoints and curves, OOF
predictions, blind predictions, combination weights, and calibration parameters.
Verify checkpoint reload parity, complete identities, finite predictions, and the
750-row submission schema. Continue using the existing allocation ledger for its
authorized campaign; a protocol change does not reset spending.

Submission is separate and requires an explicit instruction. Follow `AGENTS.md`:
prediction CSV only, Open Code false, empty report link, no code/checkpoints or
links, cooldown respected, receipt preserved. Historical one-time upload
authorizations are not reusable.
