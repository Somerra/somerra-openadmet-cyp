# Challenge-like hit-expansion benchmark

This is a parallel, retrospective regression benchmark. It does not replace the
preserved Workbench folds, launch training, or authorize a submission. Its first
version evaluates raw predictions only; the separately authorized historical
Workbench calibration workflow is unchanged.

## Scientific contract

The organizers expanded 25 potent hits from each of CYP1A2, CYP2C9, and CYP3A4
into ten similar compounds per hit, then measured the 750-compound test set across
all four CYPs. CYP2D6 did not drive hit selection. The training assay matrix is
sparse, while the official test matrix is dense. Analog series originating from
one hit remain together in the public/private leaderboard division.

Primary references, retained privately for attribution:

- [Challenge construction and leaderboard split](https://www.openadmet.org/blog/announcing-openadmets-cyp-inhibition-blind-challenge/)
- [Assay and tiering design](https://openadmet.github.io/octant-cyp-inhib-blog-post/)
- [Official dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)
- [Portal rules](https://huggingface.co/spaces/openadmet/cyp-challenge/blob/main/config.py)

Our proxy uses existing labeled library molecules in place of newly purchased
analogs. The published description does not verify the exact seed identities or
fingerprint settings. Ranking seeds by direct-inhibition pIC50 and using Morgan
fingerprints are explicit reconstruction assumptions, not exact organizer parity.

The fixed procedure is:

1. Standardize molecular identities using the repository's parent standardizer
   without stereochemistry. Remove official blind parents. Sort unique parents,
   shuffle with NumPy seeds 42–46, and reserve the first half as the training seed
   pool. All aliases of one parent receive the same role.
2. Rank only seed-pool labels. Select 25 unique parent hits per CYP1A2, CYP2C9,
   and CYP3A4 by descending direct pIC50, breaking ties by molecule ID. Preserve
   selections shared across endpoints and keep every selected seed in training.
3. Pass only structures to analog selection. Rank candidates from the other half
   using Tanimoto similarity on Morgan radius-2, 2,048-bit fingerprints without
   chirality. Retain up to ten nearest candidates per seed with similarity at
   least 0.45; ties use canonical parent order. Never pad with distant compounds.
4. Form connected components of the seed–analog nomination graph. Shared analogs
   merge their nominating hit families. Assign entire analog families to validation
   or test, largest first, breaking equal-size ties by smallest analog parent.
   Choose the less populated role, preferring validation on equal loads.
5. Keep remaining official molecules and eligible auxiliary/external records in
   training. Remove every evaluation/blind parent alias from every training source.
   Distinct close analogs remain eligible, including seed hits. Do not apply the
   older global analog-exclusion policy: that would remove the intended training
   context. Preserve auxiliary tasks as separate outputs.

The 0.45 cutoff was chosen during a structural feasibility comparison against
released blind structures. It is fixed, test-chemistry-informed design. Candidate
assay values, missingness, and model performance never select analogs or roles.
There is no search over splits during preparation. Label coverage is assessed only
after assignments are frozen.

## Prepare and audit

From the repository root:

```bash
.venv/bin/python -m somerra_openadmet.hit_expansion prepare \
  --source-manifest /mnt/data/somerra-openadmet-runs/hit-expansion-evaluation-v1-20261101/manifest.json \
  --workbench-root /mnt/data/somerra-openadmet-runs/briford-20261028 \
  --workbench-hashes docs/results/workbench-protocol-artifacts.json \
  --output runs/hit-expansion-close-analogs-v2-20261101

.venv/bin/python -m somerra_openadmet.hit_expansion audit \
  --root runs/hit-expansion-close-analogs-v2-20261101 \
  --output runs/hit-expansion-close-analogs-v2-20261101/audit
```

Existing output paths are rejected. For a new preparation or audit, choose a fresh
output path. Preparation verifies the pinned inputs, previous split CSVs, and all
historical Workbench hashes before and after writing. It stages the complete
bundle before publishing the output directory. Source snapshots must remain
available at their recorded paths for subsequent verification.

Each `repeat-42` through `repeat-46` directory contains:

| Artifact | Contract |
| --- | --- |
| `train.csv` | Eligible official/auxiliary/external records; sample weight starts at 1 |
| `validation.csv` | Official targets and credible intervals for checkpoint selection |
| `test.csv` | Official targets and credible intervals for frozen strategy scoring |
| `assignments.csv` | Official IDs, parent identity, seed flags, family, role; no assay labels |
| `seed-hits.csv` | Selected seed IDs, selection endpoints, and seed-pool pIC50 values |
| `analog-links.csv` | All retained nominations, ranks, similarities, families, and roles |
| `excluded-training-records.csv` | Removed source rows with evaluation/blind exclusion reasons |

The manifest records input and output hashes, parameters, versions, source
attribution, endpoint coverage, exclusion counts per task, and scientific limits.
Preparation source and evaluator code are copied into the private bundle. The
historical `workbench_fold` appears only in assignments for provenance; it must
not determine membership in this benchmark. Old fold/cluster fields are removed
from trainer tables. Use the shared role files across all model families and task
subsets; never resplit a filtered specialist table.

The audit exports nearest neighbors and similarity summaries against official
training, all training sources, and selected seeds. It also records scaffold
overlap, label/credible-interval coverage, family sizes, exclusions, repeated
evaluation exposure, and a parent-by-repeat role matrix. Similarity summaries
weight molecule rows; parent counts and duplicate-parent exposure are reported
separately. Empty Murcko scaffolds count as a shared scaffold in the diagnostic.

## Fresh fitting and evaluation interface

Training is a separate task. Each repeat requires new fitting that excludes its
test molecules from every learned component. Train-only rows fit target scalers
and count-derived task weights. Validation selects checkpoints and development
choices. Freeze the entire strategy before scoring test predictions. Do not reuse
historical challenge-trained checkpoints that have already fitted these rows.
External generic pretrained encoders remain eligible with recorded provenance.

The evaluator requires `repeat-42.csv` through `repeat-46.csv` in a predictions
directory. Each file must contain exactly the corresponding test IDs and finite
raw predictions for all four official endpoint columns, including unlabeled rows.
No calibration is fitted or applied by this evaluator.

Supply a separate `repeat-42.json` through `repeat-46.json` fit record directory:

```json
{
  "train_sha256": "<manifest hash for repeat-42/train.csv>",
  "validation_sha256": "<manifest hash for repeat-42/validation.csv>",
  "fresh_fit": true,
  "test_used_for_fitting_or_selection": false,
  "calibration": "none",
  "scalers_and_task_weights_fit_on": "train",
  "checkpoint_sha256": {
    "checkpoints/model.pt": "<actual checkpoint SHA-256>"
  }
}
```

Checkpoint paths are absolute or relative to the fit-record directory; list every
checkpoint used for the predictions. The evaluator verifies checkpoint bytes,
split hashes, prediction IDs, and the frozen metric implementation. Fit records
are trainer attestations, not proof of how training occurred.

```bash
.venv/bin/python -m somerra_openadmet.hit_expansion evaluate \
  --root runs/hit-expansion-close-analogs-v2-20261101 \
  --predictions /path/to/raw-predictions \
  --fit-records /path/to/fit-records \
  --output /path/to/new-evaluation
```

Report per-repeat endpoint metrics and counts. Fewer than 20 observed labels is
flagged as underpowered, not grounds to regenerate the split. Missing labels stay
missing. Undefined ST-RAE, including a zero denominator, is reported explicitly;
no endpoint is silently removed from a macro score. The five-repeat macro summary
is absent unless all five macro scores are defined. Cross-repeat ranges are
descriptive, not independent-sample confidence intervals.

## Limits

The close-analog benchmark has small holdouts, particularly for CYP2C9 and CYP2D6.
This can make validation checkpoint choice and normalized test metrics unstable.
It is an exploratory comparator, not a standalone reliable model-selection gate.
Existing sparse assays cannot recreate the dense official experimental panel;
requiring complete labels would introduce an additional selection bias.

The same molecule may be evaluated in multiple repeats and trained on in another
repeat. These are repeated retrospective experiments, not disjoint cross-validation
folds or five independent tests. Prior adaptive exposure remains even with fresh
fits. Repeated strategy selection using test scores adds further adaptivity.

Source code, checkpoints, reports, and attribution stay private. This module has
no submission path. Open Code remains false, report links remain empty, and portal
cooldown/receipt policies remain in force.

## Prepared artifact: 2026-11-01

The bundle is at `runs/hit-expansion-close-analogs-v2-20261101`; its manifest
SHA-256 is `95e45c59f2c16ac14291262ab7e8d5adcf63a5384b566997897cf85812a1b58e`.
The private run directory is ignored by Git. Its manifest retains the original
preparation implementation; the audit independently records its implementation
hash. No models were trained and no real model scores were produced.

| Repeat | Training rows, all sources | Validation | Test | Test labels: 1A2 / 2C9 / 2D6 / 3A4 | Median test-to-official-training similarity |
| --- | ---: | ---: | ---: | --- | ---: |
| 42 | 32,861 | 24 | 24 | 8 / 6 / 4 / 15 | 0.599 |
| 43 | 32,847 | 31 | 31 | 8 / 9 / 16 / 13 | 0.532 |
| 44 | 32,853 | 28 | 28 | 14 / 10 / 6 / 12 | 0.523 |
| 45 | 32,860 | 25 | 24 | 9 / 3 / 8 / 12 | 0.591 |
| 46 | 32,850 | 30 | 29 | 6 / 3 / 1 / 25 | 0.585 |

The official blind-to-training median is 0.587 using the same fingerprint and
standardization. Training counts include external and auxiliary records, so they
are not comparable to the official-only validation/test counts as split ratios.

The five repeats contain 274 evaluation appearances across 163 unique molecular
parents. Eighty parents occur in multiple evaluation repeats; all 163 appear in
training in at least one other repeat. Parent overlap between roles within each
repeat is zero, as is overlap with official blind parents. All protected historical
hashes verified successfully. Validation/test families number 23–29 per repeat.

Verification: 12 new tests cover label-independent selection, identity aliases,
connected families, ties and shortages, invalid inputs, immutable artifacts,
metric undefinedness, relative-path portability, and fit provenance. The original
workspace run passed 18 tests with two existing optional-dependency tests skipped.
PR validation repeats the focused tests against current main in an isolated worktree.

## Relationship to PR #28

[PR #28](https://github.com/Somerra/somerra-openadmet-cyp/pull/28) provides the v1
nearest-ten proxy and calibration comparison. This is the separately declared
close-analog v2 follow-up described there. Both can coexist; neither replaces the
saved Workbench assignments. The Python implementation imports no code from PR #28,
but preparation deliberately consumes its existing, hash-pinned local v1 manifest
to identify source snapshots and protect historical artifacts.

| Choice | PR #28 / v1 | This benchmark / v2 |
| --- | --- | --- |
| Repeats | One, seed 42 | Five, seeds 42–46 |
| Analog selection | Ten nearest, no default similarity floor | Up to ten with seed similarity >=0.45 |
| Family assignment | Closest nominating seed, round-robin roles | Connected nomination families, balanced whole-family roles |
| Roles | Train / validation / calibration / test | Train / validation / test |
| Evaluation | Raw, own-data affine, historical Workbench calibration | Raw only |
| Test rows | 162 | 24–31 per repeat |
| Median nearest-official-training similarity | 0.465 | 0.523–0.599; official blind: 0.587 |
| Sparse coverage policy | Mechanical minimum of 20 labels per endpoint/role | Explicit underpowered flags and undefined metrics; exploratory only |

The stricter chemical criterion improves structural resemblance at the cost of
substantially smaller assay cohorts. It does not establish better predictive
performance, potency-distribution matching, or calibration transfer. Keep v1 for
broader-chemistry/calibration comparisons and v2 for close-analog diagnostics.
