# Challenge-like hit expansion: one frozen analog bank

PR #29 is the maintained hit-expansion benchmark. The revised protocol replaces
small, overlapping test samples with one frozen bank of close analogs, excluded
from **every model's gradient training**. Five family-grouped checkpoint-selection
rotations produce exactly one scoring prediction per bank row. Primary evaluation
pools those predictions before computing ST-RAE; it does not average small-fold
ST-RAE ratios. Workbench artifacts and earlier expansion bundles remain intact.

The user authorized a larger bank of potent training hits on 2026-11-01, prioritizing
close chemistry and coverage over the organizers' exact top-25 quota. This is
retrospective development evaluation, not a globally untouched prospective test.
No training, calibration-policy change, or submission is included.

## Challenge resemblance and explicit approximations

The organizers expanded 25 potent hits each from CYP1A2, CYP2C9, and CYP3A4 into
ten similar compounds per hit. CYP2D6 did not select hits. Training assays are
sparse; the 750 purchased analogs were measured across all four CYPs. Analog
series from one hit remain together in the public/private leaderboard division.

Primary references, retained privately for attribution:

- [Challenge construction](https://www.openadmet.org/blog/announcing-openadmets-cyp-inhibition-blind-challenge/)
- [Assay and tiering design](https://openadmet.github.io/octant-cyp-inhib-blog-post/)
- [Official dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)
- [Portal rules](https://huggingface.co/spaces/openadmet/cyp-challenge/blob/main/config.py)

We preserve the mechanism: known training hits, structure-selected close analogs,
and held-out assay prediction, including CYP2D6. Existing measured compounds
substitute for new catalog purchases. Exact organizer seed IDs and fingerprints
are unverified. Our direct-pIC50 threshold, broader hit bank, and unequal endpoint
seed counts are explicit approximations. Structural similarity does not establish
potency-distribution matching or better leaderboard prediction. Missing assays
cannot be reconstructed by repartitioning.

## Frozen selection and role contract

1. Standardize molecular parents without stereochemistry. Exclude official blind
   parents. Sort unique official parents, shuffle with NumPy seed 42, and reserve
   the first half as the training seed pool.
2. Use labels only from that seed pool. Select every unique parent with observed
   direct pIC50 **>=5.0** for CYP1A2, CYP2C9, or CYP3A4. Retain endpoint provenance;
   CYP2D6 never selects seeds. Rank by descending potency, breaking ties by ID.
3. Pass structures only to analog selection. From the other half of the official
   parents, retain up to ten nearest analogs per seed with Tanimoto **>=0.50** using
   Morgan radius 2, 2,048 bits, no chirality. Break ties by canonical parent order.
   Candidate values and label availability cannot select analogs or groups.
4. Connect all retained seed–analog nominations. Shared analogs merge seed families.
   Allocate whole families to five balanced groups: largest first, ties by smallest
   analog parent, then the lowest-index least-populated group. No label balancing.
5. Exclude the **entire bank** and official blind parents from every training source,
   including aliases and auxiliary/external records. Keep training seed hits and
   eligible unselected compounds. Distinct close training analogs remain eligible.
6. For rotation k, select its checkpoint using the other four analog groups and
   score only group k. Training CSVs are byte-identical across rotations. Each
   bank row scores once and is validation four times; it never trains a model.

Directory names `repeat-42` through `repeat-46` now identify scoring groups 0–4,
not five random splits. The seed pool always uses seed 42. Fresh fits are required;
training rows alone fit scalers and count-derived task weights. Checkpoint selection
must stay local to each rotation's validation set. Choosing one shared epoch using
all five validation sets would indirectly expose a row to its own scoring predictor.

The design was selected during a declared feasibility comparison using released
blind structural statistics and retrospective coverage diagnostics. Individual
candidate labels/masks never drive assignment. Parameters are frozen before
predictive evaluation; do not search splits or thresholds using model scores.

## Scoring and uncertainty

Concatenate the five disjoint scoring tables, verify exact IDs and family roles,
and compute shared challenge metrics once on the pooled cohort. Per-fold ST-RAE
is diagnostic: small groups may have zero denominators while the pooled metric is
well defined. Never silently drop endpoints from the macro. Require finite raw
predictions for every row, including unlabeled rows. No calibration is fitted.

Resample whole nomination families 1,000 times with seed 42 for percentile ST-RAE
intervals. Count undefined resamples; require at least 95% valid resamples for an
interval. These intervals are conditional on frozen predictions. They do not
capture training, checkpoint-selection, split-design, or adaptive search uncertainty.

The fixed bank is excluded from training but is **not globally sealed from
checkpoint selection**: a row is validation in other rotations. Its own scoring
predictor excludes that row and its family from both fitting and checkpoint
selection. Prior molecular exposure and repeated model selection still make this
development evidence. Individual fold metrics can remain underpowered.

## Commands and artifacts

From the repository root, with pinned local snapshots available:

```bash
python -m somerra_openadmet.hit_expansion prepare \
  --source-manifest /mnt/data/somerra-openadmet-runs/hit-expansion-evaluation-v1-20261101/manifest.json \
  --workbench-root /mnt/data/somerra-openadmet-runs/briford-20261028 \
  --workbench-hashes docs/results/workbench-protocol-artifacts.json \
  --output runs/hit-expansion-frozen-bank-v3-20261101

python -m somerra_openadmet.hit_expansion audit \
  --root runs/hit-expansion-frozen-bank-v3-20261101 \
  --output runs/hit-expansion-frozen-bank-v3-20261101/audit
```

Existing outputs are rejected. Preparation verifies sources and historical
Workbench hashes before and after writing, and stages the complete bundle before
publishing it. Relative input paths are resolved. Pinned source files remain
required at their recorded paths for verification.

Root-level `bank-assignments.csv`, `bank-seed-hits.csv`, and `bank-analog-links.csv`
freeze identities, selection provenance, all nominations, and scoring groups.
Each rotation exports `train.csv`, `validation.csv`, `test.csv`, `assignments.csv`,
`seed-hits.csv`, `analog-links.csv`, and `excluded-training-records.csv`.
Assignments contain no assay labels. Historical Workbench folds are provenance only.
All model families and task subsets reuse these files; never resplit specialists.

The manifest records hashes, versions, role semantics, coverage, ST-RAE denominator
readiness, exclusions per task, and limitations. Scientific code is snapshotted
privately. The audit includes similarity, family sizes, scaffold overlap, seed
potency, coverage, cross-rotation roles, and exact-once scoring. Similarity statistics
weight molecule rows; parent counts are separate.

For evaluation, supply `repeat-42.csv` through `repeat-46.csv`, each containing
exactly its test IDs and four finite raw endpoint predictions. Supply corresponding
JSON fit records:

```json
{
  "train_sha256": "<manifest hash for repeat-42/train.csv>",
  "validation_sha256": "<manifest hash for repeat-42/validation.csv>",
  "fresh_fit": true,
  "test_used_for_fitting_or_selection": false,
  "analog_bank_used_for_gradient_training": false,
  "checkpoint_selection_scope": "this_rotation_validation_only",
  "calibration": "none",
  "scalers_and_task_weights_fit_on": "train",
  "checkpoint_sha256": {"checkpoints/model.pt": "<actual checkpoint SHA-256>"}
}
```

Checkpoint paths are absolute or relative to the fit-record directory. List every
checkpoint. Hashes verify bytes and split identity; isolation fields remain trainer
attestations requiring review. Generic pretrained encoders are eligible with
provenance; historical challenge-fitted checkpoints exposed to bank rows are not.

```bash
python -m somerra_openadmet.hit_expansion evaluate \
  --root runs/hit-expansion-frozen-bank-v3-20261101 \
  --predictions /path/to/raw-predictions \
  --fit-records /path/to/fit-records \
  --output /path/to/new-evaluation
```

Outputs are `evaluation.json` and `pooled-raw-predictions.csv`. The primary result
is `primary_pooled_score`; `macro_across_all_five` is diagnostic only. The scoring
implementation must match the bundle snapshot.

## Consolidation and compatibility

This revision supersedes [PR #28](https://github.com/Somerra/somerra-openadmet-cyp/pull/28)
as the maintained hit-expansion benchmark. Its historical artifacts, attribution,
and calibration evidence remain preserved. The Workbench default is unchanged.

| Design | #28 v1 | Initial #29 v2 | Revised #29 |
| --- | --- | --- | --- |
| Seeds per selecting CYP | Top 25 | Top 25 | All seed-pool hits with pIC50 >=5 |
| Minimum seed similarity | None | 0.45 | 0.50 |
| Organization | One split, calibration role | Five overlapping splits | One bank, five family rotations |
| Test coverage | 162 rows | 24–31 rows per repeat | Exactly one score per bank row |
| Bank rows ever gradient-trained | Depends on other experiments | All 163 across other repeats | Zero in these rotations |
| Primary score | Single test | Descriptive repeat scores | Pooled unique predictions |
| Calibration | Raw, own-data, historical | Raw | Raw |

Legacy v2 audit/evaluation semantics remain supported. New preparation writes
`hit-expansion-frozen-bank-v3` to a new directory. Old manifests, assignments,
and code snapshots are not rewritten. All code, reports, and attribution stay
private. No submission path exists here; prediction-only settings and portal
cooldown/receipt policies remain in force. No GPU campaign is authorized.

## Prepared artifact: 2026-11-01

The [machine-readable evidence](results/hit-expansion-frozen-bank-20261101.json)
records the input hashes, parameters, coverage, and structural comparisons.
The run manifest SHA-256 is `9197db129654f9fc74af50a8280c2102bb20ff6745b3925c5915e4d009165eb1`.

| Quantity | Revised benchmark | Official challenge |
| --- | ---: | ---: |
| Evaluation compounds | 223 unique parents | 750 rows / 748 parents |
| Nomination families | 108 | Actual organizer identities unverified |
| Nearest-training median Tanimoto | 0.5893 | 0.5873 |
| Nearest-training 10th percentile | 0.5141 | 0.5156 |
| Nearest-training 90th percentile | 0.7330 | 0.7018 |
| CYP1A2 / 2C9 / 2D6 / 3A4 observed labels | 42 / 38 / 42 / 156 | Dense four-CYP assay panel |

Each rotation has 32,686 training records including auxiliary/external data,
178–179 validation compounds, and 44–45 scoring compounds. Validation label
counts range from 31–37 / 25–36 / 30–38 / 120–126 for the four CYPs. All validation
and pooled ST-RAE denominators are positive. This establishes metric readiness,
not a guarantee of adequate statistical power. Individual scoring folds remain
small; the pooled score is primary.

The training seed bank contains 420 / 186 / 247 qualifying hits for CYP1A2,
CYP2C9, and CYP3A4, respectively, with overlapping selections retained. These
unequal, larger hit counts differ from the organizers’ 25-per-CYP selection.
The closer structural distribution does not remove sparse assay coverage or
this selection difference.

Verification: 30 focused tests passed; nine existing Torch-dependent recipe
tests skipped because Torch is unavailable. Ruff lint/format checks passed.
The earlier v2 bundle and protected Workbench/v1 hashes also verified unchanged.
No models were trained and no real model scores or model confidence intervals
have been generated.
