# Hit-expansion evaluation and next calibrated candidate

User direction, 2026-11-01: prepare the next regression candidate using the historical
Workbench constants, lifting the prior project hold, and build a separate
expansion-style evaluation. No new upload or training budget is authorized here.
Prediction-only submission policy continues. User authorization is not an assertion
of organizer-specific clearance. Keep borrowed calibration provenance explicit.

## Calibration for the next candidate

Preserve the raw autoresearch submission (SHA-256
`6d286b67a1a42360b571c5757f8b5fb491e1fbde0d8c692b94ea6ed1beb75fdd`).
Apply the historical `briford.calibrated` formula separately to each CYP after
averaging its five fold predictions:

`max(1, target_mean + (p - mean(p)) * target_sd / sample_sd(p))`

Use the new model's OOF Pearson correlation, multiplied by the historical factor
(1A2 1.32, 2C9 1.23, 2D6 1.66, 3A4 1.07), and historical population moments.
For CYP2D6 retain the explicit target mean 3.57 and SD 0.90 override.
Do not silently clamp a correlation extrapolation outside (0, 1].

Prepared artifact:
`/mnt/data/somerra-openadmet-runs/cyp-first-autoresearch-20261031/historical-workbench-calibration/`

Its `activity_submission.csv` contains 750 validated rows with no warnings and
SHA-256 `ea06969322095ab37857ee90faacae10db52cea4929f60a16a58066f32ac06c2`.
The run manifest records every parameter, source hash and provenance limitation.
It is not submitted. The earlier raw candidate remains intact and scored 0.7151.
A raw-versus-calibrated submission of this same model would isolate the transform
more directly than comparisons between different historical models; the leaderboard
still supplies adaptive development feedback, not a new independent benchmark.

## Separate evaluation contract

The saved Workbench folds remain unchanged. This new proxy follows the direction
of the challenge's hit expansion, but cannot reproduce catalog purchases or dense
four-CYP assays from the available sparse training observations.

1. Verify frozen official, auxiliary-pool, blind and Workbench CSV hashes.
2. Standardize molecular parents without stereochemistry; aliases cannot straddle
   roles. Exclude official blind parents everywhere, including auxiliary training.
3. Randomly reserve half the available official molecular parents as a training
   seed pool (seed 42). Only labels in this pool select seed hits. Select the top
   25 observed hits each for CYP1A2, CYP2C9 and CYP3A4. CYP2D6 does not select seeds.
4. Search the remaining official parents by 2048-bit Morgan radius-2 Tanimoto,
   without chirality. Nominate ten nearest analogs per unique seed, with no default
   similarity threshold. Never inspect candidate assay values or missingness to
   choose analogs. Seeds stay in training. Other unselected parents may train too.
5. For duplicate nominations, assign each analog to its closest nominating seed.
   Shuffle these anchor groups (seed 43) and rotate them across validation,
   calibration and test roles. These groups are disjoint but are not asserted to
   be chemically dissimilar to one another. All records of a parent share a role.
6. Remove every held-out parent from **all** training sources. Retain broad auxiliary
   supervision for remaining training parents. Each evaluation file retains sparse
   labels and credible intervals; missing assays are not negative labels.
7. Fresh fits are required. Validation monitors training and selects checkpoints.
   Freeze checkpoints before using calibration labels. Fit each own-data transform
   on the calibration set only; use its correlation for the historical Workbench
   variant. Score all frozen transforms on the same test rows, using the shared
   evaluator and reporting all five metrics per CYP and macro ST-RAE.

The command requires a fit record with train/validation hashes, a checkpoint hash
per CYP, and an explicit assertion that calibration/test labels did not train models
or select checkpoints. This is an audit assertion, not proof of an external trainer's
behavior. Fit provenance must be reviewed before scientific use. Predictions must
cover the exact partition, including unlabeled rows; the scorer rejects missing,
duplicate and nonfinite predictions. Existing historical checkpoints are ineligible.

`evaluate` compares raw, fixed own-data affine (OLS slope/intercept with floor 1.0),
and historical Workbench transformations. Historical rescaling uses the scoring
cohort's **unlabeled prediction moments**, not scoring labels. This transductive
step must be stated. It does not make its borrowed constants training-only.
Invalid nonpositive slopes or historical correlation extrapolations fail explicitly.

## Feasibility audit: do not mistake nearest for close

The strict exploratory threshold of Tanimoto >=0.4 produced only 79 analog parents,
with 5–20 observed labels per endpoint/role. The coverage guard rejects scientific
scoring on that bundle. Preserve this negative feasibility result.

The nearest-ten version produces 68 unique seed hits and 522 held-out parents
(523 records due to an alias), distributed over 68 anchor groups:

| Role | Records | CYP1A2 labels | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---:|---:|---:|---:|---:|
| Training, including auxiliaries | 32,386 | 1,229 | 1,126 | 1,323 | 2,104 |
| Validation, 23 anchor groups | 195 | 58 | 60 | 49 | 93 |
| Calibration, 23 groups | 166 | 62 | 50 | 56 | 71 |
| Test, 22 groups | 162 | 63 | 49 | 65 | 67 |

For distinct held-out parents, median anchor similarity is **0.307** (10th–90th
percentiles 0.257–0.437); only **15.1%** reach 0.4. This is a nearest-neighbor proxy,
not a faithful close-analog catalog expansion. A mechanical minimum of 20 observed
labels per endpoint/role permits a pilot, not a claim of statistical adequacy.
Do not optimize the similarity threshold or split seed on prediction scores.
Report coverage and sparse-label selection bias alongside every result.

No independent estimate is claimed: prior campaigns have already used these
molecules. Fresh fits remove direct current-run training leakage, but repeated
strategy selection still makes the resulting scores adaptive development evidence.
The next bounded pilot should compare the same frozen model recipes across this
proxy and saved Workbench folds, with raw/historical/own-data transforms reported
separately. It requires a declared compute allocation and GPU readiness check.

## Similarity to the official split: measured 2026-11-01

The new split moves toward the official expansion but does not closely match it.
Using identical standardized, stereo-insensitive Morgan radius-2, 2048-bit
fingerprints, compare each evaluation row with its nearest official training
parent (auxiliary sources excluded):

| Evaluation set | Rows | Reference parents | Median Tanimoto | 10th–90th percentile | Fraction >=0.60 |
|---|---:|---:|---:|---:|---:|
| Official blind | 750 | 4,901 | 0.5873 | 0.5156–0.7018 | 44.0% |
| Expansion proxy test | 162 | 4,379 | 0.4652 | 0.3522–0.5800 | 6.8% |
| Workbench fold 2 vs training folds 1,3,4 | 981 | 2,941 | 0.4194 | 0.3067–0.5536 | 4.6% |

Query statistics weight molecule rows; references deduplicate parents. Official
blind has 748 unique parents; proxy test has 162. Neither overlaps its training
parents. Different reference-set sizes also contribute to these differences.
Validation and calibration proxy medians are 0.4576 and 0.4400, respectively.

The mismatch persists against endpoint-labeled training parents. Proxy queries
here include only rows with that endpoint observed, whereas all 750 official
blind molecules were assayed for each endpoint:

| CYP | Official median | Proxy median | Proxy observed test labels |
|---|---:|---:|---:|
| CYP1A2 | 0.5172 | 0.4156 | 63 |
| CYP2C9 | 0.5179 | 0.4304 | 49 |
| CYP2D6 | 0.4711 | 0.3857 | 65 |
| CYP3A4 | 0.5377 | 0.4531 | 67 |

Nearest-seed median similarity is 0.3099 for proxy test, versus 0.5212 for official
blind against reconstructed top-25 hits for CYP1A2, CYP2C9 and CYP3A4 (71 distinct
parents). The actual organizer seed identities are not verified: this second
comparison is approximate. Nearest seed and nearest training molecule are distinct
statistics; the earlier 0.307 figure described assigned anchors across all holdouts.

The likely mechanism is limited analog availability: the official workflow searched
a purchasable catalog, while this proxy must reuse a small measured library. Ten
nearest neighbors need not be close analogs. Keep v1 as a more distant-chemistry
stress test, not the sole basis for predicting official performance. A future v2
could match structural similarity distributions using structures only, with a
separately declared protocol and sparse-label feasibility check. This PR does not
implement v2 or change any saved folds. Matching structure would not establish
matching potency distributions or calibration transfer.

The [summary CSV](results/hit-expansion-similarity-20261101.csv) and
[manifest](results/hit-expansion-similarity-20261101.json) retain the measurements
and input hashes. Reproduce the audit without model training:

```bash
PYTHONPATH=src python scripts/audit_expansion_similarity.py \
  --split /mnt/data/somerra-openadmet-runs/hit-expansion-evaluation-v1-20261101 \
  --source /mnt/openclaw-home/somerra-openadmet-cyp/runs/regression-hillclimb-20261026-data \
  --workbench-csv /mnt/data/somerra-openadmet-runs/briford-20261028/data/mt_aux8.csv \
  --output /mnt/data/somerra-openadmet-runs/hit-expansion-similarity-replay
```

## Commands

```bash
python -m somerra_openadmet.expansion_evaluation prepare \
  --source /mnt/openclaw-home/somerra-openadmet-cyp/runs/regression-hillclimb-20261026-data \
  --workbench /mnt/data/somerra-openadmet-runs/briford-20261028/data \
  --output /mnt/data/somerra-openadmet-runs/hit-expansion-evaluation-v1-20261101

python -m somerra_openadmet.expansion_evaluation evaluate \
  --root /mnt/data/somerra-openadmet-runs/hit-expansion-evaluation-v1-20261101 \
  --calibration-predictions CALIBRATION_PREDICTIONS.csv \
  --test-predictions TEST_PREDICTIONS.csv \
  --fit-record FIT_RECORD.json --output NEW_RESULT_DIRECTORY
```

All prediction files use `Molecule_Name` and the four canonical pIC50 columns.
Preparation saves assignments, seed-hit nominations, analog similarities, four
partition files, implementation snapshot and hashes. Outputs never replace an
existing split. No command trains a model or uploads a submission.
