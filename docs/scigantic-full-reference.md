# Full Scigantic benchmark and exact published folds

Use the complete published R25 system as the external reference. Our depth-four
baseline and CheMeleon-initialized 26-task model are individual adaptations, not
R25 reproductions. The preferred deployed candidate remains one full-data model.

## Components

| Recipe | Initialization / change | Tasks | Published pooled OOF MA-ST-RAE on all 4,905 official rows |
|---|---|---:|---:|
| armA | Random, width 700, depth 5 | 22 | 0.7381 |
| armB | Random, width 2048, depth 4 | 22 | 0.7189 |
| armC | B architecture, general CheMeleon encoder | 22 | 0.7332 |
| armD | B architecture, organizer CYP-finetuned encoder | 22 | 0.7515 |
| ivB2048 | B, interval loss, uniform record weights | 22 | 0.6891 |
| ivC2048 | C, interval loss, uniform record weights | 22 | 0.6848 |
| mu2048 | B, MAE, uniform record weights | 22 | 0.7173 |
| ii2048 | B, interval loss, inverse-variance weights | 22 | 0.7424 |
| mt_aux8 | Smaller architecture, organizer-only tasks | 8 | 0.6962 |
| union18 | Smaller architecture, external auxiliary tasks | 18 | 0.6991 |
| aux_ac50 | Smaller architecture, qHTS AC50 tasks | 22 | 0.7223 |
| cp_interval | Smaller architecture, interval loss | 22 | 0.6623 |
| g26_tdi_armB | B plus four TDI-condition tasks | 26 | Not available: no published OOF file |

These are scores of published third-party predictions, not retrained models. All
12 available OOF files cover the same official rows and were scored by our shared
challenge scorer against our verified official labels and bounds. Lower is better.
They are diagnostics, not a substitute for the complete R25 official score of 0.4436.
Pretrained checkpoint label exposure is not independently audited; OOF refers to
the fine-tuning stage. Different member training protocols also limit causal comparisons.

## Exact-fold adoption

The 22-task and 26-task CheMeleon assignments agree on every shared record.
The published 26-task pool has 32,909 records, the same IDs as our adapted pool;
all four official scored-label columns match within absolute tolerance 1e-7,
including missingness. Its standardized parents do not cross folds and none match
our official blind parents. This does not prove every auxiliary value or featurization
choice is identical between implementations.

| Held-out fold | Official held-out records | Total held-out pool records |
|---|---:|---:|
| 0 | 1,002 | 6,566 |
| 1 | 946 | 6,580 |
| 2 | 960 | 6,594 |
| 3 | 1,005 | 6,590 |
| 4 | 992 | 6,579 |

For each CheMeleon run, test fold k, validate on (k+1) modulo five, and train on the
other three folds. Their runner uses seed 42+k (one seed per fold), a maximum of
200 epochs, and patience 30 on validation loss. Our earlier 100-epoch runs with
seed 42 and checkpoint selection by scored-endpoint MA-ST-RAE are not a literal
training reproduction, even after adopting the same assignments.

There is no one fold assignment shared by the entire R25 system. The Workbench
members have different splits and use four folds for training and one for
validation/OOF. The audit's per-fold rows use the canonical CheMeleon assignment as
common evaluation strata; that does NOT assert every component trained on the same
remaining folds. Fold-number disagreements alone are not a full partition-distance
measure, since fold labels can be permuted. Some published files omit fold IDs.

## Full predictor reconstruction

The pinned `scripts/build_r25.py` was executed on the frozen member predictions
and incumbent submissions. The rebuilt 750-row CSV agrees with the published R25
submission across all 3,000 predictions to maximum absolute error 9.81e-6, within
upstream's 1e-4 tolerance. It is not byte-identical due to numerical optimization
variation. This verifies the combiner, not retraining of its constituent networks.

R25 includes an incumbent prediction baseline, per-endpoint linear corrections
fitted on OOF predictions, CYP3A4 auxiliary-head corrections, moment restoration,
and a 20% TDI-model contribution to CYP1A2. Its blend-fitting scores are optimistic
because the same OOF labels fit the coefficients. There is no independently scored
five-fold final-R25 prediction table in this audit. The TDI member also lacks public
OOF predictions. Do not manufacture a final-R25 fold score from its blind predictions.

A complete training reproduction involves roughly 65 component fits plus tracing
and reproducing the earlier incumbent lineage; the estimate is not a claim that
65 fits alone fully recreate R25. Before launching, distinguish that work from
using the frozen published system as an external benchmark. No new comparison
training was launched during this audit, and no third-party predictions were submitted.

## Reproducible audit

```bash
PYTHONPATH=src python -m somerra_openadmet.reference_benchmark \
  --source PINNED_SCIGANTIC_SOURCE --bundle VERIFIED_BUNDLE --output NEW_AUDIT
```

The audit checks the pinned source revision and official bundle hashes, normalizes
member prediction column names, rejects duplicate identities, and scores the common
finite rows. It writes `component-scores.csv` (pooled plus each of five strata),
`official-folds.csv`, `pool-folds.csv`, and `audit.json` with input hashes and limitations.

Private outputs from 2026-10-28:
- `/mnt/data/somerra-openadmet-runs/scigantic-full-reference-20261028`
- `/mnt/data/somerra-openadmet-runs/scigantic-r25-rebuild-20261028`

Keep all attribution and reconstruction evidence internal. The standing hold on
third-party prediction resubmission remains in effect.
