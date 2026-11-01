# Workbench TDI classification

On 2026-10-30 the user requested a classification submission using the regression
Workbench architecture with an appropriate output head. The result predicts the
two official binary TDI endpoints, CYP2D6 and CYP3A4, from official SMILES alone.

## Recipe and data

The randomly initialized Chemprop 2.3.1 encoder retains width 700, depth 5,
dropout 0.1, NormAggregation(100), BatchNorm and two hidden FFN layers of width 300.
Training retains batch 64, Adam with Chemprop's Noam-like schedule,
learning rates 1e-4 / 1e-3 / 1e-4, two warmup epochs, 16-bit mixed precision,
maximum 400 epochs and validation-loss early stopping with patience 40.

The two-output binary head uses masked BCE on logits and sigmoid probabilities at
inference. Binary targets are not standardized. Endpoint coefficients are inverse
observed-label counts computed only on the training partition, normalized to mean
one. Class weights and oversampling are not added. This first candidate has no
auxiliary regression heads or external training labels.

There are 4,822 labeled rows and 4,818 standardized molecular parents. No parent
overlaps the 750 blinded molecules, and no conflicting parent labels were found.
The new TDI bundle uses Workbench's first-seen scaffold IDs, seed-42 group shuffle
and greedy row-count balancing into five folds. Standardized parents determine
groups; original official SMILES remain model inputs. Every fold contains both
classes for each endpoint. These are new classification assignments, not the
regression bundles' saved fold numbers.

For each fold, four folds train the model and the fifth selects the best weighted
validation BCE checkpoint. That same fold supplies development OOF probabilities.
Five blind probability arrays are averaged equally. One exact MCC-maximizing
threshold per endpoint is fitted to pooled OOF predictions; ties prefer proximity
to 0.5. These scores include checkpoint and threshold selection and are not
independent estimates. They do not complete the earlier nested promotion plan.

## Completed candidate

| Endpoint | OOF MCC at 0.5 | Selected threshold | OOF-tuned MCC | Blind positives |
| --- | ---: | ---: | ---: | ---: |
| CYP2D6 | 0.080127 | 0.21675058 | 0.111477 | 349 / 750 |
| CYP3A4 | 0.151170 | 0.18170112 | 0.309184 | 404 / 750 |
| Macro | 0.115649 | — | 0.210330 | — |

The historical LightGBM OOF-tuned macro MCC was 0.234554 on different folds;
this run does not establish an improvement. Selected epochs were 7, 14, 9, 10 and
12, followed by the prescribed 40-epoch patience. Serial GPU workers consumed
167.51 wall seconds under a separate two-hour cap. Historical regression ledgers
were unchanged.

All five checkpoint reloads reproduced their full blind arrays exactly. The initial
11 targeted tests passed, as did both repository and official tutorial TDI validators
for all 750 rows. The official validator revision is
`832ae6d6447b07c2c71059de34d07f1130388247`.

The authorized prediction-only upload was accepted on **2026-10-30 at 15:17:03 UTC**.
Open Code was false, the report link was empty, and publication participation was
false. The submitted CSV SHA-256 is
`15c3cc9d1a7b3e486928f5faf511a32ca9e83cb3307219a06a70d1de0da8f2ca`.
The subsequent leaderboard observation reported **macro MCC 0.2506, rank 102**,
Open Code "No", and report link "Not submitted". Rank is a dated observation,
not a fixed result. No code, checkpoints or attribution links were uploaded.

PR verification passed the full CPU suite (205 passed, 35 optional-dependency skips),
seven neural/validation tests, lint and formatting. Preparing the data with the
isolated PR code reproduced the original training CSV hash, including every fold.
The frozen package snapshot imported and verified successfully. Reloading all five
models with the PR code matched saved blind probabilities within 3e-8 (CSV rounding),
and the endpoint thresholds and development scores matched exactly.

## Reproduction and evidence

Use the neural Conda environment and run from the repository root. Set a fresh
`output_dir` in the config before preparing another run; existing artifacts are
never overwritten by preparation.

```bash
PYTHONPATH=src python -m somerra_openadmet.workbench_classification prepare \
  --config configs/workbench-classification.json
PYTHONPATH=src python -m somerra_openadmet.workbench_classification run \
  --output runs/workbench-classification-20261030-v1
```

The runner has no upload operation. It freezes data/fold/config hashes and package
source, records worker-time charges, and refuses incomplete or changed artifacts.
Completed folds can be reused; a failed fold directory or unsettled reservation
requires reconciliation before a retry. A failed run does not authorize a new budget.

Private local artifacts are under `runs/workbench-classification-20261030-v1`:
data, source snapshot, checkpoints, curves, ledger, raw probabilities, thresholds,
validated CSV, `predictions/submission_receipt.json`, and portal observations.
The run's original immutable snapshot remains the exact training source. This PR
isolates the existing MCC helpers and snapshots package subdirectories so the runner
is self-contained on current main; it does not retrain or alter the submitted models.

The portable [result record](results/workbench-classification-20261030.json)
contains aggregate metrics, checksums, checks, and acceptance metadata only. Blinded
predictions and checkpoints remain gitignored. Workbench attribution is internal:
the architecture and grouping recipe follow revision
`8c72c7559471febaf0962fd551b4eaab8db3a136`; see `licenses/Workbench-MIT.txt` and the
historical regression reproduction documentation.
