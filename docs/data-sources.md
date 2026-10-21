# Data sources for CYP regression

Our target is direct-inhibition pIC50 for CYP1A2, CYP2C9, CYP2D6 and CYP3A4.
Higher pIC50 means inhibition at a lower concentration. We select models using
macro-average ST-RAE on the official labels, with molecular-cluster holdouts.
An auxiliary source is useful only if it improves those held-out predictions;
its size or biological relevance alone is not evidence of a gain.

This guide describes the implemented adapters and the completed campaign snapshots.
“Potential value” below is a modeling hypothesis, not a demonstrated result.
Snapshot counts are before fold-specific auxiliary exclusions, not the number of
examples available in every training fold.

## Source map

| Source / signal | Role in regression | Location | Current evidence |
| --- | --- | --- | --- |
| Official direct-inhibition pIC50 | Primary supervision and evaluation | `data/prepared/official-direct/` | Required reference; strongest maintained models use these labels |
| Official TDI-condition pIC50 | Related potency targets | `data/prepared/official-auxiliary/` | Tested only within the auxiliary bundle; individual benefit unknown |
| Official Emax | Maximal-effect targets | Same auxiliary snapshot | Individual benefit unknown |
| Official single-concentration log2 fold change | Fixed-concentration response targets | Same auxiliary snapshot | Individual benefit unknown |

These are conventional locations for new snapshots. The completed campaign's
original snapshots remain under `runs/regression-{corpus,auxiliary}-v1/`.
One [shared catalog](../configs/README.md) registers these paths; each model recipe
selects named `data_sources`. CPU evaluation accepts direct labels only. Auxiliary sources currently
feed the neural multitask screen, not the CPU incumbent.

## 1. Official direct-inhibition labels: the task-defining source

Upstream: [OpenADMET dataset card at our pinned revision](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test/blob/146282d2ce1532f1803679a834a9bec9dac1e482/README.md).
Raw file: `data/raw/cyp-challenge-TRAIN_inhibition.csv`.
Adapter: [`data_sources/official.py`](../src/somerra_openadmet/data_sources/official.py), `prepare()`.

The prepared snapshot contains 4,905 rows, 4,901 standardized molecular parents,
and 4,565 molecular clusters. Targets are sparse across the four endpoints.
`training.csv` retains SMILES, observed pIC50, confidence bounds, parent identity,
cluster and fold assignment. `manifest.json` binds the snapshot to input hashes.

Value: this source measures the actual prediction target, so it anchors every
comparison. Its confidence bounds support the uncertainty-aware evaluation metric.
The official-label nested incumbent scores 0.743511 MA-ST-RAE. Frozen CheMeleon
features reach 0.739029 on the same labels, but do not pass promotion gates: that
is model/representation evidence, not evidence for adding assay data.

Limitations: label coverage differs by endpoint and nearby chemical analogues
can make random splits optimistic. Preparation standardizes parent identities,
excludes blinded-test parents, and persists cluster folds. Missing targets remain
missing; they are not converted into inactive labels. Training uncertainty and
evaluation uncertainty are distinct: the standard neural Huber loss does not use
confidence bounds, while the separate interval-loss experiment does.

## 2. Official TDI-condition potency: related, not interchangeable

Raw file: `data/raw/cyp-challenge-TRAIN_TDI.csv`, from the same pinned official
dataset. Adapter: `official.py`, `auxiliary()`.

The regression adapter selects columns containing `_pIC50_TDI_condition` as
separate continuous targets. It does **not** use `is_TDI` classification labels,
nor copy the paired direct-inhibition columns into primary supervision.

Potential value: learning potency under a related assay condition may help the
shared molecular encoder identify chemistry relevant to direct inhibition,
especially where direct labels are sparse. Keeping separate heads allows the
two conditions to have different responses.

Limitations: time-dependent and direct inhibition are different targets. Pooling
their values as if they measured the same outcome would introduce label mismatch.
Many molecules overlap other official files, so extra assay rows are not all new
chemical coverage. No matched TDI-only ablation has established its value for
direct-inhibition regression.

## 3. Official Emax: effect magnitude alongside potency

Raw file: `data/raw/cyp-challenge-TRAIN_Emax.csv`.
The auxiliary adapter selects `_EmaxVsPosCtrl_` columns for both direct and TDI
conditions, excluding confidence-bound and standard-deviation columns.

Potential value: maximal effect supplies a different response dimension from
the concentration needed for inhibition. A shared encoder might use this signal
to distinguish compounds with different response profiles even when their potency
labels look similar.

Limitations: Emax is not pIC50 and cannot be converted to it directly. It remains
a separate regression head, with training-fold normalization. Emax uncertainty is
not currently propagated into the auxiliary loss, and `is_TDI` fields in this
file are not consumed. Its incremental benefit has not been isolated.

## 4. Official single-concentration screening: additional response supervision

Raw file: `data/raw/cyp-challenge-single-concentration-TRAIN.csv`.
The adapter retains finite `log2fc_estimate` and `concentration_M`, creating a
separate target per enzyme and concentration. In the completed snapshot, each
enzyme has 4,372 prepared observations at approximately 49.5 micromolar.

Potential value: a response at a known concentration may help the encoder learn
which molecular features are associated with inhibition, without requiring a
complete concentration-response fit for every observation.

Limitations: one concentration does not identify an IC50. Response depends on
both potency and effect magnitude; saturation can conceal distinctions. The
adapter therefore preserves the response scale and concentration-specific head,
rather than inventing a potency. Its separate contribution is untested.

### How the official auxiliary bundle is prepared

The three signals above share one snapshot: 39,557 parent/task/source observations
across 6,141 parents in the completed campaign. Replicates are median-aggregated
within parent, task and source. Original SMILES remain available for features.
The manifest records task counts, input checksums and the `assays.csv` checksum.

The old 30-epoch bundled auxiliary screen scored 0.879526 versus 0.860705 for the
official-direct neural screen. Those results do not justify retaining the bundle
as an improvement or assigning credit/blame to one component. The later 10-epoch
control scores 0.770953, but uses a changed loss implementation; it is not a clean
10-versus-30-epoch comparison. See the [campaign evidence](regression-campaign.md).

## Deferred external datasets

ChEMBL and PubChem are out of scope for this PR. Their adapters, download commands,
catalog entries and external-specific training code have been removed. Follow
[ChEMBL issue #3](https://github.com/Somerra/somerra-openadmet-cyp/issues/3) and
[PubChem issue #4](https://github.com/Somerra/somerra-openadmet-cyp/issues/4) for
source rationale, prior results and single-source evaluation requirements.

The historical joint addition did not establish a benefit: MA-ST-RAE 0.772899
versus 0.770953 for the official bundle control; delta CI [-0.013851, 0.017249].
The recovered external run has incomplete provenance. Saved local snapshots and
run evidence remain intact; historical code is recoverable at commit fb848c4.
Current setup and training support only the official OpenADMET bundle (direct
and auxiliary measurements). Dataset availability is not evidence of usefulness.

## Shared safeguards and boundaries

- Auxiliary examples matching any outer-holdout parent or reaching ECFP4 Tanimoto
  similarity 0.6 to any holdout molecule are removed before fitting. Blinded-test
  parent exclusions are exact standardized-parent exclusions, not the same
  analogue filter. See [`common.py`](../src/somerra_openadmet/data_sources/common.py).
- Neural normalization is fitted only on training labels. Current losses average
  observed heads within source families and then families; batches themselves are
  not source-balanced. More rows therefore do not imply proportional loss weight.
- Raw downloads, prepared datasets and run artifacts stay out of Git. Use manifests
  to identify the actual snapshot; do not refresh it during a comparison.
- The 750 blinded test structures define prediction rows and exclusion identities,
  not training labels or a local validation set. The separate structure-prediction
  challenge file is outside this pipeline.
- CheMeleon weights and frozen embeddings are **model/representation inputs**, not
  another labeled CYP source. Their benefit must be evaluated with assay data held
  fixed. This pipeline does not ingest the encoder's pretraining corpus.
