# Mol-JEPA comparison implementation

The original-protocol screening matrix is complete: four CheMeleon controls and
all eight Mol-JEPA reference/H100 variants have five-fold OOF predictions,
checkpoint replay, and locally validated 750-row blind exports. Nothing was
promoted or submitted. See [results and limitations](mol-jepa-results-20261029.md).

## Model and environment

The user confirmed noncommercial research use on 2026-10-29. The immutable model
revision is `4c912b450175f31b5ba913a5dc921c03b27b985a`; the authors' downstream
source revision is `27f0065b69c73d18eec7a32ecc413b6dc3f33d90`. The CC BY-NC model
card, source checkout, weight/code checksums, exact pip versions, and Conda package
records remain private under `runs/mol-jepa-model` and `runs/mol-jepa-source`.

Use the isolated `runs/mol-jepa-env/bin/python` overlay on the neural Conda
environment. The Conda-native cuik_molmaker package is intentional: the pip
cuik-molmaker-pin wheel conflicts with these RDKit libraries. Do not replace it
solely to eliminate pip's missing-distribution warning. Real Chemprop/Mol-JEPA
imports, GPU training, and CPU replay were exercised successfully.

## Interfaces and invariants

Existing cache/screen/predict commands accept `encoder: mol_jepa` and adaptation
`frozen`, `tokens`, `full`, or `lora`. Invalid encoder/head combinations fail before
supervision or reservation. The model's three float32 views use cache schema 2;
individual views use the shared checksum, shape, finite-value, and identity-alignment
reader. Existing CheMeleon schema-1 caches/checkpoints remain supported.

Training uses the differentiable encode/predict path. Deterministic input graphs
are precomputed with the pinned original featurizer/collator, not a learned encoder.
Online/cached output and gradient equivalence is tested. Full tuning permits missing
gradients only for explicitly absent non-graph experts; actual full-arm optimizer
steps are covered in tests and all ten full-tuning folds completed.

LoRA freezes base parameters and encoder BatchNorm state. Its attention path calls
the adapted output projection explicitly. Pre-norm/batch-first assumptions are
checked; zero-initialized LoRA reproduces the pinned model within declared tolerance.
BF16 non-finite outputs trigger float32 fallback during probing; malformed shapes
and float32 numerical failures still fail. Fold checkpoints validate the foundation
checksum and store an absolute model directory. `restore_checkpoint(..., model_dir=...)`
allows explicit relocation while enforcing the same pin. Historical relative paths
remain readable from their original working directory or with that override.

Each fold starts from the original pretrained state. Full weight loading and
checksum verification are intentionally retained per fresh model; caching a mutable
pretrained model/state across folds is deferred. Preflight reads verified provenance
without constructing another full model. Graph caching removes the dominant repeated
featurization cost (reference epoch times: full 13.3→2.3 s; LoRA 12.2→1.15 s).

## Resumption and accounting

`python scripts/run_mol_jepa_campaign.py --resume --output PATH` verifies completed
stage configs and artifact checksums before reusing them. `--plan PLAN.json` accepts
a list of `{operation, name, config}` stage entries, allowing recovered run paths
without editing source. Partial output is preserved and rejected; supply a separately
accounted recovery config for an interrupted stage. Status text alone never skips
training. Comparisons include matched frozen/full families and explicitly separate
tokens-vs-frozen and LoRA-vs-full pairs. Export is CPU-only and never uploads.

PR #16's full-process supervisor remains in force. The user explicitly approved a
20-hour shared ceiling on 2026-10-29, recorded as an amendment preserving all earlier
charges. Reconciliation retained the interrupted token run's conservative four-hour
charge. No fresh ledger or spending reset is allowed.

The original recovery stopped on a full repository volume while saving fold 0.
All 152 recovery files were checksum-verified and relocated to
`/mnt/data/somerra-openadmet-runs/mol-jepa-recovery-20261029`; the old repository path
is a symlink. The successful recovery uses a frozen source snapshot and a 20-GiB
storage preflight. Earlier failures and charges are retained. Final evidence is under
`runs/regression-mol-jepa-recovery-20261029`: `recovery-status.json`, `final-report.json`,
`adaptation-comparisons.json`, `storage-relocation.json`, and per-run artifacts.

Review fixes after the completed campaign do not replace its source snapshot or
retroactively change its recipes. CPU regressions cover those fixes; published
screening numbers refer to the preserved executed snapshot.
