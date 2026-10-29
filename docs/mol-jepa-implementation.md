# Mol-JEPA implementation and campaign

Issue #14 uses the original reference and inner-selected H100 protocol. The
2026-10-28 CheMeleon quickpass is contextual evidence, not the matched baseline.

The user confirmed noncommercial research use on 2026-10-29. Model source,
checksums, the CC BY-NC model card, and installed dependency versions are retained
privately under `runs/mol-jepa-model`. The model revision is
`4c912b450175f31b5ba913a5dc921c03b27b985a`; downstream source is pinned to
`27f0065b69c73d18eec7a32ecc413b6dc3f33d90` under `runs/mol-jepa-source`.

Use `runs/mol-jepa-env/bin/python`, an isolated pip overlay on the existing Conda
neural environment. The Conda-native cuik_molmaker package is intentional:
installing the pip cuik-molmaker-pin wheel produces an incompatible RDKit shared
library dependency. Pip's missing-distribution warning does not describe whether
this native replacement imports. Chemprop import and representation tests pass
with the Conda package.

The existing cache/screen/predict commands accept Mol-JEPA configurations through
`encoder: mol_jepa`. Adaptation values are `frozen`, `tokens`, `full`, and `lora`.
Mol-JEPA uses cache schema 2 with three independently checksummed views and a root
completion marker. Existing CheMeleon caches/checkpoints retain schema 1.

Training calls the authors' differentiable encode/predict path. The inference
wrapper's `no_grad` path is not used for training. Full tuning leaves all encoder
parameters enabled, but permits absent gradients only for non-graph expert
encoders, whose input modalities are absent from SMILES-only data. LoRA freezes
the base model and its BatchNorm state. Its attention wrapper explicitly calls
the PEFT output projection so adapters receive gradients instead of being
bypassed by PyTorch's raw-weight access. Both train and eval avoid that bypass.

The four reference recipes follow the issue, including 60 adaptation epochs.
The H100 arms reuse bounded inner-only recipe selection. Extraction profiles batches 64–1024 independently, chooses the smallest within
5% of peak throughput, and stores float32 representations.

`python scripts/run_mol_jepa_campaign.py` runs the prerequisite CheMeleon matrix,
then Mol-JEPA extraction and eight five-fold runs, followed by private paired
comparisons and CPU blind exports. Stage allocations total 22 GPU-hours for
Mol-JEPA; the existing shared ledger and per-run projection checks remain active.
The driver stops on failure, preserves partial artifacts, and never submits.
Do not restart under new run names to evade historical charges.

The initial campaign is interrupted, not running. Its last artifact was written
at 2026-10-29 02:26 UTC during the H100 token arm. The full/LoRA arms and
final comparisons/exports are incomplete. The stale allocation needs reconciliation
before recovery. Completed scores are screening evidence only.

This PR is stacked on #16 and preserves its newer full-process campaign supervisor.
Set `SOMERRA_CAMPAIGN_LEDGER` to the existing initialized shared ledger; no new
ledger is created and no historical charge is reset. The 22-hour stage-cap sum
is not an independent allocation or permission to enlarge that shared budget.
The historical campaign used the earlier ledger implementation; its receipts
remain local.

Campaign status is in `runs/regression-mol-jepa-campaign-v2/status.json`.
The first attempt stopped before GPU reservation because of the native dependency
conflict; its failure evidence is retained in the v1 status directory. A running
campaign is not a completed issue. Reports, five-fold coverage, budget
reconciliation, and replay evidence must all be reviewed before closing #14.

Validation: 30 targeted tests passed, including all four checkpoint replay paths,
real-model gradients, LoRA frozen-state stability, token dimensions, cache
alignment/immutability, and existing CheMeleon representation tests. CPU online
outputs matched the official inference wrapper exactly on a two-molecule smoke
batch for all three representation views.

## Recovery authorized 2026-10-29

The user approved increasing the existing shared ceiling from 12 to 20 GPU-hours.
An audited amendment preserves existing charges; the previously unreconciled
campaign contributes 4.4886 hours, including the interrupted token arm's full
four-hour conservative charge. No allocation is reset. Timing pilots and future
workers use the same full-process supervisor and ledger.

Full/LoRA graph inputs are now precomputed once using the exact pinned upstream
SMILES featurizer and collator. These are deterministic input graphs, not learned
representations, so they are safe across folds. Tests verify online/cached outputs
and encoder gradients agree. Each fold still starts with the original checkpoint.
Measured reference batch-64 epochs fell from 13.3 to 2.3 seconds for full tuning
and from 12.2 to 1.15 seconds for LoRA. Epoch counts, optimizer recipes, group
splits, and inner-only selection are unchanged.

Recovery evidence and status live under
`runs/regression-mol-jepa-recovery-20261029/`. Seven completed models now have
validated CPU blind exports. The remaining five runs retain incomplete status
until their full five-fold training and replay checks pass. No upload occurs.
