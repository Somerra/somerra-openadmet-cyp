# Matched frozen-representation pilot

The primary experiment compares the predictive utility of frozen CheMeleon mean
features (2048), Mol-JEPA CLS features (512), and Atom-JEPA scalar plus higher-order
invariants (768). Atom-JEPA scalar-only features (256) are a separate ablation.
All features use one fixed molecular view; Atom-JEPA uses one fixed conformer.
Fine-tuning and conformer augmentation are deferred until the results are reviewed.

After reviewing this pilot, the user authorized a
[frozen conformer sensitivity follow-up](atom-jepa-conformer-sensitivity.md),
including standardized scalar features, within the same cumulative four-hour
GPU cap. The original pilot below remains unchanged; fine-tuning is still deferred.

The implementation uses the existing prepared corpus and outer fold 0, with seeds
42, 43, and 44. All fits share `FrozenMLP` and its training loop: input -> 512 ->
BatchNorm -> ReLU -> four outputs; Adam 1e-3, weight decay 1e-4, batch 256, 300
epochs, float32, endpoint-balanced masked MAE in raw pIC50 units, and final-epoch
selection. Batch-order hashes and update counts must match across encoders for
each seed. Different input dimensions imply different first-layer parameter counts.

Nine primary fits use raw features. Nine sensitivity fits use StandardScaler fit
only on the outer-training molecules. Three additional raw scalar Atom-JEPA fits
complete the matrix. All 21 heads are fitted fresh through the same implementation;
only the verified pretrained CheMeleon/Mol-JEPA feature caches are reused. This
avoids making head initialization depend on encoder construction or package imports.

## Geometry and model provenance

Source: `khelverskovp/atom-jepa` at
`bab08533b0d8547fc51ca9a10e3e02c36444ca3b` (MIT).
Molecular weights: `atom-jepa/atom-jepa` at
`8f7514f49693d836ca26dfe0bbae10880f75690b` (CC BY 4.0), SHA-256
`0406152686fbae9f737f2c70c9acaf2da0eba6ad57e530142adfee47df75882f`.
The private model directory retains the source-file hashes, model card, checkpoint
configuration, and complete environment dependency snapshot.

ETKDGv3 uses original stereochemical SMILES, explicit hydrogens, seed 42,
enforced chirality, no RMS pruning, and one conformer. MMFF94 runs at most 500
iterations; finite unconverged structures are retained and flagged. One
random-coordinate retry is allowed for failed embedding. Invalid structures,
disconnected fragments, absent MMFF parameters, and missing conformers fail
explicitly. Rows are never silently removed or reordered.

Conformer and feature caches contain labeled and blind molecules, but no label
values or split-dependent learned transformations. Grouping identities are used
only for validation, never as stereo-stripped encoder inputs. Higher-order norms
use `sqrt(sum_m(x**2) + 1e-8)` per atom before mean pooling. No trained upstream
head or learned normalization is imported. Cache extraction checks encoder state
stability, online/cache parity, and rotation/translation invariance.

The original provisional rotation tolerance of 1e-4 was exceeded by the upstream
model itself on both CPU and GPU: scalar relative L2 residuals were about 0.020%,
and invariant-feature residuals about 0.015%, with a maximum absolute change near
0.0011. CPU/GPU features agreed within atol/rtol 1e-4. The retained audit is
`runs/regression-atom-jepa-representations-v1-diagnostics/rotation.json`.
Before evaluating CYP scores, the geometry-only check was set to maximum absolute
change 0.005 and relative L2 change 0.001 (0.1%). This inherited model limitation
is recorded separately from unchanged cache parity and 1e-4 pIC50 prediction
replay requirements. Both failed extraction attempts remain charged.

## Environment and execution

For a fresh checkout, create the pinned GPU environment, fetch and verify the
reviewed source and weights, then install the verified source:

```bash
conda env create -f environment-atom-jepa.yml
conda activate somerra-atom-jepa
python -m pip install --no-build-isolation --no-deps -e .
python scripts/setup_atom_jepa.py
python -m pip install --no-build-isolation --no-deps -e runs/atom-jepa-source
python scripts/setup_atom_jepa.py --verify-only
```

The setup script pins the Git commit and Hugging Face revision, verifies every
reviewed source/model file against `configs/atom-jepa-provenance.json`, and writes
the exact local provenance expected by the loader. It does not execute downloaded
Python or replace existing assets. Interrupted downloads remain marked `.partial`.
Source and model licenses and attribution are retained in the private pin and
downloaded artifacts. No model weights or challenge data are committed.

Keep the activated Conda binary directory on PATH: the encoder uses its `ptxas`.
The historical pilot used the equivalent `runs/atom-jepa-env` overlay on
`runs/neural-env`; its first attempt failed without that PATH entry. Both failed
attempts remain in the original ledger. The fresh environment pins the directly
used scientific/model dependencies; it is not a byte-for-byte lock of every
transitive Conda dependency or the CUDA driver.

The full three-model comparison also needs the prepared official corpus and
the CheMeleon/Mol-JEPA prerequisite caches described in
[the CheMeleon runbook](chemeleon-implementation.md) and
[the Mol-JEPA runbook](mol-jepa-implementation.md). Restore their private cache and
model directories with original manifests intact. Mol-JEPA's existing permitted-use
evidence is required; Atom-JEPA setup does not create it. Prepare a new corpus only
when reproducing from scratch, never over an existing split:

```bash
openadmet-cyp download --output data/raw
openadmet-cyp regression prepare --data-dir data/raw --output-dir runs/regression-corpus-v1
```

GPU stages require `SOMERRA_CAMPAIGN_LEDGER` to name the existing initialized,
absolute shared campaign ledger, as required by current `main`. Do not create or
reset a ledger to run this pilot. Bind the **original existing** pilot history to
that campaign once, without copying either ledger or allocating GPU time:

```bash
python -m somerra_openadmet.frozen_pilot_budget \
  --campaign-ledger "$SOMERRA_CAMPAIGN_LEDGER" \
  --bind-history /absolute/original-checkout/runs/regression-gpu-budget.json
```

The shared campaign retains the absolute history path, its binding-time digest,
and committed pilot hours. Binding is idempotent for that path and refuses to
redirect an existing binding. All pilot stages and drivers resolve this binding,
independent of checkout or working directory. Missing or empty history is refused.
An explicit `ledger` config must name that same absolute file; relative or
different paths are rejected. Old generated configs with a relative `ledger`
must omit that field or use the bound absolute path before launching new work.
Historical run paths from other checkouts remain charged. CPU report replay also
uses the binding for accounting, but reserves no GPU time.
The original ledger's four-hour subset remains an additional atomic cap; new GPU
work is also reserved and charged by the current full-process campaign supervisor.
CPU-only setup, checks, and report replay allocate no GPU time.

```bash
python -m somerra_openadmet.cli regression cache-conformers \
  --config configs/representations-atom-jepa-conformers.json
python -m somerra_openadmet.cli regression cache-representations \
  --config configs/representations-atom-jepa-cache.json
python scripts/run_frozen_representation_pilot.py
```

Run these stages only when their output does not already exist. `--resume` on the
pilot driver skips only completed fits whose checkpoint, predictions, current
training data, split assignments, feature cache, and recipe still match.
Conformer evaluation and report replay perform the same current-input checks.
It refuses to overwrite failed fits. Do not regenerate completed
caches. All GPU work, including failed extraction attempts, is charged to the
existing shared ledger and the incremental four-hour pilot limit. The extraction
allocation is 1.5 hours and each head fit reserves at most 0.1 hour, reduced by the
remaining pilot balance. Both are ceilings, not estimates; the global 72-hour
ledger remains authoritative. CPU geometry generation has its own two-hour cap.

## Artifacts and interpretation

The campaign writes configs, frozen-head checkpoints (including scaling buffers),
training histories, split IDs, paired predictions, per-endpoint and macro metrics,
parameter counts, CPU replay errors, timings, and memory evidence. The report is
`runs/regression-frozen-representation-pilot-v1/report.md`.

The paired bootstrap resamples clusters jointly across endpoints and seeds and
estimates differences in mean per-seed ST-RAE; it does not score a seed ensemble.
Single-fold intervals remain exploratory. A later five-fold experiment needs a
new budget decision and must preserve the common downstream recipe.

Run manifests distinguish a completed requested pilot fold from complete five-fold
OOF coverage. The blind-export gate still rejects these partial-fold runs. No
model promotion, challenge submission, or organizer message is part of this work.
All code, weights, attribution, and reports remain private. Any future submission
remains prediction-only with `open_source_code=false` and `model_report_link=""`.

See [the recorded results](atom-jepa-results-20261029.md) for the complete pilot
summary and the distinction between historical results and PR hardening checks.
The `frozen-pilot-cpu` CI job installs PyTorch explicitly, requires zero skipped
pilot tests, and uses no model checkpoints or GPU. Run it locally with:

```bash
python -m venv .venv-pilot
.venv-pilot/bin/python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
.venv-pilot/bin/python -m pip install -r requirements/frozen-pilot-cpu.txt
.venv-pilot/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv-pilot/bin/python -m pytest tests/test_atom_jepa.py tests/test_frozen_benchmark.py \
  tests/test_frozen_pilot_budget.py tests/test_conformer_sensitivity.py \
  tests/test_frozen_resume.py tests/test_atom_jepa_setup.py
```
