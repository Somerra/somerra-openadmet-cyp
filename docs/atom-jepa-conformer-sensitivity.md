# Frozen Atom-JEPA conformer sensitivity follow-up

This extension uses the remaining balance of the original four-GPU-hour pilot,
including all historical extraction attempts and diagnostics. Fine-tuning remains
deferred. All artifacts remain private; no submission or blind prediction export.

The design is fixed before new results are examined:

- Same persisted outer fold 0 and MLP seeds 42/43/44.
- Same 300-epoch, batch-256, Adam 1e-3, weight-decay 1e-4, float32 MLP:
  input -> 512 -> BatchNorm -> ReLU -> four outputs; endpoint-balanced masked
  MAE in raw pIC50; final epoch. Same molecule batches and optimizer update counts.
- Both scalar (256) and invariant (768) features are standardized. Statistics are
  fitted to outer-training rows of the original seed-42 conformer only and remain
  identical across the 1/2/10-conformer training arms. This isolates conformer
  effects from a changed scaler. No outer labels select settings or checkpoints.
- Three new standardized scalar single-conformer fits. Existing standardized
  invariant single-conformer fits are reused with checksum validation.
- Twelve new fits: two feature views x 2/10 training conformers x three head seeds.
  Each molecule occurs once per epoch; a separate seeded RNG selects one available
  conformer uniformly for that molecule. Neither sample count nor update budget
  increases. The molecular shuffle RNG remains unchanged.
- Every head is evaluated on conformer 0, the first two conformers, and all ten.
  Multiple-conformer scores average predictions, not embeddings. Additional
  per-conformer predictions quantify sensitivity without retraining or selection.
- The first conformer and its embeddings are reused exactly. Nine additional
  independent ETKDGv3 seeds (43 through 51), explicit hydrogens, enforced chirality,
  and MMFF94 (500 steps) provide nested sets of 1, 2, and 10 conformers. Finite
  unconverged geometries are flagged; failures abort rather than remove rows.
  No energy or label based conformer selection. Duplicate relaxed conformations
  may occur, particularly for rigid molecules; ten generated views does not imply
  ten distinct conformational minima.
- Extra geometries and embeddings are generated only for the 4,905 labeled
  molecules. The frozen encoder runs in evaluation mode; its weights and buffers
  must be bitwise unchanged. Geometry and feature cache checksums are retained.

The primary follow-up contrast is standardized scalar vs standardized invariant
with one conformer. Conformer contrasts compare train-2/test-1 and train-10/test-1
with train-1/test-1 (augmentation), test-2/test-10 with test-1 for a fixed head
(prediction averaging), and train-2/test-2 or train-10/test-10 with train-1/test-1
(combined effect). All results are reported, with means/sample SD across seeds and
paired cluster-bootstrap intervals. Baseline CheMeleon and Mol-JEPA scores are
fixed references from the original pilot; this extension does not tune them.

This is an exploratory follow-up motivated by already-observed pilot results.
Repeated evaluation on fold 0 is not independent confirmation. Additional-fold
validation would be needed before claiming a general model ranking.

Artifacts are nested under
`runs/regression-frozen-representation-pilot-v1/conformer-sensitivity-v1` so the
original budget accountant charges every GPU stage, including failed attempts.
Each stage reserves at most the remaining balance; CPU geometry stages are timed
separately. The original completed pilot artifacts are not overwritten.

Run with the existing environment (completed stages are checksum-verified and
reused; incomplete stages are refused rather than overwritten):

```bash
# Activate the environment and set the existing campaign ledger as described
# in atom-jepa-frozen-pilot.md. No new allocation is initialized here.
python scripts/run_atom_jepa_conformer_sensitivity.py
```

The driver also accepts `--stage geometry`, `extract`, `fit`, or `report`.
`report.md`, `summary.csv`, `paired_comparisons.json`, and `budget_summary.json`
are written in the follow-up root. Checkpoints and immutable embeddings permit
replay without loading a trainable encoder.
