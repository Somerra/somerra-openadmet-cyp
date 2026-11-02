# Two-rotation hit-expansion evaluation

On November 2, 2026, the user requested two rotations to enlarge each internal
test set. The saved v5 bundle derives from the existing v4 bank without selecting
new hits, changing bank membership, or changing any training rows. Earlier
bundles and saved Workbench folds remain intact.

| Rotation | Training rows | Checkpoint validation | Internal test | D6 test labels |
| --- | ---: | --- | --- | ---: |
| 42 | 32,686, fixed | B: 111 compounds | A: 112 compounds | 23 |
| 43 | 32,686, fixed | A: 112 compounds | B: 111 compounds | 19 |

Group A has 26/22/23/73 measured direct-inhibition labels for A2/C9/D6/A4;
group B has 16/16/19/83. These counts were inspected after assignment and did
not influence the grouping. Whole connected seed–analog nomination families
are assigned largest first to the group with fewer parents, with deterministic
structural and group-index tie breaks. Assay values and availability masks do
not enter assignment.

The pooled scoring cohort remains 223 unique compounds, with 42 D6 labels.
Each row is scored once, using the model checkpoint chosen on the other group.
Training CSVs are byte-identical to v4. Both groups and official blind parents
remain excluded from all training sources. Validation sets are smaller than
under three rotations, so checkpoint selection may become noisier. The new
grouping does not create additional independent evidence or undo prior adaptive
exposure of these compounds.

Use fresh matched fits for the incumbent and candidates. Historical checkpoints
selected using v4 validation groups cannot supply valid v5 test predictions:
their selection may have seen rows now assigned to their test group. Keep
training-seed replication separate from these two evaluation rotations. Pool
the two disjoint scoring tables before computing ST-RAE, and continue checking
Workbench OOF performance and ranking before promoting submission candidates.

Private bundle: `runs/hit-expansion-frozen-bank-v5-two-rotations-20261102`.
The manifest records source hashes, exact grouping rules, per-assay coverage,
and the requirement for fresh fits. No training or submission was launched by
this split change.

Reproduce into a new, nonexistent output directory:

```bash
.venv/bin/python -m somerra_openadmet.hit_expansion repartition \
  --root /mnt/data/somerra-openadmet-runs/hit-expansion-frozen-bank-v4-20261101 \
  --output /path/to/new-two-rotation-bundle
.venv/bin/python -m somerra_openadmet.hit_expansion audit \
  --root /path/to/new-two-rotation-bundle \
  --output /path/to/new-audit
```

`prepare` retains its historical three-rotation default; `repartition` explicitly
creates the separately versioned two-rotation protocol. Research runners must
read its manifest's `repeats: [42, 43]`, rather than assume three rotations.
