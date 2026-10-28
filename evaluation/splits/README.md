# Preserved evaluation splits

Scigantic model reconstruction and experiments are paused at the user's request
on 2026-10-28. Split identities are preserved here independently of those model
branches. Existing split generation, loaders and evaluation code remain on main.
This snapshot does not change the active autoresearch task or launch training.

`frozen-20261028/` is a portable, private, versioned assignment snapshot:

- `identities.csv`: 32,909 pool record IDs, standardized molecular parents,
  scaffold identities, and whether each record belongs to official training.
- `assignments.csv`: exact published reference fold, our balanced scaffold fold,
  and historical molecular partition for each official training record. Auxiliary
  records have an empty molecular partition; this does not mean they are excluded.
- `official-blind.csv`: the 750 external blind identities and parents, kept outside
  every development split. Their labels are unavailable locally.
- `manifest.json`: file SHA-256 hashes, original artifact hashes, counts and roles.

These files contain no activity values, predictions, or model checkpoints. They
remain in the private repository. Original data bundles and model artifacts remain
in their private run directories; this is a split archive, not a full data backup.
Join by `Molecule_Name` and validate parent identity; never attach a fold by row order.
Missing IDs, changed parents or new records require explicit handling, not a silent
random assignment or regeneration of the saved folds.

## Preserved protocols

| Protocol | Official labeled records | Status |
|---|---|---|
| Molecular similarity matching | Train 2,944 / validation 980 / internal test 981 | Existing autoresearch task and historical trials |
| Our balanced scaffold folds | 980 / 981 / 982 / 981 / 981 | Completed scaffold comparison |
| Exact published CheMeleon folds | 1,002 / 946 / 960 / 1,005 / 992 | Latest requested reference-fold comparison; assignments audited, new matched training not run |

All cover the same 4,905 official training molecules. The two five-fold maps also
cover all 32,909 eligible pool records. They are different partitions and must not
be confused. The reference fold map is not a universal split for all R17/R25
components: other members used different protocols.

For five-fold evaluation, held-out fold k is internal test, (k+1) modulo five is
validation, and the remaining three folds are training. Use validation to select
checkpoints. Held-out predictions evaluate that selected checkpoint; averaging
models that trained on those molecules would invalidate held-out evaluation.
Repeated use of held-out scores to choose recipes makes them development evidence,
not an untouched final test. Prior experiments have already exposed these molecules.

The completed single-model adaptation used our balanced fold 0 as validation and
folds 1–4 as training: 3,925 official training and 980 validation records. It had no
separate internal test. This view is fully recoverable from `balanced_fold`; it
must not be confused with the rotating 60/20/20 evaluation above. Final all-data
refitting is a separate training phase, not permission to alter evaluation folds.

Molecular matching assigns official records to fixed train/validation/internal-test
partitions. Auxiliary records whose standardized parents occur in either holdout
are excluded from training; otherwise eligible auxiliary parents can train. For
scaffold protocols, all sources inherit the same fold membership. Across every
protocol, exact official-blind parents remain excluded from model fitting. Model
configuration may select a subset of training sources without moving held-out IDs.

## Verification and reuse

Run `python -m pytest tests/test_preserved_splits.py` from the repository root.
Checks cover hashes, coverage, record/parent consistency, fold counts, parent and
balanced-scaffold separation, and official-blind exclusion. Hash validation does
not require access to the old absolute source paths recorded for provenance.
Existing training commands still use their verified data bundles; this snapshot
preserves the identity mapping for future model-independent loaders. It does not
pretend that exact reference-fold training is already integrated into those commands.

Do not replace this snapshot when experimenting with a new split. Version a new
assignment separately and retain this one for reproducibility. No model-family
choice, ensemble requirement or new training authorization is attached to it.
