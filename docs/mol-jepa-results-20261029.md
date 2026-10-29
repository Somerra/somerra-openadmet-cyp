# Mol-JEPA five-fold comparison — completed 2026-10-29

All eight Mol-JEPA variants and four matched CheMeleon controls completed five-fold OOF evaluation, CPU checkpoint replay, and validated 750-row blind prediction export. Nothing was promoted or submitted.

The primary frozen-MLP comparison uses identical prepared labels, Butina folds, masked raw-pIC50 MAE, head hidden width, and fixed training recipe. Native embedding widths differ (2,048 vs 512), so first-layer parameter counts differ. Full tuning additionally uses method-specific reference learning rates (CheMeleon 1e-4; Mol-JEPA 5e-5). These are matched recipe/family comparisons, not proof of intrinsic encoder superiority.

## Scores

Lower MA-ST-RAE is better. All results are screening evidence.

| Model/arm | Reference | H100 inner-selected |
|---|---:|---:|
| chemeleon frozen | 0.995752 | 1.033987 |
| chemeleon full | 0.983386 | 0.970013 |
| mol-jepa frozen | 0.952127 | 0.968801 |
| mol-jepa tokens | 0.796517 | 0.801297 |
| mol-jepa full | 0.874752 | 0.883469 |
| mol-jepa lora | 0.824743 | 0.826240 |

## Paired comparisons

Candidate minus reference; 1,000 paired molecular-cluster bootstrap samples. These intervals are unadjusted for exploratory multiple comparisons.

| Comparison | Relative improvement | 95% delta interval |
|---|---:|---|
| frozen-h100 | 6.30% | [-0.096843, -0.035739] |
| frozen-reference | 4.38% | [-0.071315, -0.014005] |
| full-h100 | 8.92% | [-0.112162, -0.059028] |
| full-reference | 11.05% | [-0.135972, -0.079535] |
| tokens-vs-frozen-reference | 16.34% | [-0.177040, -0.131932] |
| lora-vs-full-reference | 5.72% | [-0.071684, -0.029910] |
| tokens-vs-frozen-h100 | 17.29% | [-0.190986, -0.143162] |
| lora-vs-full-h100 | 6.48% | [-0.079069, -0.037191] |

Frozen/full pairs compare Mol-JEPA with CheMeleon. Tokens-vs-frozen and LoRA-vs-full compare Mol-JEPA adaptations; these extra pairs were added at review after results were available and are not predeclared encoder ablations.

## Limits and next step

The token transformer is the best Mol-JEPA arm (0.796517), but the separate [CheMeleon controls](chemeleon-controls-20261029.md) include cached-feature LightGBM at 0.775495 and a CPU incumbent at 0.743511. The weak reference MLP heads limit claims about encoder quality. Do not present this study as a new best overall system or select a submission from these outer-fold results alone.

Broad selection remains exploratory; promotion requires the separately budgeted nested/repeated/scaffold confirmation protocol. Fusion remains issue #15 and should be treated as a controlled follow-up, not an established improvement. No new training or GPU spend is implied by this report.

## Validation and accounting

All 40 Mol-JEPA fold checkpoints passed CPU replay; maximum absolute difference was 3.82e-6 pIC50 against the 1e-4 threshold. All 12 model variants have valid local blind exports. Final shared ledger usage is 17.727689 hours out of the explicitly approved 20-hour ceiling, leaving 2.272311 hours. The successful recovery took about 73 minutes. Charges include earlier unrelated shared work and the conservative interrupted-run charge; the ledger total is not Mol-JEPA training time alone.

The model pins, exact environment, per-endpoint scores, parameters, memory, timings, source snapshots, checkpoint hashes, bootstrap outputs, and prediction files remain in private local artifacts. See [implementation and evidence paths](mol-jepa-implementation.md). All failed-run artifacts were retained; the final recovery manifest, rather than historical stopped manifests, is the completion source of truth.
