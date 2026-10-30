# Auxiliary-target relationships, PCA, and specialist follow-up

Analysis date: 2026-10-30. Private research documentation. The follow-up is tracked in [issue #23](https://github.com/Somerra/somerra-openadmet-cyp/issues/23) and the [specialist plan](issues/workbench-all-aux-single-model-per-cyp.md).

## Findings and decision

Auxiliary labels capture substantial shared assay signal. The training ablations support keeping auxiliary supervision collectively, but neither their results nor label PCA establish that every auxiliary is independently beneficial. Retain all 22 auxiliaries for the proposed four-specialist experiment, including qHTS. Each specialist has its own scored potency head and the 22 auxiliary heads. No specialist training was performed as part of this analysis.

Removing the four qHTS potency columns changes how global PCA distributes variance, while preserving its main relationship to measured primary potency. That is evidence of redundant label information, not evidence that dropping those tasks improves a trained predictor.

## Task inventory and preparation

The local expanded recipe has 26 targets: four official direct-inhibition pIC50 targets; four single-concentration log2 fold changes near 49.5 micromolar; four TDI-condition pIC50 targets; five ChEMBL potency targets; five PubChem maximum-response targets; and four PubChem qHTS potency targets. The two five-target families include CYP2C19. Historical Workbench mt_aux8 contains four primary plus four auxiliary targets; union18 contains four primary plus 14 auxiliaries. These are different scopes.

The [26-target dictionary](results/auxiliary-target-task-dictionary.csv) records every target's identity, definition, unit, direction, source, preprocessing, observed count, clipping, and censoring limitations. ChEMBL is an external bioactivity database; qHTS denotes quantitative high-throughput screening. Here qHTS potency comes from AC50 values despite the stored `pic50_qhts` names. Higher pIC50 denotes greater potency. PubChem maximum response is signed percentage response, with negative values indicating inhibition and upstream clipping to [-150, 50]. ChEMBL/PubChem labels may reuse measurements; original assay record identifiers are unavailable in this bundle, so common provenance is not proven by numerical agreement.

The frozen `runs/regression-hillclimb-20261026-data` bundle was checked against the existing run manifest. Its manifest SHA-256 is `520ac3056a0df3046c730a9d409bb7f94b136dabb7e1bd48bc3e30e94ff0b910`. The input has 32,909 records, reduced to 32,875 standardized molecular parents by taking the median within each parent/task: 34 duplicate records collapsed. Parent identity uses the frozen neutralized largest organic fragment without stereochemistry. Replicate counts describe bundle records, not laboratory repeats. No supplied blinded parent overlapped the analyzed rows.

There are 184,198 observed labels, 78.45% missing entries, and zero complete 26-target rows. Missing labels are not inactivity or censoring bounds. Confidence intervals remain metadata rather than PCA variables. The retained 22-target analysis uses exactly the same 32,875 parents.

## Transformations, correlations, and missing data

Compare four fixed representations, each followed by per-target standardization:

1. Existing pIC50, log2FC, and maximum-response scales (primary reference).
2. Inverse logs: potency to micromolar concentration via `10^(6-p)` and log2FC to fold change via `2^x`; maximum response unchanged. The potency direction reverses.
3. Signed log: apply `sign(x)*log1p(abs(x))` to maximum response only.
4. Yeo-Johnson fitted to each existing-scale target.

Distribution diagnostics cover quantiles, skewness, ties, clipping and extreme-value influence. Do not choose a representation merely for larger correlations or explained variance. Exponentiation can amplify extremes: for inverse-transformed CYP2D6 TDI potency, the most extreme 1% accounts for about 84.9% of variance in the saved diagnostics.

Pairwise Pearson and Spearman correlations use jointly observed labels. Coefficients below 30 pairs are suppressed; 30–99 are flagged. Reported edges receive 500 molecular-parent bootstrap resamples. The following native-scale direct-potency/single-concentration relationships include pair counts and descriptive Pearson 95% bootstrap intervals:

| task_a | task_b | n | pearson | spearman | pearson_low | pearson_high |
| --- | --- | --- | --- | --- | --- | --- |
| CYP1A2_pIC50_direct_inhibition | CYP1A2_log2fc_4.9504951e-05M | 1412 | -0.887 | -0.862 | -0.898 | -0.874 |
| CYP2C9_pIC50_direct_inhibition | CYP2C9_log2fc_4.9504951e-05M | 1285 | -0.831 | -0.896 | -0.851 | -0.811 |
| CYP2D6_pIC50_direct_inhibition | CYP2D6_log2fc_4.9504951e-05M | 1490 | -0.705 | -0.828 | -0.727 | -0.684 |
| CYP3A4_pIC50_direct_inhibition | CYP3A4_log2fc_4.9504951e-05M | 1805 | -0.882 | -0.936 | -0.890 | -0.872 |

For matching ChEMBL/qHTS labels, Pearson correlations span 0.9947–0.9977 on 2,512–5,400 paired parents. Between 96.3% and 98.4% agree within 1e-6. Exact per-pair counts and agreement statistics are in the [aggregate evidence](results/auxiliary-target-analysis-evidence.json).

## Complete-case PCA

Transformations and scaling were fitted separately within each cohort, using identical rows across representations. These existing-scale results show the strong selection differences:

| group | n | targets | pc1_fraction | components_80 |
| --- | --- | --- | --- | --- |
| primary | 41 | 4 | 51.2% | 2 |
| chembl | 2171 | 5 | 42.0% | 4 |
| pubchem_ac50 | 791 | 4 | 41.5% | 3 |
| pubchem_response | 16130 | 5 | 47.9% | 4 |
| single | 4372 | 4 | 55.7% | 3 |
| tdi | 42 | 4 | 47.3% | 3 |
| original8 | 41 | 8 | 41.9% | 3 |
| external14 | 675 | 14 | 32.3% | 6 |

The original eight-target subset has only 41 complete parents. Its components cannot be assumed representative of the full labeled dataset. The single-concentration and maximum-response families provide much larger complete cohorts.

## Global PCA and removal of qHTS

Observed labels were transformed/standardized before Bayesian-ridge iterative imputation: five posterior draws, seeds 42–46, 20 iterations each. Completed matrices were standardized for PCA. Median imputation is a sensitivity analysis. Masking 10% of observed entries, while retaining at least one observed entry per row, evaluated per-task reconstruction against a median baseline. This random masking does not validate systematically missing source/assay blocks. Availability/source associations were also diagnosed. Global PCA remains imputation-dependent.

Full spectra, scores, PCA coefficients and target–component correlations were saved. Coefficients describe the fitted linear axes; correlations describe association with scores and are not interchangeable. Cross-imputation comparisons align component order/sign and inspect subspace angles when axes rotate. Signs themselves are arbitrary.

The qHTS comparison includes a fresh 22-target imputation and a fixed-completion control that removes qHTS from the previously completed 26-target matrix. The latter isolates direct column weighting but still contains indirect qHTS information through the original imputation. Complete-case external comparisons use the same 675 parents first; the expanded 10-target cohort of 1,859 is reported separately.

On existing scales, seed-42 global PC1 explained variance changes from 22.71% (26 targets) to 27.51% (fresh 22 targets); the fixed-completion 22-target control yields 26.39%. These percentages have different total-variance denominators. They do not establish improved prediction or better imputation. Original 26-target median imputation yields only 13.38% for PC1, versus 22.71% for posterior seed 42; eight versus thirteen components reach 80% variance, respectively.

The table below correlates component scores with each **observed** primary target, using its measured-label subset rather than imputed target values. Values are means over five sign/order-aligned imputations; the after_min/after_max columns are draw ranges, not confidence intervals. Representation: existing scales; global cohort: 32,875 parents; n: observed primary labels.

| task | n | before_mean | after_mean | after_min | after_max |
| --- | --- | --- | --- | --- | --- |
| CYP1A2_pIC50_direct_inhibition | 1412 | 0.695 | 0.702 | 0.694 | 0.709 |
| CYP2C9_pIC50_direct_inhibition | 1285 | 0.792 | 0.789 | 0.776 | 0.802 |
| CYP2D6_pIC50_direct_inhibition | 1490 | 0.124 | 0.120 | 0.097 | 0.139 |
| CYP3A4_pIC50_direct_inhibition | 2334 | 0.859 | 0.846 | 0.842 | 0.856 |
| CYP1A2_log2fc_4.9504951e-05M | 4372 | -0.700 | -0.714 | -0.726 | -0.702 |
| CYP2C9_log2fc_4.9504951e-05M | 4372 | -0.825 | -0.822 | -0.829 | -0.816 |
| CYP2D6_log2fc_4.9504951e-05M | 4372 | -0.422 | -0.438 | -0.443 | -0.431 |
| CYP3A4_log2fc_4.9504951e-05M | 4372 | -0.810 | -0.806 | -0.810 | -0.795 |
| CYP1A2_pIC50_TDI_condition | 1413 | 0.688 | 0.696 | 0.689 | 0.702 |
| CYP2C9_pIC50_TDI_condition | 1285 | 0.791 | 0.790 | 0.781 | 0.802 |
| CYP2D6_pIC50_TDI_condition | 1494 | 0.157 | 0.154 | 0.133 | 0.169 |
| CYP3A4_pIC50_TDI_condition | 3582 | 0.831 | 0.814 | 0.805 | 0.825 |
| cyp1a2_pic50_chembl | 10967 | 0.223 | 0.265 | 0.231 | 0.294 |
| cyp2c9_pic50_chembl | 10552 | 0.150 | 0.096 | 0.067 | 0.151 |
| cyp2d6_pic50_chembl | 8412 | 0.096 | 0.055 | 0.016 | 0.079 |
| cyp3a4_pic50_chembl | 15774 | 0.140 | 0.090 | 0.043 | 0.159 |
| cyp2c19_pic50_chembl | 10990 | 0.138 | 0.100 | 0.065 | 0.133 |
| cyp1a2_max_response | 16130 | -0.347 | -0.373 | -0.385 | -0.361 |
| cyp2c9_max_response | 16130 | -0.208 | -0.231 | -0.263 | -0.213 |
| cyp2d6_max_response | 16130 | -0.236 | -0.270 | -0.285 | -0.256 |
| cyp3a4_max_response | 16130 | -0.232 | -0.227 | -0.247 | -0.202 |
| cyp2c19_max_response | 16130 | -0.252 | -0.295 | -0.315 | -0.268 |

PC1 remains associated with CYP1A2, CYP2C9 and CYP3A4 potency, and inversely with their log2FC assays. PC2 continues to distinguish CYP2D6 potency from its single-concentration response. After qHTS removal, PC3 most closely matches old PC4 and contrasts CYP1A2/CYP2D6 with CYP2C9/CYP3A4. PC4 is less stable and rotates relative to old PC3; PC5 is largely external-assay variation with weaker primary-target associations. Detailed target rankings are in the local comparison report. These are descriptive associations, not causal or predictive effects.

## What training ablations establish

The later 100-epoch full-recipe study achieved best validation MA-ST-RAE 0.65986; direct-only scored 0.82017, without single-concentration supervision 0.73313, without ChEMBL 0.68935, without TDI 0.68810, without PubChem response 0.66473, and without qHTS potency 0.66002. Lower is better. The near equality of full and without-qHTS does not resolve a small effect from one seed.

These are selected checkpoints from **one seed-42 random molecular-parent 80/20 split**, not a named scaffold fold or five-fold average: 3,924 official training rows and 981 validation rows (3,920 and 981 parents). All auxiliary records for validation parents were excluded, but analog/cluster overlap remained. The model was the Scigantic-derived Chemprop encoder, width 2,048/depth 4/dropout 0.2, with a two-layer width-1,024 prediction head and batch 128. Data and optimizer exposure differ when auxiliary sources are removed; the full run was resumed. Thus this is supporting development evidence, not an isolated, independently replicated auxiliary-effect estimate.

The earlier 15-epoch five-fold width-700 campaign is a separate experiment: full 0.67944 versus direct-only 0.76340, while without-TDI scored 0.66629. Its conflicting TDI direction is another reason to avoid claiming that each auxiliary is essential. Local evidence: `runs/campaign-20261024-data-ablation/summary.csv`, `runs/reference-member-20261025/reference-100/attempt-1/run_manifest.json`, and `runs/reference-member-20261025/ablations-100/*/run_manifest.json`.

## Generalist loss and proposed follow-up

The [specialist plan](issues/workbench-all-aux-single-model-per-cyp.md) documents the exact current weights and competition scoring rules. Historical Workbench weights its four standardized primary losses by inverse label counts (mean coefficient one), and each auxiliary by 0.3. The expanded full 26-target generalist uses task coefficients of one plus nonuniform row weights. Both divide weighted error sums by observed-entry count; neither guarantees equal aggregate task influence.

The proposed four-model experiment retains Workbench's architecture and training recipe, adds all 22 local auxiliaries, proposes primary coefficient 3.0 versus 0.3 per auxiliary, and selects each checkpoint on its focal primary validation ST-RAE. The multiplier is a proposal, not a learned optimum. Compare with a matched shared 26-output Workbench control, keeping preprocessing and molecular partitions fixed. The experiment tests specialization plus reweighting and incurs four encoders' aggregate cost. It does not isolate either factor separately.

## Artifacts and verification

Full local reports (not committed with this documentation):

- `runs/target-label-analysis-20261030-v1/report.html` and `target-label-analysis.pdf` (91 pages).
- `runs/target-label-pca-without-qhts-20261030-v1/report.html` and `pca-without-qhts-comparison.pdf` (19 pages).
- Each run includes CSVs, source snapshots, transformation parameters, manifests, versions, seeds, and checks. The original run has 79 PNG and 79 SVG figures.

Both saved verification receipts passed: 56 original PCAs and 56 new comparison PCAs, four deterministic imputation reruns per run, unchanged frozen inputs/observed labels, PCA orthogonality and correlation checks, and independent checks of twenty primary-target/component correlations in the comparison. These are completed-run checks, not newly repeated training or a claim that this documentation PR executes the analysis. [Aggregate evidence and artifact hashes](results/auxiliary-target-analysis-evidence.json) preserve the receipts and key tables.

The reproducible CPU scripts and module snapshots remain in the local run artifacts/workspace: `scripts/analyze_target_labels.py`, `scripts/verify_target_label_analysis.py`, `scripts/compare_pca_without_qhts.py`, and `scripts/verify_pca_without_qhts.py`. This documentation-only PR does not package executable reproduction or large per-molecule outputs; the local artifacts are required for a full rerun.

Source definitions are attributed per task in the dictionary. Upstream Workbench source was inspected at `205029d30942f062548806f3ff8d0536bbb24db2`; Scigantic source at `e70a2920d611684c9aaff07400d951e2e79dd5f6`. Package versions and bundle file hashes are in the aggregate evidence. Unresolved raw-assay lineage and systematic missingness remain limitations.

Keep all implementation, reports and provenance private. Nothing in this PR authorizes training, challenge submission, publication, or organizer messages. Future authorized challenge submissions remain prediction-only with `open_source_code=false` and an empty `model_report_link`.
