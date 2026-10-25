# Next regression cycle

## Objective and fixed evaluation boundary

Improve our trained submission (0.7790), using Scigantic reference (R25; 0.4436)
as an explicitly attributed external benchmark.
The challenge's provided blinded test remains our test set. Validation is drawn
only from official training data. Eligible official auxiliary and external data
may supplement training, with the existing held-out parent/analogue exclusions.
Keep validation and official test metrics separate. Preserve private source
attribution; submit predictions only, Open Code false and report link empty.

## Reference reuse is on hold

Scigantic reference (R25) reconstructs Scigantic's published predictions. Its 0.4436
score is reference reproduction evidence, not a result of our trained model. Our
own trained submission scored 0.7790. The prediction checksum is
`bc5b8065790ce5e47e087c7a0f0a9ab5e9e0904e1c524a481d24f13491a3134d`.

Do not resubmit this reference as a new candidate. Published portal guidance permits
external training data and pretrained models and does not require open code for
standard ranking, but does not explicitly settle reuse of another participant's
published prediction CSV. Organizer clarification and an acceptable attribution
arrangement remain unresolved. No organizer has been contacted. If required
attribution conflicts with the user's no-link policy, stop and explain the conflict;
do not silently publish a link or present third-party predictions as our work.

The local readiness record is on hold. A validated CSV alone is not submission
clearance. Develop and evaluate our own trained candidate instead.

## Ordered work

1. **Establish a comparable strong reference.** Inventory R25 members, training
   schedules, pretraining, fold assignments, weighting and calibration. Reproduce
   the available frozen predictions separately from retraining. Identify which
   upstream OOF results are genuinely held out for every fitting/calibration step.
   Deliver a component/provenance table and a bounded retraining estimate before
   committing compute. Do not equate the 15-epoch width-700 screen with R25.
2. **Compare validation designs within official training.** Retain current cluster
   CV as a fixed comparator. Audit Scigantic folds, and add a parent-grouped random
   comparator. Compare similarity, scaffold coverage, endpoint label availability,
   confidence intervals and molecular properties with the official train/test
   boundary. Any test-chemistry-matched design must use structures only, be declared
   before model selection and be reported as test-informed. Freeze the chosen
   splits before fitting; do not tune splits to obtain favorable model rankings.
3. **Run a small matched confirmation.** After explicit finite compute allocation,
   compare a strong reference with all auxiliary groups versus without TDI, using
   the same architecture, initialization policy, schedule and splits. Use repeated
   seeds and predeclare either fixed-step exposure or report the fixed-epoch
   confounding. Avoid repeating the entire 54-cell screen initially.
4. **Check calibration and combinations.** Fit calibration and any blend weights
   inside training/inner-validation folds, then evaluate on untouched outer folds.
   Blind prediction means/SDs are diagnostics, not justification for arbitrary
   rescaling to R25. Examine endpoint errors and performance by similarity strata.
5. **Select one candidate for official evaluation.** Record the reference, validation
   evidence, uncertainty, candidate checksum and intended hypothesis. Verify IDs,
   units, finite values, fold/checkpoint provenance and privacy fields. Use a
   separately authorized submission and preserve its receipt and official score.

## Decision criteria and limits

Report matched validation differences, per-endpoint regressions, seed variability
and paired uncertainty. Existing local numerical gates are screening criteria, not
proof of superiority to the official-test incumbent. Two structurally different
submissions cannot establish validation/test score correlation. Do not promote a
candidate as an official improvement until its official score supports that claim.
No new training allocation or automatic upload is authorized by this plan.

## Review follow-up: analogue exclusions and challenge design

The organizer describes the test set as analogue expansion around potent training
hits, not a random or uniformly distant chemical holdout. This explains why our
cluster validation is more chemically separated and motivates a hit-expansion
validation comparator within official training. Sparse training versus dense test
assays also changes endpoint coverage and should be measured.

The published portal FAQ permits external training data; no requirement for our
0.6 test-analogue cutoff was found in the inspected guidance. The cutoff is our
conservative research choice, not an organizer rule. Preserve the completed study's
filter and artifacts. In a separately budgeted matched comparison, compare exact
parent exclusions against exact-plus-analogue exclusions, keeping validation-fold
exclusions fixed and recording per-source molecules/labels removed. This isolates
the test-based filter; report its mismatch with validation exclusions explicitly.
New adapter manifests separate exact-parent from analogue-only exclusions and
record excluded source observations. The original excluded-row CSV permits a
retrospective count without changing historical manifests. That audit found 144
excluded upstream rows/parents, all analogue-only; exact-parent exclusions were zero
in this pinned derivative table. This does not describe exclusions in other sources.

Sources inspected 2026-10-25: [portal FAQ](https://openadmet-cyp-challenge.hf.space/)
and [organizer dataset construction](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/).
These links are private documentation only, never submission metadata.
