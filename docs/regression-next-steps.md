# Next regression cycle

## Objective and fixed evaluation boundary

Improve the official challenge test score from the verified R25 reference (0.4436).
The challenge's provided blinded test remains our test set. Validation is drawn
only from official training data. Eligible official auxiliary and external data
may supplement training, with the existing held-out parent/analogue exclusions.
Keep validation and official test metrics separate. Preserve private source
attribution; submit predictions only, Open Code false and report link empty.

## Immediate submission candidate

The verified R25 reconstruction is available without new fitting. Its prediction
SHA256 is `bc5b8065790ce5e47e087c7a0f0a9ab5e9e0904e1c524a481d24f13491a3134d`.
Reusing it would restore the stronger baseline, not constitute a new improvement.
The last accepted submission was 2026-10-25 01:27:49 UTC; allow the 12-hour cooldown
plus a conservative margin, targeting no earlier than 13:29 UTC. Recheck the live
portal and local receipts immediately before any upload. Do not assume a new
submission deletes historical metadata. No upload is part of this PR update.

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
