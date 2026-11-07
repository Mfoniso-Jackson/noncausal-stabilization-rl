# Multiple-Comparison Correction Across the Cue-Validity Lineage

This is a retrospective statistical-policy application across
already-collected data (no new experiments run, except the capacity
sweep's results, added below once available). It answers the
publication plan's Section 7 gap: "a stated correction (Holm-Bonferroni
or an explicit false-discovery-rate target)... rather than treating
[these comparisons] as free exploration."

## Policy

**Holm-Bonferroni, applied within each experiment's own pre-registered
comparison set (one family per experiment), at alpha=0.05.** Rationale
for the family boundary: each cue-validity experiment's pre-registration
names a specific set of conditions/validity-level cells as its decisive
comparisons, decided before data collection; treating that whole set as
one family (rather than, say, every p-value across the entire lineage as
one giant family, or each cell as its own unconnected test) matches how
the claims are actually used -- a reader evaluating "did this experiment
find the effect" is implicitly asking about every cell in that
experiment's own design at once. Comparisons whose own pre-registered
success criterion is **non-significance** (the shuffled/placebo control:
success means p >= 0.05) are excluded from this correction and reported
separately, uncorrected, since Holm-Bonferroni corrects in the direction
of making significance harder to achieve -- applying it to a test
designed to find *no* effect would make it easier, not harder, to "pass,"
which is the wrong direction for added rigor. The seed-topped-up
`neural_raw` result (not the original, less-powered one) is used in the
neural-reversal family below, to avoid counting the same underlying
comparison twice.

## Results, by experiment

| Experiment (family) | n | Survives correction |
|---|---:|---|
| MVE (criterion 1) | 2 | None (both were already non-significant, as expected -- tabular null) |
| Proxy-dependence (criterion 1) | 6 | `linear_raw` and `linear_orthogonal`, both validity levels (4/6); `tabular` does not (expected null) |
| Reversal (pre-registered raw-gap) | 6 | `linear_raw` and `linear_orthogonal`, both validity levels (4/6); `tabular` does not. (Significance here reflects the gaps reliably *differing*, in the direction the pre-registered criterion itself judged as not supporting the hypothesis -- see that experiment's own report.) |
| Nonlinear (coordinate-dependence test) | 4 | `linear` architecture, both validity levels; `neural` survives only at v=0.5 (p=0.021 -> adjusted 0.042), not at v=0.95 (already non-significant before correction) |
| Nonlinear (own decorrelate-vs-naive effect, H4's basis) | 6 | `linear_interaction_only` and `neural_raw`, both validity levels (4/6); `neural_interaction_only` does not (already non-significant before correction). H4's claim is unaffected by correction. |
| **Neural reversal (pre-registered normalized)** | 6 | **None.** See correction below -- this reverses an earlier reported result. |
| Optimizer ablation (primary) | 4 | `neural_raw` both validity levels; `neural_interaction_only` only at v=0.95 |
| Replay buffer sweep | 8 | All 8 (every p-value was already <0.0001 before correction) |
| Capacity sweep | 10 | Widths 8, 16, 32 at both validity levels (6/10); width 64 does not survive at either level after correction (v=0.95's raw p=0.0147 adjusts to 0.059, just missing); width 128 was already null |

Full per-cell raw and Holm-adjusted p-values are in the analysis script
output, reproduced in `reports/multiple_comparison_correction_report.md`.

## Correction to a previously-reported result: neural reversal

`docs/cue_validity_neural_reversal.md`'s addendum reported
`linear_interaction_only` as supporting the gradient-asymmetry mechanism
at both validity levels, citing p=0.0237 (v=0.5) and p=0.0190 (v=0.95)
against the uncorrected alpha=0.05 threshold. Under Holm-Bonferroni
correction across that experiment's own full pre-registered family of 6
comparisons (3 conditions x 2 validity levels), **both p-values adjust to
0.095 and no longer clear alpha=0.05.** The seed-topped-up `neural_raw`
result (p=0.0143 at v=0.95, previously reported as "the asymmetry is
supported there") also adjusts to 0.086 and no longer clears the
threshold. **None of the 6 cells in that experiment's family survive
correction.**

This does not mean the gradient-asymmetry mechanism is false -- the
*uncorrected* reversal experiment on `linear_raw`/`linear_orthogonal`
(the original two conditions, a separate family from the neural-reversal
extension) survives correction cleanly, and the mechanism's core claim
rests on that result plus the normalized-metric reasoning in
`docs/cue_validity_reversal.md`. What this correction overturns is
specifically the **extension** of that finding to `linear_interaction_only`,
`neural_raw`, and `neural_interaction_only` -- the neural-reversal
experiment's own family-wise evidence for the mechanism generalizing to
those three conditions is weaker than previously reported, once held to
the same correction standard applied here to everything else. This is
flagged as a correction to `docs/cue_validity_neural_reversal.md` and to
any summary (including the publication plan) that reported the
`linear_interaction_only` or topped-up `neural_raw` results as
"supported" without this correction.

## Capacity sweep, after correction

The capacity sweep's own monotonic-decline framing is unaffected by
correction (effect size itself does not change), but the boundary of
where the effect stops being statistically detectable tightens: widths
8, 16, and 32 survive correction at both validity levels; width 64 -- 
significant before correction at v=0.95 (p=0.0147) -- does not survive
(adjusted p=0.059, just missing). The honest statement is therefore "the
effect is detectable through width 32 and not reliably beyond it," not
"through width 64," once the full 10-cell family is corrected together.

## What this changes in the overall claim

The core, publication-plan-central result -- `linear_raw`/`linear_orthogonal`
show large persistent `proxy_dependence` after decorrelation
(proxy-dependence experiment), tabular does not, and reversal produces
faster adaptation than decorrelation for the same two linear conditions
(reversal experiment) -- is **unaffected**: every cell underpinning it
survives Holm-Bonferroni correction within its own experiment's family,
most with p<0.0001. The weakened claim is specifically the *extension* of
the reversal/gradient-asymmetry story to the three additional
representations tested only in the neural-reversal experiment and its
seed top-up; that extension should now be reported as **not
significant after correction**, not as "supported," until a
better-powered, pre-registered-as-corrected rerun is done.
