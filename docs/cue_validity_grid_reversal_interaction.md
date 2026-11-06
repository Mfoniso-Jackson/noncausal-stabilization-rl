# Grid Environment: Reversal and Interaction-Only Extension

Written and committed before any results from this experiment exist.
Builds on `docs/cue_validity_cross_environment.md` (which confirmed
tabular-null/linear-positive/neural-positive-with-matched-budget under
**decorrelation only**) and on this session's multiple-comparison
correction policy (`docs/multiple_comparison_correction.md`), applied
here from the start rather than retrofitted.

## What this tests

Two of this lineage's established manipulations have not yet been tried
in `CueValidityGridEnv`: the **reversal** intervention (does the
gradient-asymmetry mechanism -- reversal adapts faster than decorrelation
-- replicate here too?) and an **interaction-only** feature encoding
(does removing direct coordinate access shrink the effect here too, as
it did in the original environment's nonlinear experiment?).

## Design

- **Reversal arm**: `linear_raw` only (the grid environment's cleanest,
  most decisive positive condition), `env.reverse()` in place of
  `env.decorrelate()`, same `cue_validity` levels {0.5, 0.95}, 15 seeds,
  same budget as the main cross-environment replication (5000
  acquisition, 1500 post-intervention in 30 blocks of 50). Reversed and
  naive-reversed arms, same naive-control methodology used throughout
  this lineage. **Primary metric is the normalized fraction-of-required-
  adaptation, from the start** (not the raw gap) -- this lesson was
  already learned the hard way in `docs/cue_validity_reversal.md` and is
  applied prospectively here rather than repeated. `tabular` is not
  re-run under reversal (expected null, already well-established); this
  extension does not re-test it.
- **Interaction-only arm**: new feature encoding,
  `grid_interaction_only_features()`: the same 5 non-cue features as
  `CueValidityGridEnv.observation_vector()` (`x_norm, y_norm,
  at_decision_flag, dx_norm, dy_norm`) plus 3 interaction terms
  (`cue_signed * x_norm`, `cue_signed * y_norm`, `cue_signed *
  at_decision_flag`, where `cue_signed = 2*cue_bit - 1`) in place of the
  raw `cue_bit` -- 8 features total, no term that is purely the cue bit
  alone, mirroring `cue_validity_nonlinear_experiment.py`'s
  `interaction_only_features()` exactly. `linear_interaction_only`
  (`DQNAgent` over these 8 features), decorrelate intervention only (not
  reversal -- matching the scope of the original nonlinear experiment),
  same validity levels, seeds, and budget. `decorrelate`/`naive` arms
  only; compared against `linear_raw`'s already-committed decorrelate
  result from `cue_validity_cross_environment_experiment.py`.

## Metrics

`proxy_dependence`, read via the same exact-probe method (probe at the
decision cell, cue fixed to 0/1).

## Success / failure criteria (fixed before data collection)

**Reversal**: the gradient-asymmetry mechanism is supported if
`linear_raw`'s reversal fraction-adapted is significantly larger
(`compare_groups`, p < 0.05, Cohen's d reported) than its decorrelation
fraction-adapted (from the already-committed cross-environment data), at
either validity level.

**Interaction-only**: the effect is coordinate-dependent if
`linear_interaction_only`'s decorrelate-vs-naive gap is significantly
smaller than `linear_raw`'s (the same criterion
`cue_validity_nonlinear_experiment.py` used), and/or if
`linear_interaction_only` itself shows no significant decorrelate-vs-naive
effect at all.

**Correction applied from the start**: both families (2 cells each) are
Holm-Bonferroni corrected within their own family before being reported
as significant, per `docs/multiple_comparison_correction.md`'s policy.
Null or non-surviving results are valid, reportable outcomes.

## What would change this plan after the fact

Nothing above will change after results are inspected, following the
same discipline as every prior pre-registration in this lineage.

## Addendum (added after the run below; the sections above are unchanged)

**Reversal: confirmed, decisively, even after correction.**
`linear_raw`'s reversal fraction-adapted (0.975, 0.876) is significantly
larger than its decorrelation fraction-adapted (0.467, 0.419) at both
validity levels, p<0.0001 both (Holm-adjusted p<0.0001 both, Cohen's
d=5.84/2.09). The gradient-asymmetry mechanism replicates in the grid
environment, completing the pattern already shown in the original
environment.

**Interaction-only: coordinate-dependent, own effect borderline.** The
coordinate-dependence test (`linear_raw`'s gap vs.
`linear_interaction_only`'s) is decisive (p<0.0001 both levels, survives
correction). `linear_interaction_only`'s own decorrelate-vs-naive effect
(0.042/0.028 and 0.013/0.028) does not survive Holm correction within its
own 2-cell family (raw p=0.044/0.031, adjusted 0.063/0.063) -- a
borderline-to-null own effect, smaller than `linear_raw`'s (0.056-0.106
in the main cross-environment report). This is consistent with, and
extends, the original environment's graded-effect finding: removing
direct coordinate access shrinks the effect in this environment too.

Net result: the grid environment now reproduces every piece of the
original environment's story -- tabular null, linear strong under both
decorrelation and reversal, a smaller/borderline effect once direct
coordinate access is removed -- holding up under the same
multiple-comparison correction standard applied throughout this
session.
