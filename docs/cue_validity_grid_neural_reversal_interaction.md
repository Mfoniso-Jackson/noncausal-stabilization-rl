# Grid Environment: neural_raw Reversal and neural_interaction_only Extension

Written and committed before any results from this experiment exist.
Completes the grid environment's representation x intervention matrix:
`linear_raw` has been tested under decorrelate, reversal, and (via
`linear_interaction_only`) coordinate-dependence
(`docs/cue_validity_grid_reversal_interaction.md`); `neural_raw` has only
been tested under decorrelate (`docs/cue_validity_cross_environment.md`'s
extension, plus the step-count-matched retest). This experiment adds
`neural_raw` under reversal and a new `neural_interaction_only` condition
under decorrelate.

## What this tests

Whether the gradient-asymmetry mechanism (reversal adapts faster than
decorrelation) and the coordinate-dependence pattern (interaction-only
encoding shrinks the effect), both already confirmed for `neural_raw`/
`linear_interaction_only` in the *original* environment, also hold for
`neural_raw` specifically in the grid environment -- completing the
"does the whole graded-effect story replicate" question this lineage has
been building toward, for the one representation x environment
combination not yet covered.

## Design

- **Budget, decided upfront from already-established evidence**: `neural_raw`
  in this environment requires the **step-count-matched 800-episode**
  intervention budget to show any effect at all (confirmed in
  `docs/cue_validity_cross_environment.md`'s step-count-matched retest
  addendum -- 1500 episodes is null, 800 is not). Both new arms below use
  800 episodes (16 blocks of 50) for the post-intervention period, not
  1500 -- applying that lesson from the start rather than repeating the
  mistake of testing an already-known-underpowered budget.
- **Reversal arm**: `neural_raw` (`NeuralAgentWrapper`/`MLPQNet`, unchanged,
  over the 6 raw grid features), `env.reverse()` in place of
  `env.decorrelate()`, same validity levels {0.5, 0.95}, 15 seeds, same
  5000-episode acquisition. Reversed and naive-reversed arms. **Primary
  metric is the normalized fraction-of-required-adaptation**, applied
  from the start (the lesson from `docs/cue_validity_reversal.md`).
  Compared against `neural_raw`'s own step-count-matched decorrelate
  fraction-adapted (`results/cue_validity_grid_neural_stepmatched_summaries.csv`),
  not the original, known-underpowered 1500-episode result.
- **Interaction-only arm**: new feature function,
  `grid_interaction_only_features()` (identical to
  `cue_validity_grid_reversal_interaction_experiment.py`'s, reused
  unchanged: 5 base features + 3 cue-interaction terms, 8 total),
  `neural_interaction_only` (`NeuralAgentWrapper` over these 8 features),
  decorrelate intervention only, 800-episode intervention budget, same
  validity levels and seeds. Compared against `neural_raw`'s own
  step-matched decorrelate result (coordinate-dependence test, same
  criterion as `orthogonalized_proxy_feature_experiment.py`'s and this
  lineage's own nonlinear experiment).

## Metrics

`proxy_dependence`, read via the same exact-probe method.

## Success / failure criteria (fixed before data collection), with correction applied from the start

Both families (2 cells each) are Holm-Bonferroni corrected within their
own family before being reported as significant, per
`docs/multiple_comparison_correction.md`'s policy.

**Reversal**: supported if `neural_raw`'s reversal fraction-adapted is
significantly larger (Holm-corrected p < 0.05) than its step-matched
decorrelation fraction-adapted, at either validity level.

**Interaction-only**: `neural_interaction_only` is coordinate-dependent if
its own decorrelate-vs-naive gap is significantly smaller (Holm-corrected
p < 0.05) than `neural_raw`'s step-matched gap, and/or if
`neural_interaction_only` itself shows no significant decorrelate-vs-naive
effect at all (consistent with the original environment's finding that
combining nonlinearity with reduced coordinate access eliminates the
effect).

Null or non-surviving results are valid, reportable outcomes.

## What would change this plan after the fact

Nothing above will change after results are inspected, following the
same discipline as every prior pre-registration in this lineage.

## Addendum (added after the run below; the sections above are unchanged)

**Reversal: confirmed decisively, survives correction at both validity
levels.** `neural_raw`'s reversal fraction-adapted (1.350, 1.166) exceeds
its step-matched decorrelation fraction-adapted (0.898, 0.894), p=0.0065/0.0111
raw, both adjusting to p=0.0131 after Holm correction (Cohen's d=1.11/1.03).
This completes the grid environment's reversal story: both `linear_raw`
and `neural_raw` show the gradient-asymmetry mechanism there, matching
the original environment.

**Interaction-only: coordinate-dependence confirmed at v=0.5, not at
v=0.95.** The coordinate-dependence test (raw gap vs. interaction-only
gap) is significant at v=0.5 (p<0.0001, survives correction) but not at
v=0.95 (p=0.061, does not survive). `neural_interaction_only`'s own
decorrelate-vs-naive effect does not show the expected direction at
either level: at v=0.5 its naive arm (0.033) is actually *larger* than
its decorrelate arm (0.015) -- a significant difference (raw p=0.0097)
but in the reverse direction, so `has_own_effect` is correctly `False` by
the pre-registered criterion (which requires the significant difference
to point the expected way). At v=0.95 there is no significant own effect
either direction (p=0.74). This reversed-direction result at v=0.5 is
reported honestly as an unexplained, secondary observation -- it does not
by itself overturn the coordinate-dependence conclusion at v=0.5 (which
compares gap *magnitudes* between `neural_raw` and
`neural_interaction_only`, not the sign of `neural_interaction_only`'s own
effect), but it means `neural_interaction_only`'s own behavior in this
environment is less clean than the original environment's version of the
same condition, and is not further diagnosed here.

**Net picture**: the grid environment now reproduces the original
environment's full representation x intervention story for `neural_raw`
(decorrelate, reversal, and partial coordinate-dependence), with one
honestly-reported loose end (`neural_interaction_only`'s own
direction-reversed effect at v=0.5) left for a future, more targeted
investigation rather than smoothed over.
