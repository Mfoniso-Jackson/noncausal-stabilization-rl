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
