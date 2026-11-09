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

## Addendum: diagnosis of the direction-reversed effect (diagnostic run, not a new pre-registered experiment -- same status as the PPO entropy-collapse diagnostic in docs/cue_validity_ppo.md)

**Hypothesis, checked directly rather than assumed**: the `naive` arm
trains its entire 800-episode budget at the low, fixed intervention
epsilon (0.04), with no exploration-rich phase at all, while the
`decorrelate` arm inherits a well-explored network from an 800-episode
*annealed*, high-start acquisition phase before its own 800-episode
intervention phase at the same low epsilon. If `interaction_only`
features are harder to fit cleanly from limited, low-exploration data
than `raw` features, the `naive` arm's Q-network could simply be
noisier/less converged than `decorrelate`'s by the time both are probed
-- and because `proxy_dependence` is reported as `|value|`, extra noise
inflates the naive arm's apparent magnitude directly. This would be a
genuine confound (unequal effective training regimes between arms), not
evidence of real anti-persistence -- structurally analogous to, but
mechanistically distinct from, the PPO total-episode-count confound
documented in `docs/cue_validity_ppo.md` (there: entropy-collapse
trajectory length; here: exploration-driven value-function noise).

**Direct check**: reran the `naive` arm at v=0.5 (same 15 seeds, same
features, same environment, same 800 total episodes) with the SAME
annealed epsilon schedule used for `decorrelate`'s acquisition phase,
instead of the fixed low intervention epsilon. Validated the unmodified
reproduction path first (seed 0's fixed-epsilon rerun reproduced the
original experiment's value, 0.006933, exactly).

| condition | signed mean | signed std | abs mean |
|---|---:|---:|---:|
| `naive`, fixed low epsilon (as in the real experiment) | 0.0155 | 0.0362 | 0.0335 |
| `naive`, annealed epsilon (diagnostic) | 0.0070 | 0.0193 | 0.0175 |
| `decorrelate`, v=0.5 (unchanged, for comparison) | 0.0035 | 0.0202 | 0.0148 |

**Result: the hypothesis is supported.** Matching `naive`'s exploration
schedule to `decorrelate`'s roughly halves both its variance (std 0.0362
to 0.0193) and its mean magnitude (abs mean 0.0335 to 0.0175), bringing
it in line with `decorrelate`'s own floor (std 0.0202, abs mean 0.0148)
rather than well above it. The direction-reversed, nominally significant
result in the main experiment is therefore substantially (if not
necessarily entirely, given this is a single diagnostic run rather than
a new pre-registered comparison with its own significance test) an
artifact of the `naive` arm's low-exploration training regime leaving its
`interaction_only` value function noisier than `decorrelate`'s, not
evidence of genuine anti-persistence. This is consistent with -- and
resolves the apparent tension with -- the main nonlinear experiment's
own finding (Section 7.4 of the manuscript) that `neural_interaction_only`
is statistically indistinguishable from its naive-control floor: here, in
the grid environment specifically, that floor itself was measured
unevenly between arms.

**What this does not do**: it does not re-run the comparison as a new
pre-registered test with its own significance criterion (that would
require pre-registering an exploration-schedule-matched variant of the
whole experiment before collecting new confirmatory data, which was
judged unnecessary given how directly this diagnostic already explains
the direction and magnitude of the original anomaly). The original
experiment's own reported numbers in
`reports/cue_validity_grid_neural_reversal_interaction_report.md` are left
unchanged; this addendum explains them rather than overturning the
pre-registered verdict (`has_own_effect: False`, which was already the
correct, honest reading given the pre-registered criterion).
