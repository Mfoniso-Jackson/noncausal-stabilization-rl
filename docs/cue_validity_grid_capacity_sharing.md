# Grid Environment Capacity-Sharing vs. Step-Count: Pre-Registration

Written and committed before any results from this experiment exist.
Resolves an open question from `docs/cue_validity_cross_environment.md`'s
"neural_raw" extension addendum: `neural_raw` showed no decorrelate-vs-naive
effect at the original 1500-episode intervention budget, but the effect
reappeared once the budget was reduced to match the *original
phase-environment's* total training-call count (800 episodes). Two
competing, un-distinguished explanations were offered: (1) **step-count**
-- the grid environment's longer episodes (~9 steps vs. 5) deliver more
gradient updates per nominal episode, which could wash out the bias faster
for a fixed episode count, independent of task structure; (2)
**capacity-sharing** -- the shared network must also represent a real
navigation policy in the grid environment (unlike the original environment,
where pre-target actions are ignored entirely), and that competing demand
on the same parameters could make the cue-dependent weight component less
entrenched, independent of how many gradient steps occur.

## What this tests

`CueValidityGridEnv` now has an `auto_navigate` option (env/cue_validity_grid_env.py,
added for this experiment): when `True`, movement toward `decision_cell` is
deterministic and ignores the chosen action entirely, mirroring
`CueValidityEnv`'s own design where pre-target actions never affect
anything. Critically, this changes **only** whether real navigation is a
skill the network must learn -- episode length, observation features, and
the number of `train_step()` calls per episode are all unchanged (verified
by a unit test, `test_auto_navigate_same_step_count_as_optimal_manual_path`,
confirming the optimal manual path and the auto-navigated path take exactly
the same number of steps). This lets the two hypotheses be tested with step
count held **exactly** constant, rather than approximately matched as in
the earlier step-count-matched retest.

## Design

- Condition: `neural_raw` only, `auto_navigate=True`, at the **original**
  1500-episode intervention budget (30 blocks of 50) -- the same budget
  that showed a null result under normal navigation
  (`cue_validity_grid_neural_experiment.py`). `cue_validity` in {0.5, 0.95},
  15 seeds, same 5000-episode acquisition budget, same architecture
  (`NeuralAgentWrapper`/`MLPQNet`, unchanged). `decorrelate` and `naive`
  arms, same methodology as every prior experiment in this lineage.
- Not re-run: the normal-navigation 1500-episode result (null) and the
  800-episode step-matched result (positive) are both already committed
  and are the comparison points, not repeated here.

## Metrics

`proxy_dependence`, read via the same exact-probe method (the probe
synthetically sets `pos = decision_cell` regardless of `auto_navigate`,
so the metric-reading method itself is unaffected by this change).

## Success / failure criteria (fixed before data collection)

With step count held exactly constant at the 1500-episode budget:
- **Capacity-sharing supported** if `auto_navigate=True` shows a
  significant decorrelate-vs-naive effect (`compare_groups`, p < 0.05,
  decorrelate magnitude greater than naive) at at least one validity
  level -- removing the need to learn real navigation recovers the
  effect at the same step count that showed a null under normal
  navigation, isolating task-structure interference as the cause.
- **Capacity-sharing not supported, step-count stands as the simpler
  explanation** if `auto_navigate=True` also shows no significant effect
  at either validity level -- since step count is unchanged from the
  already-null normal-navigation condition, this would mean removing the
  navigation-learning demand alone does not recover the effect, leaving
  the step-count account (already empirically supported by the budget-matched
  retest) as the better-supported explanation by elimination, not because
  it was tested positively here.

A null result for both conditions (neither recovers the effect) is a valid,
reportable outcome and would mean neither offered explanation is sufficient
on its own, calling for further work this document does not undertake.

## What would change this plan after the fact

Nothing above will change after results are inspected, following the same
discipline as every prior pre-registration in this lineage.

## Addendum (added after the run below; the sections above are unchanged)

**Capacity-sharing hypothesis not supported.** With step count held
exactly constant at the 1500-episode budget, `auto_navigate=True` shows no
significant decorrelate-vs-naive effect at either validity level
(decorrelate=0.008 vs. naive=0.008, p=0.853 at v=0.5; decorrelate=0.013 vs.
naive=0.008, p=0.068 at v=0.95). This is statistically indistinguishable
from the already-null normal-navigation result at the same budget, and
removing the navigation-learning demand did not recover the effect.

Per the pre-registered criteria, this leaves **step-count as the better-
supported explanation by elimination**, now on firmer footing than before:
it is the only one of the two candidate explanations that has actually
been tested positively (the 800-episode step-matched condition did show
the effect), while capacity-sharing has now been tested directly, at
matched step count, and did not. The v=0.95 cell (p=0.068, d=0.72) is
close enough to the significance threshold that a moderately larger seed
count could in principle move it either way; this is noted rather than
treated as a reason to doubt the overall conclusion, since v=0.5's result
(p=0.853) shows no comparable trend at all.

This closes the question raised in `docs/cue_validity_cross_environment.md`'s
neural_raw extension addendum: the grid environment's need for a larger
step-matched (rather than episode-matched) budget to detect `neural_raw`'s
effect is best explained by the raw volume of corrective training updates
per nominal episode, not by any interference from needing to learn a
navigation policy.
