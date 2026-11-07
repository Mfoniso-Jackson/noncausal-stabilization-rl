# PPO Capacity Sweep: Pre-Registration

Written and committed before any results from this experiment exist.
Parallel to `docs/cue_validity_capacity_sweep.md` (the DQN/`neural_raw`
capacity sweep), but using PPO's now-validated methodology from
`docs/cue_validity_ppo.md`: an 800-episode acquisition budget (calibrated
to PPO's own entropy-collapse dynamics, not borrowed from DQN) and a
total-episode-matched naive control (2300 total episodes for both arms),
established there as the methodologically correct comparison for this
algorithm. Applying both corrections from the start, rather than
rediscovering them per width.

## What this tests

Whether PPO's small, direction-consistent-but-underpowered effect
(`docs/cue_validity_ppo.md`'s final result, Cohen's d 0.3-0.5 at the
default 32-unit hidden layer) changes with network capacity, the same
question the DQN capacity sweep asked of `neural_raw`.

## Design

- Widths: **{16, 64}**, in addition to the already-collected width-32
  result (`results/cue_validity_ppo_retest_summaries.csv` for
  `decorrelate`, `results/cue_validity_ppo_total_matched_naive_summaries.csv`
  for the total-matched `naive`) -- reused directly, not rerun. Scoped to
  2 new widths rather than DQN's 5-point sweep, given this experiment's
  secondary priority and PPO's higher per-run cost.
- `cue_validity` in {0.5, 0.95}, 15 seeds, matching every PPO experiment
  in this lineage.
- `decorrelate` arm: 800-episode acquisition, 1500-episode
  post-intervention (2300 total). `naive` arm: 2300 total episodes,
  entirely in the decorrelated environment (the total-matched
  methodology).
- `PPOAgent(hidden_size=16)` / `PPOAgent(hidden_size=64)`, all other
  hyperparameters at their defaults, unchanged.

## Metrics

`proxy_dependence`, read via the same exact-probe method.

## Success / failure criteria (fixed before data collection)

Report, for each of the 3 widths (16, 32 reused, 64) at each validity
level, whether the decorrelate-vs-total-matched-naive gap is significant
(`compare_groups`, p < 0.05, Cohen's d, Holm-Bonferroni correction within
this 6-cell family). Given the width-32 result is already known to be
underpowered rather than null, this sweep is explicitly **not** expected
to show a clean monotonic capacity curve the way the DQN sweep did --
reported descriptively, whatever the shape, not forced into that framing.

## What would change this plan after the fact

Nothing above will change after results are inspected, following the
same discipline as every prior pre-registration in this lineage.

## Addendum (added after the run below; the sections above are unchanged)

**No width shows a significant effect after correction, as anticipated
given the width-32 result was already known to be underpowered rather
than null.** More informative is the shape: effect size is **not
monotonic in capacity**, unlike DQN's clean capacity curve
(`docs/cue_validity_capacity_sweep.md`). Cohen's d is weakest at width 16
(0.10 at v=0.5, 0.21 at v=0.95), peaks at the default width 32 (0.32,
0.54), and **reverses sign** at width 64 (-0.29, -0.13). A reversed sign
means `naive` was numerically larger than `decorrelate` at that width --
the same wrong-direction pattern seen in this document's earlier,
confound-ridden attempts, now reappearing at a specific capacity rather
than universally. With only 15 seeds per cell and effect sizes this
small, this could be genuine non-monotonicity (e.g. a wider network
converges faster, reaching further along its own entropy-collapse
trajectory within the same 2300-episode budget, mechanically producing a
smaller or reversed gap -- directly analogous to the `auto_navigate`
capacity-sharing-vs-step-count distinction worked out for the grid
environment) or could be noise at this seed count; this document does not
have enough seeds to tell the two apart and does not claim to. Reported
descriptively, as pre-registered, not forced into a capacity-curve
narrative the data does not clearly support.
