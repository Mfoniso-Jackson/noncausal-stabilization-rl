# Cue Validity Cross-Environment Replication: Pre-Registration

Written and committed before any results from this experiment exist.
Environment: `env/cue_validity_grid_env.py::CueValidityGridEnv`, already
implemented and unit-tested (`tests/cue_validity_grid_env_test.py`, 11
tests, all passing) before this document was written -- building and
testing the environment is infrastructure, not data collection, and is
not itself subject to this pre-registration discipline; the experiment
that uses it is. This is the item named in the publication plan's Section
5 ("cross-environment replication") and is the explicit gate in Section
12's Go/No-Go criteria: "proceed to full paper if cross-environment
replication confirms the graded pattern in at least one structurally
different task," "narrow scope... if cross-environment replication fails
or is inconsistent."

## What this tests

Every positive result so far comes from one environment family
(`CueValidityEnv`), varied only by parameter. This experiment asks whether
the core pattern -- persistent proxy_dependence after decorrelation under
function approximation, absent under tabular representation, absent at
the validity floor -- replicates in a structurally different task:
`CueValidityGridEnv` embeds the same causal mechanism (one designated
causal action pays off only when an unobserved per-episode hidden state is
1; cue correlates with hidden state at swept validity; decorrelate flips
that off) inside real multi-step 2D grid navigation, where
`CueValidityEnv.step()` instead ignores the agent's action entirely until
a single phase, a trivial countdown. Reaching the decision cell in the new
environment requires the agent to actually learn spatial credit
assignment, not just count steps -- a genuine structural difference, not
a relabeling of the same task.

## Design

- Environment: `CueValidityGridEnv(cue_validity, grid_size=5, seed=...)`.
  `grid_size=5`, start `(0,0)`, decision cell `(4,4)`, `max_steps=20`
  (Manhattan distance 8 + 12 slack, matching `CueValidityEnv`'s
  `horizon_slack` convention). Verified by direct smoke test (not a
  pre-registered result, an infrastructure check): a tabular SARSA(lambda)
  agent learns to navigate to the decision cell in ~9 steps on average
  within 5000 episodes (trained in 1.3s), and shows a clean
  acquisition-phase `proxy_dependence` of ~0.21 at validity=0.95 --
  directly comparable in magnitude to `CueValidityEnv`'s own acquisition
  signal, confirming the environment is learnable and the metric is
  sensitive before committing to the full design below. A linear agent
  (`DQNAgent` over `observation_vector()`) was likewise checked to learn
  within the same budget.
- Conditions: **`tabular`** (`SarsaLambdaAgent`, reused unchanged from
  `experiments/cue_validity_mve_experiment.py`, with
  `state_count = env.state_count` in place of `env.horizon * 2 + 2`) and
  **`linear_raw`** (`DQNAgent` over the 6-feature `observation_vector()`,
  reused unchanged from `experiments/cue_validity_nonlinear_experiment.py`'s
  `LinearAgentWrapper` pattern). Neural and interaction-only-encoding
  conditions are **not** included in this first replication -- the
  Go/No-Go bar is "at least one structurally different task" showing the
  graded pattern, and the minimal pair that demonstrates
  representation-dependence (tabular null vs. linear positive) is
  sufficient to test that bar, matching this lineage's own
  minimum-viable-first philosophy (`docs/cue_validity_mve.md`). Extending
  to neural/interaction-only here is explicitly deferred to a follow-up,
  contingent on this core pair replicating.
- `cue_validity` in **{0.0 (floor), 0.5, 0.95}** -- matching the original
  MVE's three-level design (floor control included, not just the two
  non-floor levels used by the later experiments in this lineage), since
  this is a foundational replication and deserves the same rigor as the
  first experiment in the lineage.
- 15 seeds per condition/validity/arm.
- Episode budget: **5000** acquisition episodes (increased from the
  3000 used throughout the rest of this lineage, because this environment
  requires learning navigation jointly with the decision, confirmed
  sufficient by the smoke test above), 1500 post-intervention episodes in
  30 blocks of 50 (unchanged).
- Intervention: **decorrelate only** (not reversal) -- this experiment
  tests whether the core persistence result replicates in a new task
  structure; reversal and neural conditions in the new environment are
  explicitly deferred.
- Arms per (condition, validity): `none` (acquisition only, no
  intervention -- diagnostic, mirrors the MVE), `decorrelate` (acquisition
  then `env.decorrelate()` then continued training), `naive` (fresh agent,
  decorrelated environment only, same post-intervention budget).

## Metrics

`proxy_dependence` (primary) and `CDS` (secondary), read via the same
exact-probe method used throughout this lineage: set `env.pos =
env.decision_cell` and `env.cue_present` directly, read the agent's
action-values at that synthetic state, exactly as `probe_q_values` does in
`cue_validity_proxy_dependence_experiment.py` for the phase-based
environment.

## Success / failure criteria (fixed before data collection)

The cross-environment replication **confirms the graded pattern** (the
Go/No-Go "proceed to full paper" condition, as far as this one experiment
can establish it) if, in `CueValidityGridEnv`:

1. `linear_raw` shows a significant decorrelate-vs-naive
   `proxy_dependence` gap (`compare_groups`, p < 0.05, decorrelate
   magnitude greater than naive, Cohen's d reported) at at least one
   non-floor validity level (0.5 or 0.95).
2. `tabular` shows no such significant gap at either non-floor validity
   level (mirroring H2).
3. The floor control (`cue_validity=0.0`) shows no significant gap for
   either condition (mirroring the original MVE's criterion 2 -- the
   comparison methodology itself must not manufacture a spurious gap).

All three must hold for the replication to be reported as confirming the
pattern. If (1) fails (`linear_raw` shows no effect anywhere), or (2)
fails (`tabular` unexpectedly shows an effect), or (3) fails (the floor
shows a spurious gap), the replication is reported as **failing or
inconsistent** -- the Go/No-Go "narrow scope" condition -- exactly as
observed, not reframed. A partial result (e.g. criterion 1 holds at only
one of the two non-floor levels) is reported as a qualified, not a clean,
replication.

## What would change this plan after the fact

Nothing in the metrics, conditions, or criteria above will change after
results are inspected, following the same discipline as every prior
pre-registration in this lineage.

## Addendum (added after the run below; the sections above are unchanged)

**All three criteria are met cleanly.** `linear_raw` shows a significant
decorrelate-vs-naive gap at both non-floor validity levels (p<0.0001 both,
Cohen's d=6.12 at v=0.5, d=5.44 at v=0.95 -- large effects, comparable in
scale to the original environment's `linear_raw`/`linear_orthogonal`
results). `tabular` shows no significant gap at either level (p=0.25,
0.21). The floor control shows no spurious effect for either condition
(p=0.35, 0.29). **The publication plan's Go/No-Go "proceed to full paper"
condition is met by this experiment's own pre-registered terms.**

This is a materially stronger and more decisive result than merely
"replicates" -- the effect sizes here (d=5.4-6.1) are among the largest
in the entire lineage, and the pattern (null under tabular, large
significant effect under linear, no floor artifact) reproduces exactly,
in a task that required the agent to learn real multi-step spatial
navigation to even reach the decision point, not just a trivial phase
countdown. This substantially strengthens the claim that the persistence
phenomenon is about representation (shared, parametric value estimates
leaking across differently-cued contexts), not an artifact of
`CueValidityEnv`'s specific, structurally minimal task design.

What this does not yet establish: reversal, neural agents, and
interaction-only encoding were explicitly deferred in this environment
(see Design); the graded pattern found in the original environment
(shrinking under either factor, vanishing under both) is not yet checked
here. These remain natural follow-ups, now with a strong prior that they
will reproduce, but not confirmed. This addendum does not change the
criteria above; the verdict stands exactly as the pre-registered test
computed it.
