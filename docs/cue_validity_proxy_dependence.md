# Cue Validity Proxy-Dependence & Representation: Pre-Registration

Written and committed before any results from this experiment exist. This
is a direct continuation of `docs/cue_validity_mve.md` and
`docs/tabular_rl_cannot_be_superstitious.md`; read those first.

## What changed from the MVE, and why

The MVE's primary metric was CDS (softmax mass on the non-causal action).
`docs/tabular_rl_cannot_be_superstitious.md` showed this cannot persist
under tabular TD learning for a structural reason: the non-causal action
never pays off, so it cannot acquire stable value regardless of
representation. That argument does not extend to this repository's other
established metric, `proxy_dependence` (P(causal action | cue present) -
P(causal action | cue absent), used in
`experiments/causal_reversal_agent_experiment.py` and
`experiments/neural_recurrent_reversal_experiment.py`): this asks whether
the policy's *correct*-action probability is still modulated by a
now-invalidated cue, which is ordinary slow re-convergence after a
distribution shift and is not ruled out by the CDS argument, under any
agent class including tabular.

This experiment tests `proxy_dependence` (primary) and CDS (secondary, for
continuity with the MVE) together, across three agent/representation
conditions, to directly test whether any effect found is tied to a
specific linear representation -- the question this repository's own
`linear_feature_ablation_experiment.py` -> `orthogonalized_proxy_feature_experiment.py`
-> `nonlinear_interaction_proxy_experiment.py` chain already investigated
for a different environment, and concluded was the case there.

## Design

- Environment: `env/cue_validity_env.py::CueValidityEnv` with
  `cue_phase = target_phase` (the cue is folded into the decision state
  itself, so the causal and cue-associated actions genuinely compete for
  the same terminal-reward credit at the same state -- the structural
  change `docs/tabular_rl_cannot_be_superstitious.md` showed was necessary,
  even though it does not revive CDS as a usable metric there).
- Agent/representation conditions (3):
  1. `tabular`: the same SARSA(lambda) agent as the MVE, over
     `observe()`'s discrete state index.
  2. `linear_raw`: `agents.dqn_agent.DQNAgent` (this repository's existing,
     validated dependency-free linear agent) over
     `observation_vector()`'s raw 6 features.
  3. `linear_orthogonal`: the same `DQNAgent`, over the same 6 features
     passed through the fixed 6x6 Hadamard rotation already used in
     `orthogonalized_proxy_feature_experiment.py` -- an invertible linear
     mixing of the identical information, so any difference between this
     condition and `linear_raw` isolates representation (coordinate
     accessibility), not information content.
- `cue_validity` in {0.0 (floor), 0.5, 0.95}. For 0.5 and 0.95: a
  decorrelate arm (acquisition then `decorrelate()`, continued training in
  blocks) and a naive-control arm (fresh agent, trained only in the
  already-decorrelated environment for the same post-intervention episode
  budget), exactly as in the MVE. For 0.0: an undisturbed arm and its own
  naive control, as the same spurious-gap check.
- 15 seeds per condition (reduced from the MVE's 30 to control the added
  cost of the two linear-agent conditions; this is a power/cost tradeoff
  made explicitly, not a change hidden in the results).
- Episode budget: same as the MVE (3000 acquisition, 1500
  post-intervention in 30 blocks of 50).

## Metrics

- `proxy_dependence` (primary): `metrics/cue_validity.py::proxy_dependence`,
  computed from two probe evaluations of the trained agent/function at the
  target-phase decision state, once with cue_present=1 and once with
  cue_present=0 -- a direct function query for the linear/tabular agents,
  not a noisy Monte-Carlo estimate.
- `CDS` (secondary, continuity with the MVE):
  `metrics/cue_validity.py::cue_dependence_score`, probed the same way.
- Both read at the acquisition-end checkpoint and at each post-intervention
  block, exactly as in the MVE.

## Success / failure criteria (fixed before data collection)

The hypothesis is **supported** for a given agent/representation condition
only if, for at least one non-zero validity level:
1. the decorrelate arm's final-block `proxy_dependence` is significantly
   higher in magnitude (two-sample t-test, p < 0.05, Cohen's d reported)
   than the matched naive-control arm's, AND
2. the floor control does not itself show a comparable gap.

Representation is implicated as necessary (not merely the environment or
metric) if `linear_raw` shows a supported effect that `linear_orthogonal`
does not, or shows it more weakly (smaller effect size at matched p), under
otherwise identical seeds, episodes, and intervention timing -- mirroring
`orthogonalized_proxy_feature_experiment.py`'s own criterion for calling an
effect a "linear accessibility artifact." If `tabular` also shows a
supported effect on `proxy_dependence` (plausible, since this metric is not
ruled out by the tabular argument), that is reported as evidence that
representation sharing is not a precondition for this particular construct
of dependence, which would need to be stated plainly rather than
downplayed.

A null result in every condition is a valid, reportable outcome.

## Addendum (added after the run below; the sections above are unchanged)

Result: the pre-registered criteria are met for both `linear_raw` and
`linear_orthogonal` at both non-zero validity levels (p < 0.0001, Cohen's d
8-24 in every supported cell), and not met for `tabular` at either level
(p = 0.90, d near zero). `linear_orthogonal`'s effect size is comparable to
`linear_raw`'s, so representation is not implicated as a raw-coordinate
artifact by this experiment's own pre-registered criterion. Replay-buffer
staleness is excluded by direct calculation (the buffer cycles roughly 2.5x
over during the measurement windows). This is the first properly
controlled, pre-registered, statistically decisive positive finding in
this research program. See `reports/cue_validity_proxy_dependence_report.md`
for the full results tables and the gradient-asymmetry mechanism proposed
to explain why decorrelation (used here) and reversal (used in
`experiments/spurious_correlation_reversal_experiment.py`, tabular agent)
might differ in extinction speed -- not yet tested directly in one matched
design, and recommended as the next experiment. This addendum does not
change any criterion above.

## What would change this plan after the fact

Nothing in the metric definitions, conditions, or success criteria above
will change after results are inspected. Any bug found in the shared
`env/cue_validity_env.py` or `metrics/cue_validity.py` modules themselves
will be fixed in a separate, dated commit noting that re-running was
required, exactly as in the MVE's pre-registration.
