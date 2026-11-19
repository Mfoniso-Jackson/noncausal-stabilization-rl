# Cue Validity Uncertainty Sweep: Pre-Registration

Written and committed before any results from this experiment exist.
Direct response to `docs/symbolic_stabilization_audit.md` Section 6's
recommended next experiment: a properly controlled test of RQ3 ("does
uncertainty increase non-causal stabilization"), fixing the confound found
in the pre-existing `GridCueEnv`/`run_pipeline.py` uncertainty pipeline
(Section 2 of that audit), where agent exploration/regularization
hyperparameters were set as a direct function of the uncertainty label
itself.

## What this tests, and the one non-negotiable design constraint

Whether an agent's post-intervention persistence on a non-causal cue grows
with environmental outcome uncertainty, when uncertainty is manipulated
**only** as an environment property and **no agent-side setting is ever a
function of the uncertainty condition**. This is the fix the audit
identified as the single necessary correction to the prior design; every
other methodological choice below reuses this project's already-validated
conventions unchanged.

## Design

- Environment: `env/cue_validity_env.py::CueValidityEnv`, `cue_phase =
  target_phase = 4` (same fold-in-the-decision-state design validated since
  `cue_validity_proxy_dependence_experiment.py`), with the new `reward_noise`
  parameter added to the environment this session (see that class's
  docstring): with probability `reward_noise`, a correctly-earned causal
  reward is withheld. This is the single new environment property; nothing
  else about the environment changes.
- `reward_noise ∈ {0.0, 0.3, 0.6}` ("none", "moderate", "high"), crossed with
  `cue_validity ∈ {0.0, 0.95}`. `v=0.0` is the floor control at every noise
  level: it must remain null (no decorrelate-vs-naive effect) regardless of
  noise, which rules out reward noise itself manufacturing a spurious
  apparent effect via increased Q-value estimation variance feeding the
  `abs()` transform in `proxy_dependence` (the same artifact mechanism
  identified and confirmed in `docs/cue_validity_grid_neural_reversal_interaction.md`'s
  addendum for `neural_interaction_only`). `v=0.95` is the primary condition,
  reusing the validity level with the strongest, most-replicated effect in
  this project's prior work.
- Agent: `linear_raw` only (`agents/dqn_agent.py::DQNAgent`, unwrapped,
  identical hyperparameters to every prior use of this condition --
  `lr=0.015, gamma=0.95, replay_size=8000, batch_size=32, target_update=50,
  train_frequency=4`). Fixed regardless of `reward_noise` or `cue_validity`,
  by construction (this is the constraint the whole experiment exists to
  enforce). No other agent class is tested here; this experiment is about
  the uncertainty manipulation, not a second representation sweep.
- 15 seeds, matching the established convention for this effect-size scale
  (`cue_validity_proxy_dependence_experiment.py`).
- Episode budget: identical to every prior `linear_raw` experiment in this
  lineage -- 3000 acquisition episodes (annealed epsilon 1.0 to 0.04), then
  `decorrelate()`, then 1500 post-intervention episodes (30 blocks of 50,
  fixed epsilon 0.04) for the decorrelate arm; the naive-control arm trains
  only the 1500-episode post-intervention budget, in a pre-decorrelated,
  same-noise-level environment, matching this lineage's established and
  already-justified naive-control convention (post-intervention-episode
  matching is valid for DQN-style agents because `proxy_dependence` there is
  training-length-independent once converged, unlike PPO's entropy-collapse
  dynamic documented in `docs/cue_validity_ppo.md` -- not applicable here
  since no PPO condition is tested).
- `reward_noise` is set identically for both arms at a given cell and is
  never read, passed to, or otherwise accessible to either agent; this is
  checked directly in the implementation (the agent constructor call sites
  take no noise-derived argument).

## Metrics

`proxy_dependence` (primary), `cue_dependence_score` (secondary, reported
for continuity), both via the existing exact-probe method
(`probe_q_values`/`read_metrics`, unchanged from
`cue_validity_proxy_dependence_experiment.py`).

## Success / failure criteria (fixed before data collection)

Primary, pre-specified comparison: at `v=0.95`, is the decorrelate-vs-naive
Cohen's d larger at `reward_noise=0.6` than at `reward_noise=0.0`? This is
the direct operationalization of "uncertainty increases persistence,"
analogous in spirit to the pre-existing pipeline's own "high vs. baseline"
framing but now on a design without that pipeline's confound.

Secondary, full-family criterion: each of the 6 cells (3 noise levels x 2
validity levels) tested individually (`compare_groups`, p < 0.05,
decorrelate magnitude greater than naive), Holm-Bonferroni corrected across
all 6 cells together as one pre-registered family, following this project's
established within-experiment correction policy
(`docs/multiple_comparison_correction.md`). The `v=0.0` floor cells are
expected, and required for the result to be interpretable at all, to show
no effect at any noise level; if any floor cell shows a significant
effect, that invalidates the `v=0.95` result at the same noise level as
likely noise-variance inflation rather than genuine uncertainty-modulated
persistence, and this will be reported as such rather than explained away.

## What would change this plan after the fact

Nothing above will change after results are inspected, following the same
discipline as every prior pre-registration in this project.

## Result

**The primary comparison is not supported, and the direction is the
opposite of the central hypothesis.** Cohen's d at `v=0.95` decreases
monotonically as `reward_noise` increases: 23.59 (noise=0.0) to 16.24
(noise=0.3) to 11.88 (noise=0.6). All three levels remain highly
significant after Holm-Bonferroni correction (adjusted p < 0.0001 at every
level), so persistence itself is not eliminated by outcome uncertainty,
but its magnitude shrinks as uncertainty increases rather than growing.
The floor control (`v=0.0`) remains clean at every noise level (|d| < 0.3,
none significant), confirming this is not an artifact of noise inflating
`proxy_dependence`'s variance through the `abs()` transform; the shrinking
pattern at `v=0.95` is a real effect, not measurement noise.

**Mechanistic reading, offered as interpretation, not re-tested here**:
`reward_noise` makes the true causal relationship (hidden state to reward)
itself noisier and harder to learn confidently during acquisition, not
just the cue relationship. A linear agent that has learned a weaker,
less-confident causal Q-value distinction to begin with has less
cue-conditioned dependence to lose in the first place, net of whatever
uncertainty-driven increase in cue-reliance (if any) this design would
also predict. This is the opposite causal pathway from the one the
pre-existing confounded pipeline implicitly assumed (Section 2 of
`docs/symbolic_stabilization_audit.md`), where more "uncertainty" was
mechanically forced to translate into more cue-directed behavior by
construction, rather than being allowed to interact with how well the
causal relationship itself gets learned.

**This is the first valid test of RQ3 in this project's history, and its
answer, within this specific environment and this specific operationalization
of "uncertainty" as reward-delivery noise, is no: uncertainty does not
increase non-causal stabilization here, and the data are more consistent
with the opposite.** This should not be over-generalized: `reward_noise` is
one specific form of outcome uncertainty (noise on an already-correct
action's payoff), not uncertainty about the hidden state itself, the cue's
own reliability, or the environment's dynamics, and only `linear_raw` was
tested. Whether other forms of uncertainty (e.g. a noisier or
time-varying `cue_validity`, rather than noisier reward delivery) produce
a different pattern is a natural, separately-motivated follow-up, not
assumed to generalize from this result.
