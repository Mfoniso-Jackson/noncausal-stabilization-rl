# Cue Validity PPO Experiment: Pre-Registration

Written and committed before any results from this experiment exist.
Tests H4 from the research directive this branch responds to ("different
reinforcement learning algorithms may exhibit different patterns of cue
dependence and adaptation") directly: `agents/ppo_agent.py` was a
non-functional stub before this session and has never been tested on
this repository's validated cue-validity pipeline. This experiment plugs
a real, tested PPO implementation (`docs`/commit: actor-critic, GAE,
clipped surrogate -- see `tests/ppo_agent_test.py`) into the exact same
environment, metric, and naive-control methodology already used for
`tabular`, `linear_raw`, and `neural_raw`.

## What this tests

Whether PPO -- an on-policy, actor-critic algorithm whose action
probabilities come directly from a policy network's logits, as opposed
to DQN-style off-policy Q-learning -- shows the same decorrelate-vs-naive
`proxy_dependence` persistence pattern already established for
`linear_raw`/`neural_raw` (both significant) and absent for `tabular`
(null). A null result for PPO would narrow H1's scope further (to
value-based function approximation specifically); a positive result
would extend it to on-policy actor-critic methods too.

## Design

- Condition: `ppo` only, `agents.ppo_agent.PPOAgent` over the same
  6-feature `observation_vector()` used by `linear_raw`/`neural_raw`.
- Environment: `env/cue_validity_env.py::CueValidityEnv`, `cue_phase =
  target_phase = 4`, matching every prior experiment in this lineage.
- `cue_validity` in {0.5, 0.95}, 15 seeds, matching the established
  convention.
- Episode budget matched exactly to the rest of this lineage: 3000
  acquisition episodes, 1500 post-intervention episodes, in 30 blocks of
  50 -- each block of 50 episodes is exactly one PPO rollout (buffer
  fills across 50 full episodes, each ending in a `done=True` transition
  so `last_value=0.0` is always the correct bootstrap) followed by one
  `update()` call, aligning PPO's natural update cadence with this
  lineage's existing block-based metric-reading cadence.
- PPO hyperparameters: defaults from `agents/ppo_agent.py`
  (`lr=3e-4, gamma=0.95, gae_lambda=0.95, clip_eps=0.2, epochs=4,
  minibatch_size=32, entropy_coef=0.01`), not tuned for this task
  specifically -- stated as a limitation, consistent with how this
  lineage has handled other agents' hyperparameters throughout.
- Arms: `decorrelate` (acquisition, then `env.decorrelate()`, continued
  training) and `naive` (fresh agent, decorrelated environment only, same
  post-intervention budget), the same naive-control methodology used
  throughout this lineage.

## Metrics

`proxy_dependence` (primary) and `CDS` (secondary), read via the same
exact-probe method (`probe_env.phase = target_phase`,
`probe_env.cue_present` fixed to 0/1), with
`PPOAgent.policy_logits_for_features()` supplying the "Q-values" input
to `metrics/cue_validity.py`'s existing functions -- those functions
apply softmax internally, and a policy network's logits are exactly the
pre-softmax quantity those functions expect, so no change to the shared
metrics module is needed.

## Success / failure criteria (fixed before data collection)

`ppo` shows **the same pattern as `linear_raw`/`neural_raw`** if its
decorrelate-vs-naive gap is significant (`compare_groups`, p < 0.05,
decorrelate magnitude greater than naive, Cohen's d reported) at at least
one validity level. It shows **the same pattern as `tabular`** (a null)
if neither validity level reaches significance. A result at only one of
the two validity levels is reported as a partial/qualified match to
whichever pattern it resembles, not forced into either camp.

## What would change this plan after the fact

Nothing above will change after results are inspected, following the
same discipline as every prior pre-registration in this lineage.
