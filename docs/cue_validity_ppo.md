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

## Addendum (added after the run below; the sections above are unchanged)

**By the pre-registered criterion, PPO shows neither pattern cleanly, and
the report's own "matches tabular" framing oversimplifies the result.**
The decorrelate arm is not merely statistically indistinguishable from
naive (tabular's actual signature, p=0.90, d≈0) -- it is *significantly
smaller* than naive, in the opposite direction from the hypothesis
(p<0.0001 both levels, Cohen's d -2.28 at v=0.5, -1.84 at v=0.95). That is
a different, more specific result than a clean null, and the experiment's
own generated report (`reports/cue_validity_ppo_report.md`) labelling it
"PPO shows the tabular pattern: True" is misleading; this addendum
corrects that framing without editing the report file itself.

The most likely explanation, checked directly rather than assumed: PPO's
**acquisition-phase** `proxy_dependence` is tiny (mean 0.017 at v=0.5,
0.032 at v=0.95 -- roughly 6-10x smaller than `linear_raw`'s ~0.19-0.21 at
the same validity levels and episode budget). PPO, with these default,
untuned hyperparameters, essentially did not learn a strong
cue-conditioned policy during acquisition in the first place -- almost
certainly because on-policy PPO is more sample-inefficient per environment
step than off-policy DQN with a replay buffer, and 3000 acquisition
episodes (60 rollout-update cycles) was tuned for DQN's convergence speed,
not PPO's. With so little genuine dependence to begin with, the
significant decorrelate-vs-naive *difference* found is more consistent
with an artifact of unequal total training (the pretrained-then-decorrelated
agent gets 4500 total episodes against the naive agent's 1500, so it has
simply had more opportunity to converge toward a low-variance policy by
the time it is probed) than with evidence about persistence or its absence
either way.

**Net verdict: this experiment is inconclusive for H4, not a clean
negative.** It does not provide evidence that PPO lacks the
persistence phenomenon; it provides evidence that this specific,
untuned, budget-matched-to-DQN configuration did not learn the task
strongly enough during acquisition to test the question at all. A
properly powered test would need either a longer acquisition budget tuned
to PPO's own convergence speed (checked via acquisition-phase
`proxy_dependence` reaching a comparable magnitude to `linear_raw`'s
before declaring the intervention phase begun) or a hyperparameter
sweep, neither of which was done here. This is reported as the honest
outcome of the pre-registered test, not reframed as either a positive or
a negative finding for H4.
