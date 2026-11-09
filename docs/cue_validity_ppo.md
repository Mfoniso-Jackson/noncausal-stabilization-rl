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

## Correction (same day, before any further data collection): "longer budget" was the wrong fix, and here is why

A diagnostic run (not a new pre-registered experiment -- tracking
`proxy_dependence` and the new `policy_entropy` metric block-by-block
during acquisition only, no intervention) shows the paragraph above's
proposed fix was backwards: **training PPO longer does not recover the
effect, it destroys it further.** `proxy_dependence` does not monotonically
grow toward `linear_raw`'s ~0.19-0.21 with more episodes; it rises from
~0 to a peak of roughly 0.07-0.16 (varying by seed, peaking somewhere in
the 550-1050 episode range across validity 0.5 and 0.95, seeds 0-2 all
checked directly) and then **falls back toward 0** as training continues,
tracking `policy_entropy`'s own steady, continued decline toward full
convergence (by episode 1500, entropy is already down to 0.15-0.46 nats
from a maximum of `ln(4)=1.386`, and still falling).

The mechanism is now understood, not just observed: `ACTION_CAUSAL`
weakly dominates in this environment regardless of the cue (it costs
nothing extra and sometimes helps; no other action ever helps -- the same
structural fact noted throughout this lineage). A fully converged policy
should therefore assign `ACTION_CAUSAL` the same, near-certain probability
regardless of cue, which drives `proxy_dependence` (the difference in that
probability between cue=1 and cue=0) toward exactly 0 at convergence --
this is not PPO failing to learn, it is PPO succeeding at learning the
cue-independent optimal policy, which is itself a form of *not* being
cue-dependent. This is structurally different from `DQN`/`linear_raw`: 
`proxy_dependence` there is read from **Q-values**, a genuine regression
target that keeps tracking the cue-conditioned expected-return difference
even after the *greedy* policy has converged to the same action in both
cue conditions -- nothing in TD regression erases that difference once
learned. `PPO`'s actor logits are not a regression target; the policy-gradient
objective has no force keeping cue-conditioned logit differences once the
advantage of deviating from the dominant action is ~0 in both cue
conditions, so those differences are free to wash out as entropy
collapses. **This is a genuine, mechanistically grounded difference
between value-based and policy-gradient methods for this metric, not an
artifact of under-training.**

This also means a clean "does persistence survive decorrelation"
test cannot be built around a *converged* PPO baseline the way it was for
`linear_raw`/`neural_raw` -- there is no stable, asymptotic
pre-intervention `proxy_dependence` for PPO in this environment to test
persistence *of*. The properly-powered test below uses a **fixed**
acquisition budget chosen from this diagnostic *before* running the real
experiment (not tuned per-seed to each seed's own peak, which would be a
form of cherry-picking) and is explicit that this is an intermediate,
pre-convergence snapshot, not a stable equilibrium -- interpreted
accordingly, not claimed to carry the same evidentiary weight as the
`linear_raw`/`neural_raw` results.

## Properly-powered retest: pre-registered before running

**Design**: `ppo` only, same two validity levels {0.5, 0.95}, same 15
seeds, same `decorrelate`/`naive` methodology. **Acquisition budget fixed
at 800 episodes** (16 blocks of 50) -- chosen from the diagnostic above
(peaks observed at 550-1050 episodes across 4 seed/validity
combinations; 800 sits within that range and is fixed in advance of
running the 15-seed experiment, not selected afterward). Post-intervention
budget unchanged (1500 episodes, 30 blocks of 50). All other
hyperparameters unchanged (defaults).

**Success / failure criteria (fixed before data collection)**: identical
in form to the original experiment -- `ppo` matches the
`linear_raw`/`neural_raw` pattern if the decorrelate-vs-naive gap is
significant (p < 0.05, decorrelate magnitude greater than naive) at
either validity level; matches `tabular`'s pattern if neither validity
level is significant. **Explicitly anticipated as a plausible outcome,
not a failure of the test**: because both arms are measured at a
transient, not-yet-converged point in PPO's own training trajectory,
the naive arm (which only ever trains in the decorrelated environment)
may itself show a nonzero, *rising* `proxy_dependence` at this episode
count, for the same entropy-collapse-driven reasons as the pretrained
arm -- in which case the comparison is confounded by both arms moving
through their own version of the same inverted-U curve, and this will be
reported and diagnosed, not minimized.

### Result

**The anticipated confound is exactly what happened.** Acquisition-phase
`proxy_dependence` at episode 800 is now substantial (0.071 at v=0.5,
0.092 at v=0.95 -- within the diagnostic's expected range, no longer tiny
as in the first attempt). But `decorrelate` is still *smaller* than
`naive` (0.016 vs. 0.024 at v=0.5, p=0.044; 0.018 vs. 0.024 at v=0.95,
p=0.084), the same direction as before. A tell-tale internal-consistency
signature confirms the mechanism: `naive`'s final `proxy_dependence` and
`policy_entropy` are **identical to four decimal places across both
validity levels** (0.0234, 0.1588) -- exactly what the entropy-collapse
account predicts, since `decorrelate()` is applied before `naive` ever
trains, so its trajectory depends only on total episode count (1500),
never on the nominal validity parameter. The `decorrelate` arm, having had
2300 total episodes (800 acquisition + 1500 intervention) against
`naive`'s 1500, has simply progressed further along the *same* entropy-collapse
curve by the time both are probed -- mechanically producing a smaller
value, with nothing to do with decorrelation's effect specifically.

**This means the standard naive-control methodology itself -- matching
post-intervention episode count, not total episode count, which is
exactly right for `DQN`/`linear_raw`/`neural_raw` because their
`proxy_dependence` is training-length-independent once converged -- is
invalid for PPO without modification**, because PPO's readout is
training-length-dependent via entropy collapse regardless of any
intervention. A genuinely fair test needs to match **total** episode
count between arms, not just post-intervention count.

## Total-episode-matched retest: the methodologically correct comparison

**Design**: reuses the already-collected `decorrelate` arm data above
unchanged (2300 total episodes: 800 acquisition + 1500 intervention). Only
`naive` is rerun, now trained for the **same 2300 total episodes**,
entirely in the decorrelated environment (instead of 1500) -- the fair
comparison this algorithm actually needs. **Success criterion, fixed
before this rerun**: identical in form to every prior test in this
lineage -- significant if `decorrelate`'s final `|proxy_dependence|`
exceeds the total-matched `naive`'s, p < 0.05, at either validity level.

### Result

**The confound is resolved -- the comparison now points the right way,
but is underpowered at 15 seeds.** `decorrelate` (0.016, 0.018) is now
numerically *larger* than the total-matched `naive` (0.014, 0.014) at
both validity levels -- the direction the original hypothesis predicted,
and the opposite of every earlier PPO attempt in this document. Neither
level reaches significance (t=0.858, p=0.398, d=0.32 at v=0.5; t=1.422,
p=0.166, d=0.54 at v=0.95), and the v=0.95 bootstrap CI for the
difference, `[-0.001, 0.009]`, only barely spans zero.

**Final, complete answer for H4 and PPO, after three successive
corrections to this experiment's own methodology:** PPO shows a small,
directionally-consistent-with-persistence effect once (1) the acquisition
budget is calibrated to PPO's own entropy-collapse dynamics rather than
borrowed from DQN, and (2) total episode count is matched between
pretrained and naive arms rather than just post-intervention count --
but the effect size (Cohen's d ~0.3-0.5) is small enough that 15 seeds
cannot distinguish it from noise. This is reported as **underpowered,
not as a negative finding**: a larger seed count (the standard escalation
used elsewhere in this lineage, e.g. `docs/cue_validity_neural_reversal.md`'s
seed top-up) is the natural next step to resolve it, not undertaken here.
Whatever a future, better-powered rerun finds, the methodological story
told across this document stands on its own: PPO's `proxy_dependence`
readout behaves fundamentally differently from DQN's precisely because it
is read from policy logits under an objective with no force preserving
cue-conditioned information once the dominant action's advantage
saturates, rather than from a regression target that preserves it
indefinitely -- a genuine, mechanistically-grounded instance of H4
("different algorithms exhibit different patterns"), independent of
whether the specific persistence effect itself is ultimately confirmed or
not with more seeds.

## Seed top-up: result

**The effect did not hold up -- and it shrank rather than merely losing
significance.** With the full n=30 per arm per validity level (seeds
0-29), both validity levels are clearly non-significant (v=0.5: t=0.287,
p=0.775, Cohen's d=0.075; v=0.95: t=1.162, p=0.250, d=0.305), and the
bootstrap difference CIs are now close to symmetric around zero
(`[-0.003, 0.004]` at v=0.5, `[-0.001, 0.005]` at v=0.95) rather than
hugging the positive side as the n=15 CIs did. Critically, this is not
"still not significant, same effect size, now with a wider CI" -- the
point estimate for Cohen's d itself dropped from 0.32 to 0.075 at v=0.5
and from 0.54 to 0.305 at v=0.95. That pattern (effect size shrinking
substantially as sample size grows, not just losing significance) is the
signature of the original n=15 estimate having been inflated by small-sample
noise, not of a real, moderate effect that merely needed more power to
detect.

**Final, resolved answer for H4 and PPO**: after correcting all three
methodological confounds documented above (DQN-calibrated acquisition
budget, the undertraining misdiagnosis, the total-episode-count confound)
and running the standard seed top-up, PPO does **not** show a
decorrelate-vs-naive persistence effect distinguishable from noise. This
converges PPO with `tabular`'s null pattern at the level of the
raw statistical result -- but, as established above, via a completely
different mechanism (`tabular` never acquires cue-conditioned value in
the first place; PPO's actor-logit readout of a genuinely-acquired
cue-conditioned policy necessarily washes out as entropy collapses). This
is itself a clean instance of H4: two algorithm classes can converge on
the same observable null for mechanistically unrelated reasons, which a
single pattern-matching look at the final numbers alone would not
distinguish -- only the full methodological history in this document
does. This closes the PPO investigation; no further seed top-up is
planned without a new, separately-motivated reason to revisit it.

## Seed top-up: pre-registered before running

**Design**: adds 15 new seeds (15-29) to both arms of the total-episode-matched
comparison above, following the identical escalation convention already used
in this lineage (`docs/cue_validity_neural_reversal.md`'s `neural_raw` seed
top-up, `experiments/cue_validity_neural_raw_seed_topup.py`): `decorrelate`
(800 acquisition + 1500 intervention, 2300 total) and `naive_total_matched`
(2300 total, entirely decorrelated), identical protocol to
`experiments/cue_validity_ppo_total_matched_experiment.py` and
`experiments/cue_validity_ppo_retest_experiment.py`, combined with the
existing 15 seeds (0-14) for a final n=30 per arm per validity level.

**Success/failure criteria (fixed before data collection)**: unchanged in
form from the total-episode-matched retest above -- significant if
`decorrelate`'s final `|proxy_dependence|` exceeds `naive_total_matched`'s
at either validity level (`compare_groups`, p < 0.05, correct direction).
**Stated honestly in advance**: a power calculation for the observed
effect sizes (d=0.32 at v=0.5, d=0.54 at v=0.95) indicates that even n=30
per arm is unlikely to reach 80% power for the smaller effect (d=0.32
needs roughly n=155 per arm for 80% power at alpha=0.05; d=0.54 needs
roughly n=55). This top-up is run anyway, as the standard escalation step
used elsewhere in this lineage, and the v=0.95 result in particular is close
enough to adequately powered that a null result here is already informative
(it would mean the bootstrap CI's near-zero lower bound was the more
accurate signal). A result that remains non-significant at n=30 will be
reported as such, not used to justify a further, larger top-up without
separately re-registering that decision.
