# Cue Validity Optimizer Ablation: Pre-Registration

Written and committed before any results from this experiment exist.
Direct continuation of `docs/cue_validity_nonlinear.md` and its report;
read both first. This is one of the "recommended additions before
publication" named in the publication plan's Section 6, and the
publication plan's own Section 12 Go/No-Go criteria name this experiment
specifically: "abandon or substantially pivot if the optimizer ablation
shows the effect is actually about Adam's momentum rather than
representation sharing." This experiment is therefore load-bearing for
the paper's central claim, not a minor robustness check, and is held to
the same pre-registration discipline as every other experiment in this
lineage.

## What this tests

The nonlinear experiment found that `neural_raw` (and, more weakly,
`neural_interaction_only`, though that cell did not reach significance)
shows decorrelate-vs-naive persistence under a 2-hidden-layer MLP trained
with Adam. Adam's momentum (first-moment estimate) could itself be a
source of apparent "memory" independent of what is actually being
claimed -- that the persistence comes from the shared, parametric
representation retaining information about the old correlation. If
switching to a momentum-free optimizer (plain SGD) removes the effect,
the published claim that this is about representation sharing, not
optimizer dynamics, would be undermined and the paper's scope would need
to narrow or the finding would need to be abandoned, per the publication
plan's own stated criterion.

## Design

- Conditions: `neural_raw` and `neural_interaction_only` only (the two
  conditions from `experiments/cue_validity_nonlinear_experiment.py` that
  use `NeuralAgentWrapper`/`MLPQNet`; `linear_interaction_only` has no
  optimizer-momentum question since `DQNAgent`'s own update rule is
  unchanged).
- Intervention: `decorrelate` only (the core persistence result being
  stress-tested), not reversal -- this experiment is about the optimizer,
  not the intervention type.
- A new agent wrapper, `SGDNeuralAgentWrapper`, identical to
  `NeuralAgentWrapper` in every respect (same `MLPQNet` architecture,
  same replay buffer size, batch size, target-network update interval,
  train-every-steps, gradient clipping, epsilon schedule) except
  `torch.optim.SGD(..., momentum=0.0)` in place of
  `torch.optim.Adam(..., lr=1e-3)`, at the **same learning rate** (1e-3).
  The learning rate is deliberately not re-tuned for SGD: this isolates
  the presence/absence of momentum as the single changed variable, at the
  cost of not knowing whether a differently-tuned SGD would behave
  differently (stated as a limitation up front, not discovered after).
- `cue_validity` in {0.5, 0.95}, 15 seeds, same episode budget as the
  nonlinear experiment (3000 acquisition, 1500 post-intervention in 30
  blocks of 50).
- Same arms as the nonlinear experiment: `decorrelate` (pretrained agent,
  acquisition then decorrelate) and `naive` (fresh agent, decorrelated
  environment only), reusing
  `cue_validity_nonlinear_experiment.py::run_pretrained_arm` /
  `run_naive_arm` with the SGD wrapper substituted for the neural
  conditions' agent constructor.

## Metrics

Unchanged: `proxy_dependence` (primary) and `CDS` (secondary), read via
the same exact-probe method.

## Success / failure criteria (fixed before data collection)

**Primary**: for each of `neural_raw` and `neural_interaction_only`, at
each validity level, does the SGD-optimizer decorrelate-vs-naive gap show
a significant effect (`compare_groups`, independent two-sample t-test, p
< 0.05, decorrelate magnitude greater than naive, Cohen's d reported),
using the same test and threshold as the original Adam-optimizer result?

- If SGD **retains** a significant effect wherever Adam showed one
  (`neural_raw` at both levels; `neural_interaction_only` showed no
  significant Adam effect to begin with, so no effect under SGD there is
  expected and not informative either way): momentum is **not** the
  driver: the effect survives removing it, consistent with the
  representation-sharing account.
- If SGD **loses** a significant effect where Adam had one (specifically
  `neural_raw`): this is the pattern the publication plan's Go/No-Go
  criterion names as grounds to abandon or substantially pivot the
  nonlinear/neural portion of the claim, and will be reported as such,
  not minimized.

**Secondary** (reported for magnitude, not for the primary verdict): a
direct comparison of the per-seed decorrelate-vs-naive gap between the
Adam condition (already-committed, `results/cue_validity_nonlinear_summaries.csv`)
and the SGD condition (this experiment), via the same two-sample test, to
quantify how much of the effect's size (not just its significance)
depends on the optimizer.

A null result for `neural_interaction_only` under SGD is expected and not
treated as evidence either way, consistent with the nonlinear experiment's
own finding that this condition's Adam effect was already not significant.

## What would change this plan after the fact

Nothing in the metrics, conditions, or criteria above will change after
results are inspected, following the same discipline as every prior
pre-registration in this lineage. Given this experiment's stated
consequence (possible abandonment/pivot), this applies with particular
force here: a result unfavorable to the representation-sharing account
will be reported as the pre-registered criteria define it, not reframed
after the fact.
