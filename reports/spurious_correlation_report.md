# Spurious Correlation Experiment

## Research question
Does an RL agent learn and preserve a non-causal action when that action is statistically associated with reward during training?

## Why this follows from previous experiments
The distractor-count experiment was likely confounded by action-space geometry, and temporal proximity alone did not produce a near-greater-than-far gradient. This experiment adds an explicit training correlation and then removes it during evaluation.

## Environment design
Fixed-horizon tabular environment. Action 0 is the only true causal reward action at phase 10. Action 1 is spurious and never rewards or changes reward availability. Actions 2 and 3 are neutral distractors. The action-space size is fixed across all conditions.

## Training/evaluation distribution shift
During training, action 1 is available at phase 9 with condition-specific probability. During evaluation, action 1 availability is decorrelated and sampled independently at each pre-reward phase with probability 0.25.

## Agent design
Tabular SARSA(lambda), alpha=0.08, gamma=0.95, lambda=0.8, epsilon 1.0->0.04, 2500 training episodes, 300 evaluation episodes, 30 seeds.

## Metrics
CDS is normalized Q-mass assigned to non-causal actions. SPI is greedy non-causal action selection when the spurious action is available. Spurious Q advantage is Q(spurious)-mean Q(neutral). Spurious policy advantage is spurious action rate minus neutral action rate.

## Results table

| condition | reward | goal_rate | SPI | spurious_rate | Q_advantage | policy_advantage |
|---|---:|---:|---:|---:|---:|---:|
| no_spurious_correlation | 0.978 +/- 0.000 | 1.000 +/- 0.000 | 0.730 +/- 0.055 | 0.220 +/- 0.043 | -0.010 +/- 0.024 | -0.291 +/- 0.097 |
| weak_spurious_correlation | 0.978 +/- 0.000 | 1.000 +/- 0.000 | 0.748 +/- 0.050 | 0.238 +/- 0.051 | -0.004 +/- 0.022 | -0.272 +/- 0.094 |
| medium_spurious_correlation | 0.978 +/- 0.000 | 1.000 +/- 0.000 | 0.767 +/- 0.042 | 0.230 +/- 0.043 | -0.012 +/- 0.016 | -0.307 +/- 0.070 |
| strong_spurious_correlation | 0.978 +/- 0.000 | 1.000 +/- 0.000 | 0.794 +/- 0.059 | 0.239 +/- 0.042 | -0.016 +/- 0.019 | -0.317 +/- 0.092 |

## Trend table

| metric | slope | 95% CI | std effect | p approx |
|---|---:|---:|---:|---:|
| spurious_action_rate | 0.0219 | +/- 0.0741 | 0.053 | 0.5627 |
| spurious_Q_advantage | -0.0050 | +/- 0.0340 | -0.027 | 0.7732 |
| SPI | 0.0728 | +/- 0.0855 | 0.152 | 0.0949 |
| reward | -0.0000 | +/- 0.0000 | 0.000 | 1.0000 |
| goal_rate | 0.0000 | +/- 0.0000 | 0.000 | 1.0000 |

## Strong-vs-none comparison

| metric | strong - none | 95% CI | p approx |
|---|---:|---:|---:|
| spurious_action_rate | 0.0190 | +/- 0.0601 | 0.5355 |
| spurious_Q_advantage | -0.0054 | +/- 0.0305 | 0.7300 |
| SPI | 0.0641 | +/- 0.0804 | 0.1183 |
| reward | 0.0000 | +/- 0.0000 | 1.0000 |
| goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |

## Interpretation
Strong condition goal_rate=1.000. Spurious action slope=0.0219. Spurious Q-advantage slope=-0.0050.

## Does this support computational superstition?
Supported: False.

## Limitations
This design creates a distribution shift between training and evaluation, so persistent spurious action use could reflect learned superstition or imperfect adaptation to cue availability. However, the spurious action never directly causes reward and action-space size is fixed, so a positive trend would not be a simple action-count artifact.

## Recommended next experiment
Add a reversal phase where the spurious action remains available but is explicitly anti-correlated with reward, then measure extinction speed of spurious action selection.
