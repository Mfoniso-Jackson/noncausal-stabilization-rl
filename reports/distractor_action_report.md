# Distractor Action Experiment

## Research question
Does superstitious behavior increase as the number of non-causal but available actions increases?

## Hypothesis
Increasing distractor actions should increase credit diffusion onto non-causal actions and increase superstitious policy choices, while preserving task learnability.

## Environment description
Tabular fixed-horizon environment with horizon 20. The true causal action is action 0 and produces reward only at phase 10. Distractor actions are available at every phase but never causally produce reward. Pre-goal distractor choices do not prevent later goal completion, allowing credit diffusion without making the task impossible.

## Agent description
Tabular Q-learning with alpha=0.12, gamma=0.95, epsilon annealed from 1.0 to 0.05, trained for 2500 episodes per seed.

## Metrics
Reward is mean evaluation return over 100 greedy episodes. Goal rate is the fraction of evaluation episodes completing the causal reward action. CDS is the average softmax-normalized Q-value mass assigned to distractor actions across pre-goal phases. SPI is the fraction of greedy pre-goal decisions selecting a distractor action.

## Results table

| condition | reward | goal_rate | CDS | SPI |
|---|---:|---:|---:|---:|
| distractors_0 | 0.978 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| distractors_2 | 0.978 +/- 0.000 | 1.000 +/- 0.000 | 0.667 +/- 0.000 | 0.659 +/- 0.035 |
| distractors_5 | 0.978 +/- 0.000 | 1.000 +/- 0.000 | 0.833 +/- 0.000 | 0.833 +/- 0.043 |
| distractors_10 | 0.978 +/- 0.000 | 1.000 +/- 0.000 | 0.909 +/- 0.000 | 0.930 +/- 0.031 |

## Trend analysis table

| metric | slope | 95% CI | standardized effect | p approx |
|---|---:|---:|---:|---:|
| reward | 0.0000 | +/- 0.0000 | 0.000 | 1.0000 |
| goal_rate | 0.0000 | +/- 0.0000 | 0.000 | 1.0000 |
| CDS | 0.0767 | +/- 0.0102 | 0.806 | 0.0000 |
| SPI | 0.0791 | +/- 0.0108 | 0.798 | 0.0000 |

## Interpretation
CDS positive trend supported: True. SPI positive trend supported: True. Minimum goal rate across distractor conditions was 1.000.

## Credit Diffusion Hypothesis
Supported: True.

## Next recommended experiment
Add an intervention phase that removes distractor actions after training and measures whether policies continue to allocate action probability or value to now-unavailable distractor analogues. This would separate harmless exploratory rituals from persistent computational superstition.
