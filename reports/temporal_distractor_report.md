# Temporal Distractor Experiment

## Research question
Was the previous distractor-count effect genuine temporal credit misassignment, or a mechanical artifact of enlarging the action space?

## Why this experiment is needed
The previous CDS values closely followed the proportion of distractor actions in the action set. This follow-up fixes the distractor types and asks whether temporal proximity to reward predicts superstitious selection and Q-value mass.

## Environment design
Tabular fixed-horizon environment with target reward at phase 10. Action 0 is the true causal action at the reward phase. Distractor actions never causally affect reward. In the temporally biased condition, near/mid/far distractors are available close to, several steps before, or far before reward respectively; neutral availability is random and unrelated to reward timing.

## Agent design
Tabular SARSA(lambda) with alpha=0.08, gamma=0.95, lambda=0.8, epsilon annealed from 1.0 to 0.04, trained for 2500 episodes.

## Metrics
Reward and goal_rate measure task learnability. CDS is total normalized distractor Q-mass when distractors are available. SPI is greedy distractor selection rate. Type-specific SPI and Q-mass compare near, mid, far, and neutral distractors.

## Results tables

| condition | reward | goal_rate | CDS | SPI | SPI_near | SPI_mid | SPI_far | Q_near | Q_mid | Q_far |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_distractors | 0.978 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| balanced_distractors | 0.978 +/- 0.000 | 1.000 +/- 0.000 | 0.800 +/- 0.000 | 0.203 +/- 0.010 | 0.170 | 0.200 | 0.197 | 0.200 | 0.200 | 0.200 |
| temporally_biased_distractors | 0.978 +/- 0.000 | 1.000 +/- 0.000 | 0.500 +/- 0.000 | 0.404 +/- 0.037 | 0.516 | 0.500 | 0.600 | 0.444 | 0.444 | 0.445 |

## Near/mid/far comparison

| condition | metric | near_minus_far | 95% CI | p approx |
|---|---|---:|---:|---:|
| no_distractors | SPI | 0.000 | +/- 0.000 | 1.0000 |
| no_distractors | Q_mass | 0.000 | +/- 0.000 | 1.0000 |
| balanced_distractors | SPI | -0.027 | +/- 0.059 | 0.3727 |
| balanced_distractors | Q_mass | -0.000 | +/- 0.000 | 0.1364 |
| temporally_biased_distractors | SPI | -0.084 | +/- 0.128 | 0.1997 |
| temporally_biased_distractors | Q_mass | -0.000 | +/- 0.002 | 0.7227 |

## Interpretation
In the temporally biased condition, SPI near-minus-far = -0.084; Q-mass near-minus-far = -0.000; goal_rate = 1.000.

## Genuine or mechanical?
Temporal proximity predicts superstitious action selection and Q-mass: False.
If true, this argues against a pure action-space-size artifact and supports temporal credit-assignment error as a contributor. If false, the earlier distractor-count effect is more likely mechanical or policy-tie geometry.

## Recommended next experiment
Hold temporal availability fixed and manipulate reward delay after near versus far distractor exposure. This would test whether the proximity gradient strengthens when credit assignment is explicitly made harder.
