# Ambiguous Causality Experiment

## Research question
Does superstition emerge when true causal control is unreliable and a non-causal cue/action is more statistically predictive of reward during training?

## Why this experiment follows from prior negative results
Distractor-count effects looked mechanical, temporal proximity alone failed, and spurious correlation alone did not produce reliable spurious preference. This experiment adds causal uncertainty: the true action is probabilistic while the spurious cue predicts the latent reward outcome during training.

## Environment design
Fixed-horizon tabular environment with reward phase 10. Action 0 is the only causal reward action. Action 1 is spurious. Actions 2 and 3 are neutral. The action space is fixed across conditions.

## Causal structure
Reward is sampled from a latent stochastic outcome and is delivered only if the agent selects the causal action at the reward phase. The spurious cue/action never changes reward probability, transitions, or action availability.

## Training/evaluation distribution shift
Training cue presence is correlated with the latent reward outcome. Evaluation has both training-distribution and decorrelated modes; decorrelated evaluation samples the cue independently of reward outcome.

## Agent design
Tabular SARSA(lambda), alpha=0.08, gamma=0.95, lambda=0.8, epsilon 1.0->0.04, 3000 episodes, 500 evaluation episodes, 30 seeds.

## Metrics
SPI is non-causal greedy action selection at the cue probe state. CDS is normalized Q-mass assigned to non-causal actions. Spurious_Q_advantage compares Q(spurious) to neutral actions. Spurious_vs_causal_Q compares Q(spurious) to Q(causal).

## Results table (decorrelated evaluation)

| condition | reward | goal_rate | SPI | spurious_rate | Q_advantage | spurious_vs_causal_Q |
|---|---:|---:|---:|---:|---:|---:|
| clear_causality | 0.927 +/- 0.003 | 0.949 +/- 0.003 | 0.533 +/- 0.182 | 0.200 +/- 0.146 | 0.011 +/- 0.031 | -0.034 +/- 0.045 |
| moderate_ambiguity | 0.680 +/- 0.007 | 0.702 +/- 0.007 | 0.633 +/- 0.175 | 0.200 +/- 0.146 | 0.002 +/- 0.036 | -0.039 +/- 0.044 |
| high_ambiguity | 0.530 +/- 0.007 | 0.552 +/- 0.007 | 0.733 +/- 0.161 | 0.167 +/- 0.136 | -0.027 +/- 0.032 | -0.028 +/- 0.035 |
| extreme_ambiguity | 0.425 +/- 0.007 | 0.447 +/- 0.007 | 0.667 +/- 0.172 | 0.300 +/- 0.167 | 0.017 +/- 0.038 | -0.010 +/- 0.049 |

## Trend analysis table

| eval mode | metric | slope | 95% CI | p approx |
|---|---|---:|---:|---:|
| training_distribution | spurious_action_rate | 0.0267 | +/- 0.0663 | 0.4306 |
| training_distribution | spurious_Q_advantage | -0.0012 | +/- 0.0155 | 0.8833 |
| training_distribution | spurious_vs_causal_Q | 0.0081 | +/- 0.0193 | 0.4114 |
| training_distribution | SPI | 0.0500 | +/- 0.0769 | 0.2023 |
| training_distribution | reward | -0.1654 | +/- 0.0064 | 0.0000 |
| training_distribution | goal_rate | -0.1654 | +/- 0.0064 | 0.0000 |
| decorrelated | spurious_action_rate | 0.0267 | +/- 0.0663 | 0.4306 |
| decorrelated | spurious_Q_advantage | -0.0012 | +/- 0.0155 | 0.8833 |
| decorrelated | spurious_vs_causal_Q | 0.0081 | +/- 0.0193 | 0.4114 |
| decorrelated | SPI | 0.0500 | +/- 0.0769 | 0.2023 |
| decorrelated | reward | -0.1654 | +/- 0.0064 | 0.0000 |
| decorrelated | goal_rate | -0.1654 | +/- 0.0064 | 0.0000 |

## Extreme-vs-clear comparison

| eval mode | metric | extreme - clear | 95% CI | p approx |
|---|---|---:|---:|---:|
| training_distribution | spurious_action_rate | 0.1000 | +/- 0.2214 | 0.3760 |
| training_distribution | spurious_Q_advantage | 0.0057 | +/- 0.0490 | 0.8191 |
| training_distribution | spurious_vs_causal_Q | 0.0231 | +/- 0.0662 | 0.4936 |
| training_distribution | SPI | 0.1333 | +/- 0.2498 | 0.2955 |
| training_distribution | reward | -0.5015 | +/- 0.0078 | 0.0000 |
| training_distribution | goal_rate | -0.5015 | +/- 0.0078 | 0.0000 |
| decorrelated | spurious_action_rate | 0.1000 | +/- 0.2214 | 0.3760 |
| decorrelated | spurious_Q_advantage | 0.0057 | +/- 0.0490 | 0.8191 |
| decorrelated | spurious_vs_causal_Q | 0.0231 | +/- 0.0662 | 0.4936 |
| decorrelated | SPI | 0.1333 | +/- 0.2498 | 0.2955 |
| decorrelated | reward | -0.5015 | +/- 0.0078 | 0.0000 |
| decorrelated | goal_rate | -0.5015 | +/- 0.0078 | 0.0000 |

## Random baseline comparison

| condition | metric | learned | random | learned - random |
|---|---|---:|---:|---:|
| clear_causality | reward | 0.927 | 0.220 | 0.707 |
| clear_causality | goal_rate | 0.949 | 0.242 | 0.707 |
| clear_causality | causal_target_rate | 1.000 | 0.254 | 0.746 |
| clear_causality | spurious_action_rate | 0.200 | 0.254 | -0.054 |
| moderate_ambiguity | reward | 0.680 | 0.157 | 0.522 |
| moderate_ambiguity | goal_rate | 0.702 | 0.179 | 0.522 |
| moderate_ambiguity | causal_target_rate | 1.000 | 0.254 | 0.746 |
| moderate_ambiguity | spurious_action_rate | 0.200 | 0.254 | -0.054 |
| high_ambiguity | reward | 0.530 | 0.119 | 0.411 |
| high_ambiguity | goal_rate | 0.552 | 0.141 | 0.411 |
| high_ambiguity | causal_target_rate | 1.000 | 0.254 | 0.746 |
| high_ambiguity | spurious_action_rate | 0.167 | 0.254 | -0.087 |
| extreme_ambiguity | reward | 0.425 | 0.093 | 0.332 |
| extreme_ambiguity | goal_rate | 0.447 | 0.115 | 0.332 |
| extreme_ambiguity | causal_target_rate | 1.000 | 0.254 | 0.746 |
| extreme_ambiguity | spurious_action_rate | 0.300 | 0.254 | 0.046 |

## Interpretation
Decorrelated extreme ambiguity reward=0.425; clear reward=0.927. Spurious action trend under decorrelation=0.0267. Spurious Q-advantage trend under decorrelation=-0.0012. Spurious-vs-causal Q trend under decorrelation=0.0081.

## Does this support computational superstition?
Supported: False.

## Limitations
The cue is predictive of latent reward outcome during training, so persistence under decorrelation is the critical test. If SPI rises without spurious Q advantage, the effect may reflect policy noise or action-space effects. If reward collapses, the agent is failing rather than superstitious.

## Recommended next experiment
Increase partial observability or add memory constraints so the agent must infer causal structure from history, then repeat the ambiguous-causality manipulation.
