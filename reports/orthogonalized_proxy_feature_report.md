# Orthogonalized Proxy Feature Control

## Research question
Does the linear DQN proxy-persistence effect require an easy raw proxy coordinate, or does it survive feature rotations and interaction-coded proxy information?

## Design
Same hidden-state reversal environment, same linear DQN, same phase schedule, same seeds, same metrics. Only the feature representation changes.

## Feature conditions
- phase_only: no proxy information.
- full_raw_proxy: original full feature vector with raw proxy coordinate.
- orthogonal_rotation_full: invertible linear rotation of the same full feature vector.
- centered_proxy_residual: proxy centered at zero while keeping phase features.
- proxy_interaction_only: no raw proxy coordinate; proxy appears only through phase interaction terms.

## Results table

| feature_condition | phase | reward | goal_rate | proxy_dep | abs_proxy_dep | proxy_action_rate | Q_adv |
|---|---|---:|---:|---:|---:|---:|---:|
| phase_only | acquisition | 0.476 +/- 0.021 | 0.498 +/- 0.021 | -0.003 +/- 0.004 | 0.005 +/- 0.002 | 0.180 +/- 0.123 | -0.000 +/- 0.001 |
| phase_only | reversal | 0.479 +/- 0.021 | 0.501 +/- 0.021 | -0.002 +/- 0.004 | 0.005 +/- 0.003 | 0.100 +/- 0.105 | -0.001 +/- 0.001 |
| phase_only | extinction | 0.482 +/- 0.018 | 0.504 +/- 0.018 | 0.008 +/- 0.014 | 0.019 +/- 0.008 | 0.230 +/- 0.134 | -0.000 +/- 0.001 |
| full_raw_proxy | acquisition | 0.476 +/- 0.021 | 0.498 +/- 0.021 | 0.850 +/- 0.067 | 0.850 +/- 0.067 | 0.200 +/- 0.090 | 0.000 +/- 0.002 |
| full_raw_proxy | reversal | 0.479 +/- 0.021 | 0.501 +/- 0.021 | -0.879 +/- 0.087 | 0.879 +/- 0.087 | 0.213 +/- 0.102 | 0.008 +/- 0.013 |
| full_raw_proxy | extinction | 0.482 +/- 0.018 | 0.504 +/- 0.018 | -0.718 +/- 0.086 | 0.718 +/- 0.086 | 0.171 +/- 0.054 | -0.003 +/- 0.003 |
| orthogonal_rotation_full | acquisition | 0.476 +/- 0.021 | 0.498 +/- 0.021 | 0.770 +/- 0.078 | 0.770 +/- 0.078 | 0.177 +/- 0.092 | -0.000 +/- 0.002 |
| orthogonal_rotation_full | reversal | 0.479 +/- 0.021 | 0.501 +/- 0.021 | -0.638 +/- 0.084 | 0.638 +/- 0.084 | 0.215 +/- 0.089 | -0.002 +/- 0.009 |
| orthogonal_rotation_full | extinction | 0.482 +/- 0.018 | 0.504 +/- 0.018 | -0.455 +/- 0.121 | 0.455 +/- 0.121 | 0.195 +/- 0.143 | -0.006 +/- 0.019 |
| centered_proxy_residual | acquisition | 0.476 +/- 0.021 | 0.498 +/- 0.021 | 0.919 +/- 0.058 | 0.919 +/- 0.058 | 0.171 +/- 0.082 | 0.000 +/- 0.002 |
| centered_proxy_residual | reversal | 0.479 +/- 0.021 | 0.501 +/- 0.021 | -0.990 +/- 0.020 | 0.990 +/- 0.020 | 0.225 +/- 0.081 | 0.003 +/- 0.013 |
| centered_proxy_residual | extinction | 0.482 +/- 0.018 | 0.504 +/- 0.018 | -0.569 +/- 0.139 | 0.569 +/- 0.139 | 0.235 +/- 0.098 | -0.002 +/- 0.009 |
| proxy_interaction_only | acquisition | 0.476 +/- 0.021 | 0.498 +/- 0.021 | 0.221 +/- 0.080 | 0.221 +/- 0.080 | 0.293 +/- 0.093 | 0.001 +/- 0.003 |
| proxy_interaction_only | reversal | 0.479 +/- 0.021 | 0.501 +/- 0.021 | -0.221 +/- 0.048 | 0.221 +/- 0.048 | 0.247 +/- 0.075 | 0.003 +/- 0.010 |
| proxy_interaction_only | extinction | 0.482 +/- 0.018 | 0.504 +/- 0.018 | -0.080 +/- 0.121 | 0.158 +/- 0.082 | 0.137 +/- 0.074 | -0.009 +/- 0.008 |

## Persistence summary

| feature_condition | acq_dep | rev_final | ext_final | ext_half | persistence_index |
|---|---:|---:|---:|---:|---:|
| phase_only | -0.003 +/- 0.004 | -0.002 +/- 0.004 | 0.008 +/- 0.014 | 330.0 | NA +/- NA |
| full_raw_proxy | 0.850 +/- 0.067 | -0.879 +/- 0.087 | -0.718 +/- 0.086 | 700.0 | 0.843 +/- 0.112 |
| orthogonal_rotation_full | 0.770 +/- 0.078 | -0.638 +/- 0.084 | -0.455 +/- 0.121 | 1200.0 | 0.739 +/- 0.134 |
| centered_proxy_residual | 0.919 +/- 0.058 | -0.990 +/- 0.020 | -0.569 +/- 0.139 | 514.3 | 0.746 +/- 0.050 |
| proxy_interaction_only | 0.221 +/- 0.080 | -0.221 +/- 0.048 | -0.080 +/- 0.121 | 850.0 | 1.267 +/- 0.439 |

## Comparison summary

| comparison | contrast | metric | difference | 95% CI | p approx |
|---|---|---|---:|---:|---:|
| acquisition | phase_only_vs_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | phase_only_vs_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | phase_only_vs_full_raw_proxy | abs_proxy_dependence | -0.8443 | +/- 0.0667 | 0.0000 |
| acquisition | phase_only_vs_full_raw_proxy | proxy_dependence | -0.8526 | +/- 0.0677 | 0.0000 |
| acquisition | phase_only_vs_full_raw_proxy | proxy_action_rate | -0.0204 | +/- 0.1529 | 0.7937 |
| acquisition | phase_only_vs_full_raw_proxy | proxy_Q_advantage | -0.0008 | +/- 0.0024 | 0.4903 |
| reversal | phase_only_vs_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | phase_only_vs_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | phase_only_vs_full_raw_proxy | abs_proxy_dependence | -0.8746 | +/- 0.0872 | 0.0000 |
| reversal | phase_only_vs_full_raw_proxy | proxy_dependence | 0.8775 | +/- 0.0864 | 0.0000 |
| reversal | phase_only_vs_full_raw_proxy | proxy_action_rate | -0.1127 | +/- 0.1390 | 0.1120 |
| reversal | phase_only_vs_full_raw_proxy | proxy_Q_advantage | -0.0084 | +/- 0.0127 | 0.1952 |
| extinction | phase_only_vs_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | phase_only_vs_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | phase_only_vs_full_raw_proxy | abs_proxy_dependence | -0.6998 | +/- 0.0834 | 0.0000 |
| extinction | phase_only_vs_full_raw_proxy | proxy_dependence | 0.7266 | +/- 0.0919 | 0.0000 |
| extinction | phase_only_vs_full_raw_proxy | proxy_action_rate | 0.0593 | +/- 0.1523 | 0.4452 |
| extinction | phase_only_vs_full_raw_proxy | proxy_Q_advantage | 0.0032 | +/- 0.0033 | 0.0562 |
| persistence | phase_only_vs_full_raw_proxy | extinction_half_life | -370.0000 | +/- 311.3461 | 0.0198 |
| persistence | phase_only_vs_full_raw_proxy | superstition_persistence_index | NA | +/- NA | NA |
| acquisition | orthogonal_rotation_full_vs_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | orthogonal_rotation_full_vs_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | orthogonal_rotation_full_vs_full_raw_proxy | abs_proxy_dependence | -0.0796 | +/- 0.0488 | 0.0014 |
| acquisition | orthogonal_rotation_full_vs_full_raw_proxy | proxy_dependence | -0.0796 | +/- 0.0488 | 0.0014 |
| acquisition | orthogonal_rotation_full_vs_full_raw_proxy | proxy_action_rate | -0.0237 | +/- 0.1351 | 0.7314 |
| acquisition | orthogonal_rotation_full_vs_full_raw_proxy | proxy_Q_advantage | -0.0008 | +/- 0.0023 | 0.4768 |
| reversal | orthogonal_rotation_full_vs_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | orthogonal_rotation_full_vs_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | orthogonal_rotation_full_vs_full_raw_proxy | abs_proxy_dependence | -0.2408 | +/- 0.0949 | 0.0000 |
| reversal | orthogonal_rotation_full_vs_full_raw_proxy | proxy_dependence | 0.2408 | +/- 0.0949 | 0.0000 |
| reversal | orthogonal_rotation_full_vs_full_raw_proxy | proxy_action_rate | 0.0022 | +/- 0.1097 | 0.9682 |
| reversal | orthogonal_rotation_full_vs_full_raw_proxy | proxy_Q_advantage | -0.0091 | +/- 0.0142 | 0.2091 |
| extinction | orthogonal_rotation_full_vs_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | orthogonal_rotation_full_vs_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | orthogonal_rotation_full_vs_full_raw_proxy | abs_proxy_dependence | -0.2632 | +/- 0.1253 | 0.0000 |
| extinction | orthogonal_rotation_full_vs_full_raw_proxy | proxy_dependence | 0.2632 | +/- 0.1253 | 0.0000 |
| extinction | orthogonal_rotation_full_vs_full_raw_proxy | proxy_action_rate | 0.0243 | +/- 0.1501 | 0.7511 |
| extinction | orthogonal_rotation_full_vs_full_raw_proxy | proxy_Q_advantage | -0.0031 | +/- 0.0194 | 0.7566 |
| persistence | orthogonal_rotation_full_vs_full_raw_proxy | extinction_half_life | 500.0000 | +/- 265.3853 | 0.0002 |
| persistence | orthogonal_rotation_full_vs_full_raw_proxy | superstition_persistence_index | -0.1039 | +/- 0.1745 | 0.2430 |
| acquisition | centered_proxy_residual_vs_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | centered_proxy_residual_vs_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | centered_proxy_residual_vs_full_raw_proxy | abs_proxy_dependence | 0.0691 | +/- 0.0296 | 0.0000 |
| acquisition | centered_proxy_residual_vs_full_raw_proxy | proxy_dependence | 0.0691 | +/- 0.0296 | 0.0000 |
| acquisition | centered_proxy_residual_vs_full_raw_proxy | proxy_action_rate | -0.0295 | +/- 0.0779 | 0.4581 |
| acquisition | centered_proxy_residual_vs_full_raw_proxy | proxy_Q_advantage | -0.0003 | +/- 0.0011 | 0.5961 |
| reversal | centered_proxy_residual_vs_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | centered_proxy_residual_vs_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | centered_proxy_residual_vs_full_raw_proxy | abs_proxy_dependence | 0.1108 | +/- 0.0806 | 0.0070 |
| reversal | centered_proxy_residual_vs_full_raw_proxy | proxy_dependence | -0.1108 | +/- 0.0806 | 0.0070 |
| reversal | centered_proxy_residual_vs_full_raw_proxy | proxy_action_rate | 0.0125 | +/- 0.0926 | 0.7920 |
| reversal | centered_proxy_residual_vs_full_raw_proxy | proxy_Q_advantage | -0.0046 | +/- 0.0066 | 0.1748 |
| extinction | centered_proxy_residual_vs_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | centered_proxy_residual_vs_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | centered_proxy_residual_vs_full_raw_proxy | abs_proxy_dependence | -0.1492 | +/- 0.1476 | 0.0475 |
| extinction | centered_proxy_residual_vs_full_raw_proxy | proxy_dependence | 0.1492 | +/- 0.1476 | 0.0475 |
| extinction | centered_proxy_residual_vs_full_raw_proxy | proxy_action_rate | 0.0644 | +/- 0.0890 | 0.1559 |
| extinction | centered_proxy_residual_vs_full_raw_proxy | proxy_Q_advantage | 0.0014 | +/- 0.0085 | 0.7535 |
| persistence | centered_proxy_residual_vs_full_raw_proxy | extinction_half_life | -185.7143 | +/- 296.1036 | 0.2190 |
| persistence | centered_proxy_residual_vs_full_raw_proxy | superstition_persistence_index | -0.0964 | +/- 0.1226 | 0.1231 |
| acquisition | proxy_interaction_only_vs_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | proxy_interaction_only_vs_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | proxy_interaction_only_vs_full_raw_proxy | abs_proxy_dependence | -0.6284 | +/- 0.0653 | 0.0000 |
| acquisition | proxy_interaction_only_vs_full_raw_proxy | proxy_dependence | -0.6284 | +/- 0.0653 | 0.0000 |
| acquisition | proxy_interaction_only_vs_full_raw_proxy | proxy_action_rate | 0.0929 | +/- 0.1326 | 0.1699 |
| acquisition | proxy_interaction_only_vs_full_raw_proxy | proxy_Q_advantage | 0.0007 | +/- 0.0042 | 0.7360 |
| reversal | proxy_interaction_only_vs_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | proxy_interaction_only_vs_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | proxy_interaction_only_vs_full_raw_proxy | abs_proxy_dependence | -0.6581 | +/- 0.0726 | 0.0000 |
| reversal | proxy_interaction_only_vs_full_raw_proxy | proxy_dependence | 0.6581 | +/- 0.0726 | 0.0000 |
| reversal | proxy_interaction_only_vs_full_raw_proxy | proxy_action_rate | 0.0342 | +/- 0.1285 | 0.6017 |
| reversal | proxy_interaction_only_vs_full_raw_proxy | proxy_Q_advantage | -0.0049 | +/- 0.0148 | 0.5120 |
| extinction | proxy_interaction_only_vs_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | proxy_interaction_only_vs_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | proxy_interaction_only_vs_full_raw_proxy | abs_proxy_dependence | -0.5604 | +/- 0.1263 | 0.0000 |
| extinction | proxy_interaction_only_vs_full_raw_proxy | proxy_dependence | 0.6385 | +/- 0.1282 | 0.0000 |
| extinction | proxy_interaction_only_vs_full_raw_proxy | proxy_action_rate | -0.0342 | +/- 0.0813 | 0.4099 |
| extinction | proxy_interaction_only_vs_full_raw_proxy | proxy_Q_advantage | -0.0053 | +/- 0.0079 | 0.1851 |
| persistence | proxy_interaction_only_vs_full_raw_proxy | extinction_half_life | 150.0000 | +/- 285.9033 | 0.3038 |
| persistence | proxy_interaction_only_vs_full_raw_proxy | superstition_persistence_index | 0.4244 | +/- 0.4531 | 0.0664 |
| acquisition_vs_random | phase_only | reward | 0.3727 | +/- 0.0212 | 0.0000 |
| acquisition_vs_random | phase_only | goal_rate | 0.3727 | +/- 0.0212 | 0.0000 |
| reversal_vs_random | phase_only | reward | 0.3697 | +/- 0.0215 | 0.0000 |
| reversal_vs_random | phase_only | goal_rate | 0.3697 | +/- 0.0215 | 0.0000 |
| extinction_vs_random | phase_only | reward | 0.3803 | +/- 0.0177 | 0.0000 |
| extinction_vs_random | phase_only | goal_rate | 0.3803 | +/- 0.0177 | 0.0000 |
| acquisition_vs_random | full_raw_proxy | reward | 0.3727 | +/- 0.0212 | 0.0000 |
| acquisition_vs_random | full_raw_proxy | goal_rate | 0.3727 | +/- 0.0212 | 0.0000 |
| reversal_vs_random | full_raw_proxy | reward | 0.3697 | +/- 0.0215 | 0.0000 |
| reversal_vs_random | full_raw_proxy | goal_rate | 0.3697 | +/- 0.0215 | 0.0000 |
| extinction_vs_random | full_raw_proxy | reward | 0.3803 | +/- 0.0177 | 0.0000 |
| extinction_vs_random | full_raw_proxy | goal_rate | 0.3803 | +/- 0.0177 | 0.0000 |
| acquisition_vs_random | orthogonal_rotation_full | reward | 0.3727 | +/- 0.0212 | 0.0000 |
| acquisition_vs_random | orthogonal_rotation_full | goal_rate | 0.3727 | +/- 0.0212 | 0.0000 |
| reversal_vs_random | orthogonal_rotation_full | reward | 0.3697 | +/- 0.0215 | 0.0000 |
| reversal_vs_random | orthogonal_rotation_full | goal_rate | 0.3697 | +/- 0.0215 | 0.0000 |
| extinction_vs_random | orthogonal_rotation_full | reward | 0.3803 | +/- 0.0177 | 0.0000 |
| extinction_vs_random | orthogonal_rotation_full | goal_rate | 0.3803 | +/- 0.0177 | 0.0000 |
| acquisition_vs_random | centered_proxy_residual | reward | 0.3727 | +/- 0.0212 | 0.0000 |
| acquisition_vs_random | centered_proxy_residual | goal_rate | 0.3727 | +/- 0.0212 | 0.0000 |
| reversal_vs_random | centered_proxy_residual | reward | 0.3697 | +/- 0.0215 | 0.0000 |
| reversal_vs_random | centered_proxy_residual | goal_rate | 0.3697 | +/- 0.0215 | 0.0000 |
| extinction_vs_random | centered_proxy_residual | reward | 0.3803 | +/- 0.0177 | 0.0000 |
| extinction_vs_random | centered_proxy_residual | goal_rate | 0.3803 | +/- 0.0177 | 0.0000 |
| acquisition_vs_random | proxy_interaction_only | reward | 0.3727 | +/- 0.0212 | 0.0000 |
| acquisition_vs_random | proxy_interaction_only | goal_rate | 0.3727 | +/- 0.0212 | 0.0000 |
| reversal_vs_random | proxy_interaction_only | reward | 0.3697 | +/- 0.0215 | 0.0000 |
| reversal_vs_random | proxy_interaction_only | goal_rate | 0.3697 | +/- 0.0215 | 0.0000 |
| extinction_vs_random | proxy_interaction_only | reward | 0.3803 | +/- 0.0177 | 0.0000 |
| extinction_vs_random | proxy_interaction_only | goal_rate | 0.3803 | +/- 0.0177 | 0.0000 |

## Interpretation
Extinction absolute proxy dependence: phase_only=0.019, raw=0.718, rotated=0.455, interaction_only=0.158.

## Does this support a coordinate artifact?
Raw coordinate-specific artifact supported: True.
Linear accessibility artifact supported: True.

If the orthogonal rotation preserves the effect, the result is not tied to one named input coordinate. If interaction-only coding weakens the effect, the result depends on how linearly accessible the proxy rule is.

## Validity checks
hidden_reward_state is never observed. Action 1 never directly causes reward. Reward only depends on hidden_reward_state and action 0. Phase probabilities and action-space size are fixed across all conditions. Random baseline is included.

## Recommended next experiment
Use an explicitly nonlinear function approximator on the interaction-only representation to test whether neural models can recover proxy reliance when linear DQN cannot.
