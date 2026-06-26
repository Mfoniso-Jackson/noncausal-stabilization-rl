# Nonlinear Interaction Proxy Experiment

## Research question
Can a nonlinear neural DQN recover proxy reliance from an interaction-only representation where linear DQN shows weak persistence?

## Design
Same hidden-state reversal environment and phase schedule. The comparison crosses agent class (linear DQN vs neural DQN) with feature representation (raw proxy vs interaction-only proxy).

## Results table

| agent | feature | phase | reward | goal_rate | proxy_dep | abs_proxy_dep | proxy_action_rate | Q_adv |
|---|---|---|---:|---:|---:|---:|---:|---:|
| linear_dqn | full_raw_proxy | acquisition | 0.476 +/- 0.021 | 0.498 +/- 0.021 | 0.850 +/- 0.067 | 0.850 +/- 0.067 | 0.200 +/- 0.090 | 0.000 +/- 0.002 |
| linear_dqn | full_raw_proxy | reversal | 0.479 +/- 0.021 | 0.501 +/- 0.021 | -0.879 +/- 0.087 | 0.879 +/- 0.087 | 0.213 +/- 0.102 | 0.008 +/- 0.013 |
| linear_dqn | full_raw_proxy | extinction | 0.482 +/- 0.018 | 0.504 +/- 0.018 | -0.718 +/- 0.086 | 0.718 +/- 0.086 | 0.171 +/- 0.054 | -0.003 +/- 0.003 |
| linear_dqn | proxy_interaction_only | acquisition | 0.476 +/- 0.021 | 0.498 +/- 0.021 | 0.221 +/- 0.080 | 0.221 +/- 0.080 | 0.293 +/- 0.093 | 0.001 +/- 0.003 |
| linear_dqn | proxy_interaction_only | reversal | 0.479 +/- 0.021 | 0.501 +/- 0.021 | -0.221 +/- 0.048 | 0.221 +/- 0.048 | 0.247 +/- 0.075 | 0.003 +/- 0.010 |
| linear_dqn | proxy_interaction_only | extinction | 0.482 +/- 0.018 | 0.504 +/- 0.018 | -0.080 +/- 0.121 | 0.158 +/- 0.082 | 0.137 +/- 0.074 | -0.009 +/- 0.008 |
| neural_dqn | full_raw_proxy | acquisition | 0.476 +/- 0.021 | 0.498 +/- 0.021 | -0.029 +/- 0.162 | 0.189 +/- 0.106 | 0.273 +/- 0.208 | -0.002 +/- 0.011 |
| neural_dqn | full_raw_proxy | reversal | 0.479 +/- 0.021 | 0.501 +/- 0.021 | -0.031 +/- 0.059 | 0.031 +/- 0.059 | 0.123 +/- 0.089 | -0.001 +/- 0.002 |
| neural_dqn | full_raw_proxy | extinction | 0.482 +/- 0.018 | 0.504 +/- 0.018 | 0.124 +/- 0.147 | 0.124 +/- 0.147 | 0.231 +/- 0.133 | -0.000 +/- 0.002 |
| neural_dqn | proxy_interaction_only | acquisition | 0.476 +/- 0.021 | 0.498 +/- 0.021 | 0.001 +/- 0.135 | 0.142 +/- 0.098 | 0.244 +/- 0.169 | -0.002 +/- 0.010 |
| neural_dqn | proxy_interaction_only | reversal | 0.479 +/- 0.021 | 0.501 +/- 0.021 | 0.051 +/- 0.085 | 0.071 +/- 0.079 | 0.215 +/- 0.181 | -0.000 +/- 0.002 |
| neural_dqn | proxy_interaction_only | extinction | 0.482 +/- 0.018 | 0.504 +/- 0.018 | 0.067 +/- 0.083 | 0.070 +/- 0.082 | 0.284 +/- 0.154 | 0.000 +/- 0.002 |

## Persistence summary

| agent | feature | acq_dep | rev_final | ext_final | ext_half | persistence_index |
|---|---|---:|---:|---:|---:|---:|
| linear_dqn | full_raw_proxy | 0.850 +/- 0.067 | -0.879 +/- 0.087 | -0.718 +/- 0.086 | 700.0 | 0.843 +/- 0.112 |
| linear_dqn | proxy_interaction_only | 0.221 +/- 0.080 | -0.221 +/- 0.048 | -0.080 +/- 0.121 | 850.0 | 1.267 +/- 0.439 |
| neural_dqn | full_raw_proxy | -0.029 +/- 0.162 | -0.031 +/- 0.059 | 0.124 +/- 0.147 | 80.0 | 0.396 +/- 0.106 |
| neural_dqn | proxy_interaction_only | 0.001 +/- 0.135 | 0.051 +/- 0.085 | 0.067 +/- 0.083 | 60.0 | 0.388 +/- 0.236 |

## Comparison table

| comparison | contrast | metric | difference | 95% CI | p approx |
|---|---|---|---:|---:|---:|
| acquisition | linear_dqn_interaction_vs_raw | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | linear_dqn_interaction_vs_raw | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | linear_dqn_interaction_vs_raw | abs_proxy_dependence | -0.6284 | +/- 0.0653 | 0.0000 |
| acquisition | linear_dqn_interaction_vs_raw | proxy_dependence | -0.6284 | +/- 0.0653 | 0.0000 |
| acquisition | linear_dqn_interaction_vs_raw | proxy_action_rate | 0.0929 | +/- 0.1326 | 0.1699 |
| acquisition | linear_dqn_interaction_vs_raw | proxy_Q_advantage | 0.0007 | +/- 0.0042 | 0.7360 |
| reversal | linear_dqn_interaction_vs_raw | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | linear_dqn_interaction_vs_raw | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | linear_dqn_interaction_vs_raw | abs_proxy_dependence | -0.6581 | +/- 0.0726 | 0.0000 |
| reversal | linear_dqn_interaction_vs_raw | proxy_dependence | 0.6581 | +/- 0.0726 | 0.0000 |
| reversal | linear_dqn_interaction_vs_raw | proxy_action_rate | 0.0342 | +/- 0.1285 | 0.6017 |
| reversal | linear_dqn_interaction_vs_raw | proxy_Q_advantage | -0.0049 | +/- 0.0148 | 0.5120 |
| extinction | linear_dqn_interaction_vs_raw | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | linear_dqn_interaction_vs_raw | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | linear_dqn_interaction_vs_raw | abs_proxy_dependence | -0.5604 | +/- 0.1263 | 0.0000 |
| extinction | linear_dqn_interaction_vs_raw | proxy_dependence | 0.6385 | +/- 0.1282 | 0.0000 |
| extinction | linear_dqn_interaction_vs_raw | proxy_action_rate | -0.0342 | +/- 0.0813 | 0.4099 |
| extinction | linear_dqn_interaction_vs_raw | proxy_Q_advantage | -0.0053 | +/- 0.0079 | 0.1851 |
| acquisition | neural_dqn_interaction_vs_raw | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | neural_dqn_interaction_vs_raw | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | neural_dqn_interaction_vs_raw | abs_proxy_dependence | -0.0472 | +/- 0.1051 | 0.3792 |
| acquisition | neural_dqn_interaction_vs_raw | proxy_dependence | 0.0301 | +/- 0.2366 | 0.8030 |
| acquisition | neural_dqn_interaction_vs_raw | proxy_action_rate | -0.0290 | +/- 0.1354 | 0.6749 |
| acquisition | neural_dqn_interaction_vs_raw | proxy_Q_advantage | -0.0002 | +/- 0.0051 | 0.9261 |
| reversal | neural_dqn_interaction_vs_raw | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | neural_dqn_interaction_vs_raw | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | neural_dqn_interaction_vs_raw | abs_proxy_dependence | 0.0402 | +/- 0.0610 | 0.1963 |
| reversal | neural_dqn_interaction_vs_raw | proxy_dependence | 0.0813 | +/- 0.1306 | 0.2226 |
| reversal | neural_dqn_interaction_vs_raw | proxy_action_rate | 0.0928 | +/- 0.1800 | 0.3124 |
| reversal | neural_dqn_interaction_vs_raw | proxy_Q_advantage | 0.0003 | +/- 0.0028 | 0.8113 |
| extinction | neural_dqn_interaction_vs_raw | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | neural_dqn_interaction_vs_raw | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | neural_dqn_interaction_vs_raw | abs_proxy_dependence | -0.0544 | +/- 0.1785 | 0.5501 |
| extinction | neural_dqn_interaction_vs_raw | proxy_dependence | -0.0571 | +/- 0.1826 | 0.5402 |
| extinction | neural_dqn_interaction_vs_raw | proxy_action_rate | 0.0532 | +/- 0.1036 | 0.3140 |
| extinction | neural_dqn_interaction_vs_raw | proxy_Q_advantage | 0.0003 | +/- 0.0025 | 0.8006 |
| acquisition | neural_vs_linear_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | neural_vs_linear_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | neural_vs_linear_full_raw_proxy | abs_proxy_dependence | -0.6605 | +/- 0.1524 | 0.0000 |
| acquisition | neural_vs_linear_full_raw_proxy | proxy_dependence | -0.8781 | +/- 0.1738 | 0.0000 |
| acquisition | neural_vs_linear_full_raw_proxy | proxy_action_rate | 0.0724 | +/- 0.2653 | 0.5926 |
| acquisition | neural_vs_linear_full_raw_proxy | proxy_Q_advantage | -0.0023 | +/- 0.0113 | 0.6846 |
| reversal | neural_vs_linear_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | neural_vs_linear_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | neural_vs_linear_full_raw_proxy | abs_proxy_dependence | -0.8486 | +/- 0.1253 | 0.0000 |
| reversal | neural_vs_linear_full_raw_proxy | proxy_dependence | 0.8486 | +/- 0.1253 | 0.0000 |
| reversal | neural_vs_linear_full_raw_proxy | proxy_action_rate | -0.0900 | +/- 0.0428 | 0.0000 |
| reversal | neural_vs_linear_full_raw_proxy | proxy_Q_advantage | -0.0081 | +/- 0.0139 | 0.2541 |
| extinction | neural_vs_linear_full_raw_proxy | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | neural_vs_linear_full_raw_proxy | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | neural_vs_linear_full_raw_proxy | abs_proxy_dependence | -0.5941 | +/- 0.1377 | 0.0000 |
| extinction | neural_vs_linear_full_raw_proxy | proxy_dependence | 0.8427 | +/- 0.1986 | 0.0000 |
| extinction | neural_vs_linear_full_raw_proxy | proxy_action_rate | 0.0602 | +/- 0.1696 | 0.4866 |
| extinction | neural_vs_linear_full_raw_proxy | proxy_Q_advantage | 0.0034 | +/- 0.0039 | 0.0844 |
| acquisition | neural_vs_linear_proxy_interaction_only | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | neural_vs_linear_proxy_interaction_only | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | neural_vs_linear_proxy_interaction_only | abs_proxy_dependence | -0.0792 | +/- 0.1190 | 0.1920 |
| acquisition | neural_vs_linear_proxy_interaction_only | proxy_dependence | -0.2196 | +/- 0.1595 | 0.0069 |
| acquisition | neural_vs_linear_proxy_interaction_only | proxy_action_rate | -0.0494 | +/- 0.1920 | 0.6141 |
| acquisition | neural_vs_linear_proxy_interaction_only | proxy_Q_advantage | -0.0033 | +/- 0.0103 | 0.5288 |
| reversal | neural_vs_linear_proxy_interaction_only | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | neural_vs_linear_proxy_interaction_only | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | neural_vs_linear_proxy_interaction_only | abs_proxy_dependence | -0.1504 | +/- 0.0680 | 0.0000 |
| reversal | neural_vs_linear_proxy_interaction_only | proxy_dependence | 0.2719 | +/- 0.1169 | 0.0000 |
| reversal | neural_vs_linear_proxy_interaction_only | proxy_action_rate | -0.0315 | +/- 0.1435 | 0.6673 |
| reversal | neural_vs_linear_proxy_interaction_only | proxy_Q_advantage | -0.0028 | +/- 0.0097 | 0.5720 |
| extinction | neural_vs_linear_proxy_interaction_only | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | neural_vs_linear_proxy_interaction_only | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | neural_vs_linear_proxy_interaction_only | abs_proxy_dependence | -0.0881 | +/- 0.0968 | 0.0744 |
| extinction | neural_vs_linear_proxy_interaction_only | proxy_dependence | 0.1472 | +/- 0.0976 | 0.0031 |
| extinction | neural_vs_linear_proxy_interaction_only | proxy_action_rate | 0.1476 | +/- 0.1906 | 0.1290 |
| extinction | neural_vs_linear_proxy_interaction_only | proxy_Q_advantage | 0.0091 | +/- 0.0085 | 0.0364 |
| persistence | neural_vs_linear_full_raw_proxy | extinction_half_life | -620.0000 | +/- 276.1058 | 0.0000 |
| persistence | neural_vs_linear_full_raw_proxy | superstition_persistence_index | -0.4465 | +/- 0.1542 | 0.0000 |
| persistence | neural_vs_linear_proxy_interaction_only | extinction_half_life | -790.0000 | +/- 125.5012 | 0.0000 |
| persistence | neural_vs_linear_proxy_interaction_only | superstition_persistence_index | -0.8784 | +/- 0.4983 | 0.0006 |

## Interpretation
Extinction absolute proxy dependence: linear/raw=0.718, linear/interaction=0.158, neural/raw=0.124, neural/interaction=0.070.

## Does nonlinear approximation recover interaction-coded proxy reliance?
Neural interaction exceeds linear interaction by >0.20: False.
Neural interaction approaches linear raw proxy dependence: False.

If neural DQN still fails on interaction-only features, the positive linear effect is best treated as a brittle representation phenomenon rather than robust computational superstition.

## Validity checks
PyTorch version: 2.2.2. hidden_reward_state is never observed. Action 1 never directly causes reward. Reward only depends on hidden_reward_state and action 0. Action-space size and phase probabilities are unchanged.
