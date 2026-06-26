# Neural and Recurrent Causal Reversal Experiment

## Research question
Does neural function approximation and recurrence increase persistent proxy reliance after a predictive cue becomes reversed and then random?

## Why this replicates the previous positive result
The previous dependency-free linear DQN showed strong proxy dependence through acquisition, reversal, and extinction. This replication keeps the same hidden-state environment and phase schedule while adding PyTorch neural and recurrent DQN agents.

## Environment design
Hidden-state POMDP. hidden_reward_state is sampled but never included in observation. Reward can occur only at phase 10 when hidden_reward_state=1 and action 0 is selected.

## Phase design
Acquisition uses 0.95/0.05 proxy probabilities. Reversal swaps them to 0.05/0.95. Extinction decorrelates the cue at 0.50/0.50.

## Agent architectures
tabular_sarsa uses SARSA(lambda). linear_dqn is the earlier dependency-free replay/target-weight baseline. neural_dqn is a PyTorch MLP with two 64-unit ReLU layers. recurrent_dqn is a PyTorch GRU over rolling observation histories of length 5.

## Training details
Seeds=10, acquisition=3000 episodes, reversal=1500, extinction=1500, evaluation every 100 episodes with 300 episodes. Neural agents use replay buffer size 50000, batch size 64, gamma=0.95, learning_rate=0.001, target update every 200 environment steps.

## Metrics
Metrics include reward, goal_rate, SPI, CDS, action rates, Q summaries, proxy_Q_advantage, proxy_vs_causal_Q, proxy_dependence, absolute proxy dependence, and proxy_action_dependence.

## Results tables

| agent | phase | reward | goal_rate | proxy_dep | abs_proxy_dep | proxy_action_rate | Q_adv |
|---|---|---:|---:|---:|---:|---:|---:|
| tabular_sarsa | acquisition | 0.476 +/- 0.021 | 0.498 +/- 0.021 | -0.009 +/- 0.132 | 0.172 +/- 0.071 | 0.253 +/- 0.054 | -0.001 +/- 0.007 |
| tabular_sarsa | reversal | 0.479 +/- 0.021 | 0.501 +/- 0.021 | 0.029 +/- 0.140 | 0.171 +/- 0.087 | 0.226 +/- 0.057 | -0.002 +/- 0.010 |
| tabular_sarsa | extinction | 0.482 +/- 0.018 | 0.504 +/- 0.018 | -0.117 +/- 0.110 | 0.161 +/- 0.083 | 0.251 +/- 0.054 | 0.002 +/- 0.006 |
| linear_dqn | acquisition | 0.476 +/- 0.021 | 0.498 +/- 0.021 | 0.850 +/- 0.067 | 0.850 +/- 0.067 | 0.200 +/- 0.090 | 0.000 +/- 0.002 |
| linear_dqn | reversal | 0.479 +/- 0.021 | 0.501 +/- 0.021 | -0.879 +/- 0.087 | 0.879 +/- 0.087 | 0.213 +/- 0.102 | 0.008 +/- 0.013 |
| linear_dqn | extinction | 0.482 +/- 0.018 | 0.504 +/- 0.018 | -0.718 +/- 0.086 | 0.718 +/- 0.086 | 0.171 +/- 0.054 | -0.003 +/- 0.003 |
| neural_dqn | acquisition | 0.476 +/- 0.021 | 0.498 +/- 0.021 | -0.029 +/- 0.162 | 0.189 +/- 0.106 | 0.273 +/- 0.208 | -0.002 +/- 0.011 |
| neural_dqn | reversal | 0.479 +/- 0.021 | 0.501 +/- 0.021 | -0.031 +/- 0.059 | 0.031 +/- 0.059 | 0.123 +/- 0.089 | -0.001 +/- 0.002 |
| neural_dqn | extinction | 0.482 +/- 0.018 | 0.504 +/- 0.018 | 0.124 +/- 0.147 | 0.124 +/- 0.147 | 0.231 +/- 0.133 | -0.000 +/- 0.002 |
| recurrent_dqn | acquisition | 0.476 +/- 0.021 | 0.498 +/- 0.021 | 0.234 +/- 0.264 | 0.350 +/- 0.202 | 0.269 +/- 0.140 | 0.001 +/- 0.006 |
| recurrent_dqn | reversal | 0.479 +/- 0.021 | 0.501 +/- 0.021 | 0.002 +/- 0.298 | 0.354 +/- 0.188 | 0.270 +/- 0.096 | 0.004 +/- 0.002 |
| recurrent_dqn | extinction | 0.482 +/- 0.018 | 0.504 +/- 0.018 | -0.012 +/- 0.260 | 0.287 +/- 0.181 | 0.230 +/- 0.148 | -0.000 +/- 0.004 |

## Learning curves summary
Saved plots include reward, signed proxy dependence, absolute proxy dependence, proxy action rate, and proxy Q advantage. Vertical markers show reversal and extinction boundaries.

## Persistence analysis
Persistence index is reported as NA when absolute acquisition-end proxy dependence is below 0.05, because the normalization denominator is too small for a meaningful ratio.

| agent | acq_dep | rev_initial | rev_final | ext_initial | ext_final | rev_half | ext_half | persistence_index |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tabular_sarsa | -0.009 +/- 0.132 | 0.018 +/- 0.136 | 0.029 +/- 0.140 | 0.028 +/- 0.136 | -0.117 +/- 0.110 | 240.1 | 433.3 | 1.363 +/- 0.621 |
| linear_dqn | 0.850 +/- 0.067 | 0.850 +/- 0.067 | -0.879 +/- 0.087 | -0.871 +/- 0.087 | -0.718 +/- 0.086 | 500.0 | 700.0 | 0.843 +/- 0.112 |
| neural_dqn | -0.029 +/- 0.162 | -0.019 +/- 0.148 | -0.031 +/- 0.059 | -0.020 +/- 0.049 | 0.124 +/- 0.147 | 120.1 | 80.0 | 0.396 +/- 0.106 |
| recurrent_dqn | 0.234 +/- 0.264 | 0.138 +/- 0.269 | 0.002 +/- 0.298 | 0.157 +/- 0.137 | -0.012 +/- 0.260 | 600.2 | 280.0 | 0.776 +/- 0.389 |

## Agent comparison

| comparison | contrast | metric | difference | 95% CI | p approx |
|---|---|---|---:|---:|---:|
| acquisition | neural_dqn_vs_tabular_sarsa | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | neural_dqn_vs_tabular_sarsa | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | neural_dqn_vs_tabular_sarsa | abs_proxy_dependence | 0.0174 | +/- 0.1438 | 0.8129 |
| acquisition | neural_dqn_vs_tabular_sarsa | proxy_dependence | -0.0192 | +/- 0.1496 | 0.8015 |
| acquisition | neural_dqn_vs_tabular_sarsa | proxy_action_rate | 0.0203 | +/- 0.1841 | 0.8286 |
| acquisition | neural_dqn_vs_tabular_sarsa | proxy_Q_advantage | -0.0014 | +/- 0.0099 | 0.7803 |
| reversal | neural_dqn_vs_tabular_sarsa | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | neural_dqn_vs_tabular_sarsa | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | neural_dqn_vs_tabular_sarsa | abs_proxy_dependence | -0.1400 | +/- 0.1139 | 0.0160 |
| reversal | neural_dqn_vs_tabular_sarsa | proxy_dependence | -0.0597 | +/- 0.1571 | 0.4564 |
| reversal | neural_dqn_vs_tabular_sarsa | proxy_action_rate | -0.1029 | +/- 0.1360 | 0.1383 |
| reversal | neural_dqn_vs_tabular_sarsa | proxy_Q_advantage | 0.0013 | +/- 0.0100 | 0.8009 |
| extinction | neural_dqn_vs_tabular_sarsa | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | neural_dqn_vs_tabular_sarsa | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | neural_dqn_vs_tabular_sarsa | abs_proxy_dependence | -0.0366 | +/- 0.1692 | 0.6713 |
| extinction | neural_dqn_vs_tabular_sarsa | proxy_dependence | 0.2411 | +/- 0.1959 | 0.0159 |
| extinction | neural_dqn_vs_tabular_sarsa | proxy_action_rate | -0.0205 | +/- 0.1258 | 0.7491 |
| extinction | neural_dqn_vs_tabular_sarsa | proxy_Q_advantage | -0.0017 | +/- 0.0062 | 0.5869 |
| persistence | neural_dqn_vs_tabular_sarsa | reversal_adaptation_half_life | -120.0000 | +/- 154.5077 | 0.1279 |
| persistence | neural_dqn_vs_tabular_sarsa | extinction_half_life | -353.3333 | +/- 188.9026 | 0.0002 |
| persistence | neural_dqn_vs_tabular_sarsa | superstition_persistence_index | -0.9671 | +/- 0.6297 | 0.0026 |
| acquisition | neural_dqn_vs_linear_dqn | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | neural_dqn_vs_linear_dqn | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | neural_dqn_vs_linear_dqn | abs_proxy_dependence | -0.6605 | +/- 0.1524 | 0.0000 |
| acquisition | neural_dqn_vs_linear_dqn | proxy_dependence | -0.8781 | +/- 0.1738 | 0.0000 |
| acquisition | neural_dqn_vs_linear_dqn | proxy_action_rate | 0.0724 | +/- 0.2653 | 0.5926 |
| acquisition | neural_dqn_vs_linear_dqn | proxy_Q_advantage | -0.0023 | +/- 0.0113 | 0.6846 |
| reversal | neural_dqn_vs_linear_dqn | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | neural_dqn_vs_linear_dqn | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | neural_dqn_vs_linear_dqn | abs_proxy_dependence | -0.8486 | +/- 0.1253 | 0.0000 |
| reversal | neural_dqn_vs_linear_dqn | proxy_dependence | 0.8486 | +/- 0.1253 | 0.0000 |
| reversal | neural_dqn_vs_linear_dqn | proxy_action_rate | -0.0900 | +/- 0.0428 | 0.0000 |
| reversal | neural_dqn_vs_linear_dqn | proxy_Q_advantage | -0.0081 | +/- 0.0139 | 0.2541 |
| extinction | neural_dqn_vs_linear_dqn | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | neural_dqn_vs_linear_dqn | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | neural_dqn_vs_linear_dqn | abs_proxy_dependence | -0.5941 | +/- 0.1377 | 0.0000 |
| extinction | neural_dqn_vs_linear_dqn | proxy_dependence | 0.8427 | +/- 0.1986 | 0.0000 |
| extinction | neural_dqn_vs_linear_dqn | proxy_action_rate | 0.0602 | +/- 0.1696 | 0.4866 |
| extinction | neural_dqn_vs_linear_dqn | proxy_Q_advantage | 0.0034 | +/- 0.0039 | 0.0844 |
| persistence | neural_dqn_vs_linear_dqn | reversal_adaptation_half_life | -379.9000 | +/- 104.4845 | 0.0000 |
| persistence | neural_dqn_vs_linear_dqn | extinction_half_life | -620.0000 | +/- 276.1058 | 0.0000 |
| persistence | neural_dqn_vs_linear_dqn | superstition_persistence_index | -0.4465 | +/- 0.1542 | 0.0000 |
| acquisition | recurrent_dqn_vs_neural_dqn | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | recurrent_dqn_vs_neural_dqn | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| acquisition | recurrent_dqn_vs_neural_dqn | abs_proxy_dependence | 0.1610 | +/- 0.1839 | 0.0863 |
| acquisition | recurrent_dqn_vs_neural_dqn | proxy_dependence | 0.2624 | +/- 0.3949 | 0.1928 |
| acquisition | recurrent_dqn_vs_neural_dqn | proxy_action_rate | -0.0034 | +/- 0.1513 | 0.9652 |
| acquisition | recurrent_dqn_vs_neural_dqn | proxy_Q_advantage | 0.0026 | +/- 0.0079 | 0.5117 |
| reversal | recurrent_dqn_vs_neural_dqn | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | recurrent_dqn_vs_neural_dqn | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| reversal | recurrent_dqn_vs_neural_dqn | abs_proxy_dependence | 0.3240 | +/- 0.2063 | 0.0021 |
| reversal | recurrent_dqn_vs_neural_dqn | proxy_dependence | 0.0328 | +/- 0.3125 | 0.8368 |
| reversal | recurrent_dqn_vs_neural_dqn | proxy_action_rate | 0.1476 | +/- 0.0966 | 0.0027 |
| reversal | recurrent_dqn_vs_neural_dqn | proxy_Q_advantage | 0.0044 | +/- 0.0029 | 0.0030 |
| extinction | recurrent_dqn_vs_neural_dqn | reward | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | recurrent_dqn_vs_neural_dqn | goal_rate | 0.0000 | +/- 0.0000 | 1.0000 |
| extinction | recurrent_dqn_vs_neural_dqn | abs_proxy_dependence | 0.1622 | +/- 0.1749 | 0.0692 |
| extinction | recurrent_dqn_vs_neural_dqn | proxy_dependence | -0.1361 | +/- 0.1855 | 0.1503 |
| extinction | recurrent_dqn_vs_neural_dqn | proxy_action_rate | -0.0012 | +/- 0.1227 | 0.9847 |
| extinction | recurrent_dqn_vs_neural_dqn | proxy_Q_advantage | -0.0003 | +/- 0.0042 | 0.9055 |
| persistence | recurrent_dqn_vs_neural_dqn | reversal_adaptation_half_life | 480.1222 | +/- 403.6819 | 0.0197 |
| persistence | recurrent_dqn_vs_neural_dqn | extinction_half_life | 200.0000 | +/- 192.9270 | 0.0422 |
| persistence | recurrent_dqn_vs_neural_dqn | superstition_persistence_index | 0.3801 | +/- 0.4036 | 0.0649 |
| acquisition_vs_random | tabular_sarsa | reward | 0.3727 | +/- 0.0212 | 0.0000 |
| acquisition_vs_random | tabular_sarsa | goal_rate | 0.3727 | +/- 0.0212 | 0.0000 |
| reversal_vs_random | tabular_sarsa | reward | 0.3697 | +/- 0.0215 | 0.0000 |
| reversal_vs_random | tabular_sarsa | goal_rate | 0.3697 | +/- 0.0215 | 0.0000 |
| extinction_vs_random | tabular_sarsa | reward | 0.3803 | +/- 0.0177 | 0.0000 |
| extinction_vs_random | tabular_sarsa | goal_rate | 0.3803 | +/- 0.0177 | 0.0000 |
| acquisition_vs_random | linear_dqn | reward | 0.3727 | +/- 0.0212 | 0.0000 |
| acquisition_vs_random | linear_dqn | goal_rate | 0.3727 | +/- 0.0212 | 0.0000 |
| reversal_vs_random | linear_dqn | reward | 0.3697 | +/- 0.0215 | 0.0000 |
| reversal_vs_random | linear_dqn | goal_rate | 0.3697 | +/- 0.0215 | 0.0000 |
| extinction_vs_random | linear_dqn | reward | 0.3803 | +/- 0.0177 | 0.0000 |
| extinction_vs_random | linear_dqn | goal_rate | 0.3803 | +/- 0.0177 | 0.0000 |
| acquisition_vs_random | neural_dqn | reward | 0.3727 | +/- 0.0212 | 0.0000 |
| acquisition_vs_random | neural_dqn | goal_rate | 0.3727 | +/- 0.0212 | 0.0000 |
| reversal_vs_random | neural_dqn | reward | 0.3697 | +/- 0.0215 | 0.0000 |
| reversal_vs_random | neural_dqn | goal_rate | 0.3697 | +/- 0.0215 | 0.0000 |
| extinction_vs_random | neural_dqn | reward | 0.3803 | +/- 0.0177 | 0.0000 |
| extinction_vs_random | neural_dqn | goal_rate | 0.3803 | +/- 0.0177 | 0.0000 |
| acquisition_vs_random | recurrent_dqn | reward | 0.3727 | +/- 0.0212 | 0.0000 |
| acquisition_vs_random | recurrent_dqn | goal_rate | 0.3727 | +/- 0.0212 | 0.0000 |
| reversal_vs_random | recurrent_dqn | reward | 0.3697 | +/- 0.0215 | 0.0000 |
| reversal_vs_random | recurrent_dqn | goal_rate | 0.3697 | +/- 0.0215 | 0.0000 |
| extinction_vs_random | recurrent_dqn | reward | 0.3803 | +/- 0.0177 | 0.0000 |
| extinction_vs_random | recurrent_dqn | goal_rate | 0.3803 | +/- 0.0177 | 0.0000 |

## Interpretation
Extinction signed proxy dependence: tabular=-0.117, linear=-0.718, neural=0.124, recurrent=-0.012.
Extinction absolute proxy dependence: tabular=0.161, linear=0.718, neural=0.124, recurrent=0.287.

## Does this support computational superstition?
Generalized proxy-reliance persistence supported: False.
Action-level superstition supported: False.
Recurrent agent appears protective relative to neural DQN: False.

## Limitations
Approximate p-values use normal approximations. Recurrent replay uses rolling histories rather than full sequence replay with hidden-state burn-in. All neural runs are CPU-only in this workspace.

## Recommended next experiment
If proxy reliance persists, add a longer extinction window and an explicit anti-proxy control. If recurrent DQN reduces persistence, test longer histories and belief-state supervision to separate memory from false latent-state stabilization.

## Validity checks
PyTorch version: 2.2.2. CUDA available: False. MPS available: False. Hidden reward state is never included in observation. Proxy action never directly causes reward. Reward only depends on hidden_reward_state and action 0. Phase probabilities are fixed as specified. Action-space size is fixed at 4. Random baseline is included. All agents use the same phase schedule.
