# Causal Reversal + Function Approximation Experiment

## Research question
Does computational superstition emerge more strongly in agents that generalize, especially when a previously predictive proxy cue becomes anti-predictive or random?

## Why agent architecture is now the key variable
Earlier tabular experiments did not show robust action-level superstition. This experiment tests whether function approximation preserves proxy-cue dependence across reversal and extinction.

## Environment design
POMDP-style hidden-state environment. Reward can occur only at phase 10 when hidden_reward_state=1 and action 0 is selected.

## Phase design: acquisition, reversal, extinction
Acquisition uses P(proxy=1|hidden=1)=0.95 and P(proxy=1|hidden=0)=0.05. Reversal swaps those probabilities. Extinction sets both to 0.50.

## Agent descriptions
tabular_sarsa is tabular SARSA(lambda). dqn is the repository's dependency-free DQN-style linear function approximator with replay and target weights. PyTorch is unavailable in this runtime, so neural DQN and recurrent DQN were not executed; recurrent_dqn is scaffolded as a recommended follow-up.

## Metrics
Metrics include reward, goal_rate, SPI, CDS, action rates, Q values, proxy_Q_advantage, proxy_vs_causal_Q, proxy_dependence, proxy_action_dependence, adaptation half-lives, and superstition_persistence_index.

## Results tables

| agent | phase | reward | goal_rate | proxy_dep | proxy_action_rate | Q_adv | SPI |
|---|---|---:|---:|---:|---:|---:|---:|
| tabular_sarsa | acquisition | 0.475 +/- 0.032 | 0.497 +/- 0.032 | 0.032 +/- 0.123 | 0.220 +/- 0.043 | -0.006 +/- 0.004 | 0.786 +/- 0.059 |
| tabular_sarsa | reversal | 0.480 +/- 0.031 | 0.502 +/- 0.031 | -0.001 +/- 0.081 | 0.174 +/- 0.057 | -0.003 +/- 0.011 | 0.699 +/- 0.081 |
| tabular_sarsa | extinction | 0.475 +/- 0.018 | 0.497 +/- 0.018 | -0.092 +/- 0.151 | 0.198 +/- 0.035 | -0.006 +/- 0.009 | 0.710 +/- 0.063 |
| dqn | acquisition | 0.475 +/- 0.032 | 0.497 +/- 0.032 | 0.848 +/- 0.068 | 0.204 +/- 0.092 | 0.000 +/- 0.002 | 0.579 +/- 0.031 |
| dqn | reversal | 0.480 +/- 0.031 | 0.502 +/- 0.031 | -0.881 +/- 0.085 | 0.211 +/- 0.100 | 0.007 +/- 0.013 | 0.568 +/- 0.031 |
| dqn | extinction | 0.475 +/- 0.018 | 0.497 +/- 0.018 | -0.723 +/- 0.083 | 0.170 +/- 0.054 | -0.003 +/- 0.003 | 0.643 +/- 0.040 |

## Learning curve summary
Learning curves are saved as PNG files with vertical markers at episode 3000 and 4500.

## Persistence analysis

| agent | acq_dep_final | rev_dep_initial | rev_dep_final | ext_dep_initial | ext_dep_final | rev_half | ext_half | persistence_index |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tabular_sarsa | 0.032 +/- 0.123 | 0.103 +/- 0.123 | -0.001 +/- 0.081 | -0.006 +/- 0.102 | -0.092 +/- 0.151 | 333.3 | 442.9 | 3.489 +/- 3.374 |
| dqn | 0.848 +/- 0.068 | 0.850 +/- 0.067 | -0.881 +/- 0.085 | -0.874 +/- 0.085 | -0.723 +/- 0.083 | 510.0 | 800.0 | 0.788 +/- 0.069 |

## Tabular vs DQN vs recurrent comparison

| comparison | agent/contrast | metric | difference | 95% CI | p approx |
|---|---|---|---:|---:|---:|
| acquisition | tabular_sarsa | reward_final_vs_random_or_zero | 0.3892 | +/- 0.0319 | 0.0000 |
| acquisition | tabular_sarsa | goal_rate_final_vs_random_or_zero | 0.3892 | +/- 0.0319 | 0.0000 |
| acquisition | tabular_sarsa | proxy_dependence_final_vs_random_or_zero | 0.0321 | +/- 0.1231 | 0.6096 |
| acquisition | tabular_sarsa | proxy_action_rate_final_vs_random_or_zero | 0.2203 | +/- 0.0434 | 0.0000 |
| acquisition | tabular_sarsa | proxy_Q_advantage_final_vs_random_or_zero | -0.0063 | +/- 0.0044 | 0.0059 |
| acquisition | dqn | reward_final_vs_random_or_zero | 0.3892 | +/- 0.0319 | 0.0000 |
| acquisition | dqn | goal_rate_final_vs_random_or_zero | 0.3892 | +/- 0.0319 | 0.0000 |
| acquisition | dqn | proxy_dependence_final_vs_random_or_zero | 0.8478 | +/- 0.0682 | 0.0000 |
| acquisition | dqn | proxy_action_rate_final_vs_random_or_zero | 0.2037 | +/- 0.0922 | 0.0000 |
| acquisition | dqn | proxy_Q_advantage_final_vs_random_or_zero | 0.0004 | +/- 0.0021 | 0.6939 |
| reversal | tabular_sarsa | reward_final_vs_random_or_zero | 0.3633 | +/- 0.0313 | 0.0000 |
| reversal | tabular_sarsa | goal_rate_final_vs_random_or_zero | 0.3633 | +/- 0.0313 | 0.0000 |
| reversal | tabular_sarsa | proxy_dependence_final_vs_random_or_zero | -0.0014 | +/- 0.0806 | 0.9729 |
| reversal | tabular_sarsa | proxy_action_rate_final_vs_random_or_zero | 0.1739 | +/- 0.0569 | 0.0000 |
| reversal | tabular_sarsa | proxy_Q_advantage_final_vs_random_or_zero | -0.0035 | +/- 0.0106 | 0.5225 |
| reversal | dqn | reward_final_vs_random_or_zero | 0.3633 | +/- 0.0313 | 0.0000 |
| reversal | dqn | goal_rate_final_vs_random_or_zero | 0.3633 | +/- 0.0313 | 0.0000 |
| reversal | dqn | proxy_dependence_final_vs_random_or_zero | -0.8814 | +/- 0.0855 | 0.0000 |
| reversal | dqn | proxy_action_rate_final_vs_random_or_zero | 0.2113 | +/- 0.0999 | 0.0000 |
| reversal | dqn | proxy_Q_advantage_final_vs_random_or_zero | 0.0067 | +/- 0.0129 | 0.3082 |
| extinction | tabular_sarsa | reward_final_vs_random_or_zero | 0.3642 | +/- 0.0184 | 0.0000 |
| extinction | tabular_sarsa | goal_rate_final_vs_random_or_zero | 0.3642 | +/- 0.0184 | 0.0000 |
| extinction | tabular_sarsa | proxy_dependence_final_vs_random_or_zero | -0.0925 | +/- 0.1514 | 0.2313 |
| extinction | tabular_sarsa | proxy_action_rate_final_vs_random_or_zero | 0.1977 | +/- 0.0353 | 0.0000 |
| extinction | tabular_sarsa | proxy_Q_advantage_final_vs_random_or_zero | -0.0056 | +/- 0.0090 | 0.2250 |
| extinction | dqn | reward_final_vs_random_or_zero | 0.3642 | +/- 0.0184 | 0.0000 |
| extinction | dqn | goal_rate_final_vs_random_or_zero | 0.3642 | +/- 0.0184 | 0.0000 |
| extinction | dqn | proxy_dependence_final_vs_random_or_zero | -0.7230 | +/- 0.0834 | 0.0000 |
| extinction | dqn | proxy_action_rate_final_vs_random_or_zero | 0.1698 | +/- 0.0535 | 0.0000 |
| extinction | dqn | proxy_Q_advantage_final_vs_random_or_zero | -0.0034 | +/- 0.0029 | 0.0200 |
| reversal_final_change | tabular_sarsa | proxy_dependence | -0.0335 | +/- 0.1310 | 0.6163 |
| reversal_final_change | dqn | proxy_dependence | -1.7292 | +/- 0.1135 | 0.0000 |
| reversal_final_change | tabular_sarsa | proxy_action_rate | -0.0464 | +/- 0.0633 | 0.1509 |
| reversal_final_change | dqn | proxy_action_rate | 0.0076 | +/- 0.1367 | 0.9134 |
| reversal_final_change | tabular_sarsa | proxy_Q_advantage | 0.0028 | +/- 0.0129 | 0.6718 |
| reversal_final_change | dqn | proxy_Q_advantage | 0.0063 | +/- 0.0122 | 0.3146 |
| reversal_final_change | tabular_sarsa | reward | 0.0050 | +/- 0.0460 | 0.8311 |
| reversal_final_change | dqn | reward | 0.0050 | +/- 0.0460 | 0.8311 |
| reversal_final_change | tabular_sarsa | goal_rate | 0.0050 | +/- 0.0460 | 0.8311 |
| reversal_final_change | dqn | goal_rate | 0.0050 | +/- 0.0460 | 0.8311 |
| extinction_final_change | tabular_sarsa | proxy_dependence | -0.0911 | +/- 0.1536 | 0.2453 |
| extinction_final_change | dqn | proxy_dependence | 0.1584 | +/- 0.1068 | 0.0037 |
| extinction_final_change | tabular_sarsa | proxy_action_rate | 0.0238 | +/- 0.0625 | 0.4561 |
| extinction_final_change | dqn | proxy_action_rate | -0.0415 | +/- 0.0751 | 0.2790 |
| extinction_final_change | tabular_sarsa | proxy_Q_advantage | -0.0021 | +/- 0.0116 | 0.7218 |
| extinction_final_change | dqn | proxy_Q_advantage | -0.0102 | +/- 0.0129 | 0.1243 |
| extinction_final_change | tabular_sarsa | reward | -0.0050 | +/- 0.0373 | 0.7925 |
| extinction_final_change | dqn | reward | -0.0050 | +/- 0.0373 | 0.7925 |
| extinction_final_change | tabular_sarsa | goal_rate | -0.0050 | +/- 0.0373 | 0.7925 |
| extinction_final_change | dqn | goal_rate | -0.0050 | +/- 0.0373 | 0.7925 |
| agent_comparison | dqn_minus_tabular_sarsa | superstition_persistence_index | -2.7008 | +/- 3.3221 | 0.1111 |
| agent_comparison | dqn_minus_tabular_sarsa | reversal_adaptation_half_life | 188.8889 | +/- 154.7602 | 0.0167 |
| agent_comparison | dqn_minus_tabular_sarsa | extinction_half_life | 300.0000 | +/- 196.0000 | 0.0027 |

## Interpretation
DQN extinction proxy_dependence=-0.723; tabular extinction proxy_dependence=-0.092. DQN persistence index=0.788; tabular persistence index=3.489. The DQN-style agent did not preserve the original acquisition direction; it adapted to the reversed cue and then retained that reversed cue dependence during extinction.

## Does this support computational superstition?
Generalized proxy-reliance persistence supported: True.
Action-level superstition supported: False.

## Limitations
The local runtime does not have PyTorch, so the DQN result uses the repository's dependency-free linear DQN-style function approximator rather than a two-layer neural DQN. Recurrent DQN was not executed. Approximate p-values use normal approximations.

## Recommended next experiment
Install PyTorch and rerun this same design with a two-layer neural DQN and a short-history GRU/LSTM DQN, keeping the same phase schedule and metrics.

## Validity checks
Proxy action never directly causes reward. Reward only depends on hidden_reward_state and action 0. Phase transition proxy probabilities are acquisition 0.95/0.05, reversal 0.05/0.95, and extinction 0.50/0.50. Action-space size is fixed. Random baseline is included.

Recurrent DQN status: scaffolded_not_run_torch_unavailable.
