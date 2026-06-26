# Hidden-State Superstition Experiment

## Research question
Does superstition emerge when the true reward-generating state is hidden and the agent must rely on a non-causal proxy cue correlated with reward during training?

## Why partial observability is the next theoretical step
Prior experiments ruled out simple distractor count, temporal proximity, spurious correlation alone, and causal ambiguity alone. Hidden causal structure is a stronger test because the agent cannot directly observe the reward-generating state.

## Environment design
POMDP-style tabular environment. Hidden reward state is sampled each episode. Reward can occur at phase 10 only when hidden_reward_state=1 and action 0 is selected.

## Hidden causal structure
The agent observes phase and proxy cue, but not hidden_reward_state. Action 1 is a proxy/spurious action and never directly causes reward. Neutral actions never cause reward.

## Proxy-cue manipulation
Proxy strength varies from no_proxy to strong_proxy by changing P(proxy=1|hidden=1) and P(proxy=1|hidden=0). Decorrelated evaluation sets both probabilities to 0.50.

## Agent design
Tabular SARSA(lambda), alpha=0.08, gamma=0.95, lambda=0.8, epsilon 1.0->0.04, 4000 training episodes, 500 evaluation episodes, 30 seeds. Both memoryless and previous-observation memory agents were run.

## Metrics
Metrics include reward, goal_rate, SPI, CDS, action rates, Q advantages, proxy_dependence for causal action selection, and proxy_action_dependence for proxy action selection.

## Results tables

| condition | agent | mode | reward | goal_rate | proxy_rate | Q_adv | proxy_dep | proxy_action_dep |
|---|---|---|---:|---:|---:|---:|---:|---:|
| no_proxy | memoryless_agent | decorrelated | 0.480 +/- 0.007 | 0.502 +/- 0.007 | 0.239 +/- 0.035 | -0.001 +/- 0.005 | -0.009 +/- 0.052 | 0.023 +/- 0.061 |
| no_proxy | memory_agent | decorrelated | 0.480 +/- 0.007 | 0.502 +/- 0.007 | 0.254 +/- 0.023 | 0.001 +/- 0.003 | 0.037 +/- 0.049 | -0.018 +/- 0.054 |
| weak_proxy | memoryless_agent | decorrelated | 0.480 +/- 0.007 | 0.502 +/- 0.007 | 0.228 +/- 0.040 | -0.001 +/- 0.006 | -0.027 +/- 0.078 | -0.053 +/- 0.054 |
| weak_proxy | memory_agent | decorrelated | 0.480 +/- 0.007 | 0.502 +/- 0.007 | 0.260 +/- 0.027 | 0.003 +/- 0.004 | 0.011 +/- 0.046 | 0.010 +/- 0.035 |
| medium_proxy | memoryless_agent | decorrelated | 0.480 +/- 0.007 | 0.502 +/- 0.007 | 0.275 +/- 0.031 | 0.004 +/- 0.004 | 0.033 +/- 0.068 | -0.031 +/- 0.056 |
| medium_proxy | memory_agent | decorrelated | 0.480 +/- 0.007 | 0.502 +/- 0.007 | 0.257 +/- 0.030 | 0.001 +/- 0.004 | -0.000 +/- 0.068 | -0.016 +/- 0.044 |
| strong_proxy | memoryless_agent | decorrelated | 0.480 +/- 0.007 | 0.502 +/- 0.007 | 0.236 +/- 0.028 | 0.000 +/- 0.004 | 0.076 +/- 0.091 | -0.054 +/- 0.064 |
| strong_proxy | memory_agent | decorrelated | 0.405 +/- 0.015 | 0.427 +/- 0.015 | 0.270 +/- 0.021 | -0.001 +/- 0.006 | 0.016 +/- 0.047 | -0.011 +/- 0.053 |

## Trend analysis

| agent | mode | metric | slope | 95% CI | p approx |
|---|---|---|---:|---:|---:|
| memoryless_agent | training_distribution_eval | proxy_action_rate | 0.0125 | +/- 0.0509 | 0.6294 |
| memoryless_agent | training_distribution_eval | proxy_Q_advantage | 0.0032 | +/- 0.0073 | 0.3890 |
| memoryless_agent | training_distribution_eval | proxy_action_dependence | -0.0692 | +/- 0.0879 | 0.1229 |
| memoryless_agent | training_distribution_eval | SPI | 0.0161 | +/- 0.0555 | 0.5703 |
| memoryless_agent | training_distribution_eval | reward | 0.0000 | +/- 0.0100 | 1.0000 |
| memoryless_agent | training_distribution_eval | goal_rate | -0.0000 | +/- 0.0100 | 1.0000 |
| memoryless_agent | training_distribution_eval | proxy_dependence | 0.1062 | +/- 0.1093 | 0.0569 |
| memoryless_agent | decorrelated_eval | proxy_action_rate | 0.0119 | +/- 0.0509 | 0.6475 |
| memoryless_agent | decorrelated_eval | proxy_Q_advantage | 0.0030 | +/- 0.0073 | 0.4242 |
| memoryless_agent | decorrelated_eval | proxy_action_dependence | -0.0697 | +/- 0.0876 | 0.1190 |
| memoryless_agent | decorrelated_eval | SPI | 0.0168 | +/- 0.0554 | 0.5525 |
| memoryless_agent | decorrelated_eval | reward | 0.0000 | +/- 0.0100 | 1.0000 |
| memoryless_agent | decorrelated_eval | goal_rate | -0.0000 | +/- 0.0100 | 1.0000 |
| memoryless_agent | decorrelated_eval | proxy_dependence | 0.1052 | +/- 0.1094 | 0.0594 |
| memory_agent | training_distribution_eval | proxy_action_rate | 0.0184 | +/- 0.0408 | 0.3762 |
| memory_agent | training_distribution_eval | proxy_Q_advantage | -0.0009 | +/- 0.0054 | 0.7372 |
| memory_agent | training_distribution_eval | proxy_action_dependence | 0.0096 | +/- 0.0778 | 0.8096 |
| memory_agent | training_distribution_eval | SPI | -0.0032 | +/- 0.0397 | 0.8757 |
| memory_agent | training_distribution_eval | reward | -0.0009 | +/- 0.0100 | 0.8645 |
| memory_agent | training_distribution_eval | goal_rate | -0.0009 | +/- 0.0100 | 0.8645 |
| memory_agent | training_distribution_eval | proxy_dependence | 0.0120 | +/- 0.0839 | 0.7786 |
| memory_agent | decorrelated_eval | proxy_action_rate | 0.0148 | +/- 0.0375 | 0.4381 |
| memory_agent | decorrelated_eval | proxy_Q_advantage | -0.0025 | +/- 0.0064 | 0.4485 |
| memory_agent | decorrelated_eval | proxy_action_dependence | -0.0020 | +/- 0.0700 | 0.9546 |
| memory_agent | decorrelated_eval | SPI | 0.0059 | +/- 0.0350 | 0.7397 |
| memory_agent | decorrelated_eval | reward | -0.0754 | +/- 0.0179 | 0.0000 |
| memory_agent | decorrelated_eval | goal_rate | -0.0754 | +/- 0.0179 | 0.0000 |
| memory_agent | decorrelated_eval | proxy_dependence | -0.0243 | +/- 0.0789 | 0.5456 |

## Strong-vs-none comparison

| agent | mode | metric | strong - none | 95% CI | p approx |
|---|---|---|---:|---:|---:|
| memoryless_agent | training_distribution_eval | proxy_action_rate | -0.0029 | +/- 0.0442 | 0.8989 |
| memoryless_agent | training_distribution_eval | proxy_Q_advantage | 0.0015 | +/- 0.0064 | 0.6512 |
| memoryless_agent | training_distribution_eval | proxy_action_dependence | -0.0759 | +/- 0.0892 | 0.0955 |
| memoryless_agent | training_distribution_eval | SPI | -0.0006 | +/- 0.0470 | 0.9807 |
| memoryless_agent | training_distribution_eval | reward | 0.0000 | +/- 0.0095 | 1.0000 |
| memoryless_agent | training_distribution_eval | goal_rate | 0.0000 | +/- 0.0095 | 1.0000 |
| memoryless_agent | training_distribution_eval | proxy_dependence | 0.0858 | +/- 0.1057 | 0.1115 |
| memoryless_agent | decorrelated_eval | proxy_action_rate | -0.0037 | +/- 0.0447 | 0.8721 |
| memoryless_agent | decorrelated_eval | proxy_Q_advantage | 0.0013 | +/- 0.0065 | 0.6996 |
| memoryless_agent | decorrelated_eval | proxy_action_dependence | -0.0769 | +/- 0.0884 | 0.0883 |
| memoryless_agent | decorrelated_eval | SPI | 0.0000 | +/- 0.0469 | 0.9987 |
| memoryless_agent | decorrelated_eval | reward | 0.0000 | +/- 0.0095 | 1.0000 |
| memoryless_agent | decorrelated_eval | goal_rate | 0.0000 | +/- 0.0095 | 1.0000 |
| memoryless_agent | decorrelated_eval | proxy_dependence | 0.0851 | +/- 0.1052 | 0.1128 |
| memory_agent | training_distribution_eval | proxy_action_rate | 0.0202 | +/- 0.0346 | 0.2533 |
| memory_agent | training_distribution_eval | proxy_Q_advantage | -0.0003 | +/- 0.0047 | 0.9000 |
| memory_agent | training_distribution_eval | proxy_action_dependence | 0.0187 | +/- 0.0865 | 0.6714 |
| memory_agent | training_distribution_eval | SPI | 0.0006 | +/- 0.0408 | 0.9780 |
| memory_agent | training_distribution_eval | reward | -0.0009 | +/- 0.0095 | 0.8585 |
| memory_agent | training_distribution_eval | goal_rate | -0.0009 | +/- 0.0095 | 0.8585 |
| memory_agent | training_distribution_eval | proxy_dependence | 0.0159 | +/- 0.0741 | 0.6734 |
| memory_agent | decorrelated_eval | proxy_action_rate | 0.0160 | +/- 0.0311 | 0.3136 |
| memory_agent | decorrelated_eval | proxy_Q_advantage | -0.0019 | +/- 0.0067 | 0.5745 |
| memory_agent | decorrelated_eval | proxy_action_dependence | 0.0066 | +/- 0.0757 | 0.8637 |
| memory_agent | decorrelated_eval | SPI | 0.0091 | +/- 0.0330 | 0.5905 |
| memory_agent | decorrelated_eval | reward | -0.0754 | +/- 0.0165 | 0.0000 |
| memory_agent | decorrelated_eval | goal_rate | -0.0754 | +/- 0.0165 | 0.0000 |
| memory_agent | decorrelated_eval | proxy_dependence | -0.0205 | +/- 0.0677 | 0.5518 |

## Random baseline comparison

| condition | agent | metric | learned | random | learned-random |
|---|---|---|---:|---:|---:|
| no_proxy | memoryless_agent | reward | 0.480 | 0.104 | 0.376 |
| no_proxy | memoryless_agent | goal_rate | 0.502 | 0.126 | 0.376 |
| no_proxy | memoryless_agent | proxy_action_rate | 0.239 | 0.252 | -0.013 |
| no_proxy | memoryless_agent | causal_action_rate | 0.275 | 0.249 | 0.027 |
| no_proxy | memory_agent | reward | 0.480 | 0.104 | 0.376 |
| no_proxy | memory_agent | goal_rate | 0.502 | 0.126 | 0.376 |
| no_proxy | memory_agent | proxy_action_rate | 0.254 | 0.252 | 0.003 |
| no_proxy | memory_agent | causal_action_rate | 0.250 | 0.249 | 0.002 |
| weak_proxy | memoryless_agent | reward | 0.480 | 0.104 | 0.376 |
| weak_proxy | memoryless_agent | goal_rate | 0.502 | 0.126 | 0.376 |
| weak_proxy | memoryless_agent | proxy_action_rate | 0.228 | 0.252 | -0.024 |
| weak_proxy | memoryless_agent | causal_action_rate | 0.287 | 0.249 | 0.039 |
| weak_proxy | memory_agent | reward | 0.480 | 0.104 | 0.376 |
| weak_proxy | memory_agent | goal_rate | 0.502 | 0.126 | 0.376 |
| weak_proxy | memory_agent | proxy_action_rate | 0.260 | 0.252 | 0.009 |
| weak_proxy | memory_agent | causal_action_rate | 0.237 | 0.249 | -0.011 |
| medium_proxy | memoryless_agent | reward | 0.480 | 0.104 | 0.376 |
| medium_proxy | memoryless_agent | goal_rate | 0.502 | 0.126 | 0.376 |
| medium_proxy | memoryless_agent | proxy_action_rate | 0.275 | 0.252 | 0.023 |
| medium_proxy | memoryless_agent | causal_action_rate | 0.237 | 0.249 | -0.011 |
| medium_proxy | memory_agent | reward | 0.480 | 0.104 | 0.376 |
| medium_proxy | memory_agent | goal_rate | 0.502 | 0.126 | 0.376 |
| medium_proxy | memory_agent | proxy_action_rate | 0.257 | 0.252 | 0.005 |
| medium_proxy | memory_agent | causal_action_rate | 0.247 | 0.249 | -0.002 |
| strong_proxy | memoryless_agent | reward | 0.480 | 0.104 | 0.376 |
| strong_proxy | memoryless_agent | goal_rate | 0.502 | 0.126 | 0.376 |
| strong_proxy | memoryless_agent | proxy_action_rate | 0.236 | 0.252 | -0.016 |
| strong_proxy | memoryless_agent | causal_action_rate | 0.275 | 0.249 | 0.027 |
| strong_proxy | memory_agent | reward | 0.405 | 0.104 | 0.300 |
| strong_proxy | memory_agent | goal_rate | 0.427 | 0.126 | 0.300 |
| strong_proxy | memory_agent | proxy_action_rate | 0.270 | 0.252 | 0.018 |
| strong_proxy | memory_agent | causal_action_rate | 0.241 | 0.249 | -0.007 |

## Training vs decorrelated evaluation
Both evaluation modes are reported separately. Decorrelated evaluation is the primary persistence test because the proxy cue no longer predicts hidden_reward_state.

## Memoryless vs memory-agent comparison
Both agents use the same learning algorithm. The memory agent observes previous+current observation, allowing short-history dependence.

## Interpretation
Memory-agent strong_proxy decorrelated reward=0.405; goal_rate=0.427; proxy_action_rate=0.270; proxy_Q_advantage=-0.001.

## Does this support computational superstition?
Action-level superstition supported: False.
Proxy reliance without proxy-action superstition: False.

## Limitations
The proxy cue is observational, and tabular agents may learn to use the causal action under cue states rather than selecting the proxy action itself. If reward drops to random, the task is too hard; if proxy dependence rises without proxy action rate, the result is proxy reliance rather than full action superstition.

## Recommended next experiment
Use a recurrent or belief-state agent and add an extinction/reversal phase where the proxy cue becomes anti-predictive, then measure persistence and extinction speed.
