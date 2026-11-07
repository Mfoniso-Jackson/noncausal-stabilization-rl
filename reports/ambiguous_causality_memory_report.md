# Ambiguous Causality Memory Experiment

## Research question
ambiguous_causality_report.md found no reliable dose-response between causal ambiguity and spurious-action reliance. Its own recommended next step was to add partial observability or memory constraints so the agent must infer causal structure from history, then repeat the manipulation. Does separating the spurious cue from the decision point by an explicit memory gap reveal or strengthen a dose-response that the single-step-ahead design could not?

## Why this is a genuinely new manipulation, not a replication
The original experiment showed the cue exactly one step before the decision and additionally compared a 'training_distribution' vs a 'decorrelated' evaluation mode. Because the cue bit is only ever encoded in the observation at that single probe phase and never at the decision phase itself, the decision-phase action (and therefore reward/goal_rate) cannot depend on which generative mode produced an episode -- both modes reduce to the same fixed, Q-table-determined choice. This experiment does not repeat that two-mode comparison; instead it manipulates something that genuinely changes the task: the number of pure, cue-free filler steps (`memory_gap`) inserted between the single cue exposure and the decision phase, which changes the temporal credit-assignment distance the SARSA(lambda) eligibility trace must bridge between seeing the cue and learning from the eventual reward.

## Environment design
Same latent-reward / spurious-cue causal structure as the original experiment (reward is sampled from a latent Bernoulli outcome with probability `reward_prob` and delivered only if the causal action is chosen at the decision phase; the cue is a single per-episode Bernoulli draw with probability `cue_given_reward` if the latent outcome is true and 0.1 otherwise, and it never affects transitions, reward, or action availability). The cue is shown once, at phase 3. The decision phase is pushed out to 3 + 1 + memory_gap, and every phase in between offers only the causal action with no cue information at all -- at memory_gap=0 the decision happens immediately after the cue (closest to the original design); larger memory_gap widens the gap the eligibility trace must cross.

## Design
The same 4 ambiguity-level conditions as the original experiment (reward_prob/cue_given_reward pairs) are crossed with memory_gap in [0, 2, 5, 8], 30 seeds each (480 total seed-condition runs). Agent and training hyperparameters are unchanged: SARSA(lambda), alpha=0.08, gamma=0.95, lambda=0.8, epsilon 1.0->0.04, 3000 training episodes, 300 evaluation episodes.

## Results table

| condition | gap | reward | goal_rate | SPI | spurious_rate | Q_advantage | spurious_vs_causal_Q |
|---|---:|---:|---:|---:|---:|---:|---:|
| clear_causality | 0 | 0.939 +/- 0.005 | 0.949 +/- 0.005 | 0.767 +/- 0.154 | 0.300 +/- 0.167 | 0.015 +/- 0.036 | 0.020 +/- 0.037 |
| clear_causality | 2 | 0.935 +/- 0.005 | 0.949 +/- 0.005 | 0.733 +/- 0.161 | 0.133 +/- 0.124 | -0.024 +/- 0.022 | -0.027 +/- 0.029 |
| clear_causality | 5 | 0.929 +/- 0.005 | 0.949 +/- 0.005 | 0.900 +/- 0.109 | 0.400 +/- 0.178 | 0.010 +/- 0.020 | 0.022 +/- 0.019 |
| clear_causality | 8 | 0.923 +/- 0.005 | 0.949 +/- 0.005 | 0.800 +/- 0.146 | 0.300 +/- 0.167 | 0.005 +/- 0.013 | 0.004 +/- 0.016 |
| moderate_ambiguity | 0 | 0.696 +/- 0.009 | 0.706 +/- 0.009 | 0.767 +/- 0.154 | 0.267 +/- 0.161 | 0.006 +/- 0.041 | 0.002 +/- 0.041 |
| moderate_ambiguity | 2 | 0.692 +/- 0.009 | 0.706 +/- 0.009 | 0.667 +/- 0.172 | 0.367 +/- 0.175 | 0.012 +/- 0.021 | -0.021 +/- 0.034 |
| moderate_ambiguity | 5 | 0.686 +/- 0.009 | 0.706 +/- 0.009 | 0.633 +/- 0.175 | 0.300 +/- 0.167 | 0.007 +/- 0.015 | -0.005 +/- 0.021 |
| moderate_ambiguity | 8 | 0.680 +/- 0.009 | 0.706 +/- 0.009 | 0.567 +/- 0.180 | 0.267 +/- 0.161 | 0.003 +/- 0.011 | -0.008 +/- 0.015 |
| high_ambiguity | 0 | 0.546 +/- 0.009 | 0.556 +/- 0.009 | 0.700 +/- 0.167 | 0.233 +/- 0.154 | -0.008 +/- 0.045 | -0.014 +/- 0.048 |
| high_ambiguity | 2 | 0.542 +/- 0.009 | 0.556 +/- 0.009 | 0.733 +/- 0.161 | 0.233 +/- 0.154 | -0.009 +/- 0.024 | -0.007 +/- 0.028 |
| high_ambiguity | 5 | 0.536 +/- 0.009 | 0.556 +/- 0.009 | 0.767 +/- 0.154 | 0.233 +/- 0.154 | 0.007 +/- 0.018 | 0.003 +/- 0.020 |
| high_ambiguity | 8 | 0.530 +/- 0.009 | 0.556 +/- 0.009 | 0.700 +/- 0.167 | 0.367 +/- 0.175 | 0.003 +/- 0.012 | -0.000 +/- 0.012 |
| extreme_ambiguity | 0 | 0.439 +/- 0.010 | 0.449 +/- 0.010 | 0.733 +/- 0.161 | 0.267 +/- 0.161 | -0.005 +/- 0.044 | -0.018 +/- 0.048 |
| extreme_ambiguity | 2 | 0.435 +/- 0.010 | 0.449 +/- 0.010 | 0.733 +/- 0.161 | 0.333 +/- 0.172 | 0.012 +/- 0.030 | 0.015 +/- 0.033 |
| extreme_ambiguity | 5 | 0.429 +/- 0.010 | 0.449 +/- 0.010 | 0.767 +/- 0.154 | 0.233 +/- 0.154 | 0.001 +/- 0.015 | 0.001 +/- 0.020 |
| extreme_ambiguity | 8 | 0.423 +/- 0.010 | 0.449 +/- 0.010 | 0.767 +/- 0.154 | 0.333 +/- 0.172 | 0.004 +/- 0.012 | 0.004 +/- 0.012 |

## Ambiguity-level dose-response, by memory gap

| memory gap | metric | r | p |
|---:|---|---:|---:|
| 0 | spurious_action_rate | -0.0337 | 0.7147 |
| 0 | spurious_Q_advantage | -0.0710 | 0.4408 |
| 0 | spurious_vs_causal_Q | -0.1184 | 0.1979 |
| 0 | SPI | -0.0426 | 0.6443 |
| 2 | spurious_action_rate | 0.1180 | 0.1993 |
| 2 | spurious_Q_advantage | 0.1395 | 0.1285 |
| 2 | spurious_vs_causal_Q | 0.1826 | 0.0459 |
| 2 | SPI | 0.0165 | 0.8577 |
| 5 | spurious_action_rate | -0.1394 | 0.1289 |
| 5 | spurious_Q_advantage | -0.0626 | 0.4971 |
| 5 | spurious_vs_causal_Q | -0.1105 | 0.2295 |
| 5 | SPI | -0.0705 | 0.4442 |
| 8 | spurious_action_rate | 0.0481 | 0.6021 |
| 8 | spurious_Q_advantage | -0.0146 | 0.8745 |
| 8 | spurious_vs_causal_Q | 0.0228 | 0.8048 |
| 8 | SPI | 0.0082 | 0.9292 |

## Memory-gap main effect

Pooled across all ambiguity levels, memory_gap vs spurious_action_rate: r = 0.0445, p = 0.3304.
Pooled across all ambiguity levels, memory_gap vs spurious_Q_advantage: r = 0.0240, p = 0.6004.
Within extreme_ambiguity only, memory_gap vs spurious_action_rate: r = 0.0227, p = 0.8058.
Within extreme_ambiguity only, memory_gap vs spurious_Q_advantage: r = 0.0171, p = 0.8530.

## Interpretation
At memory_gap=0, the ambiguity-level dose-response on spurious_action_rate is r=-0.0337, p=0.7147 (consistent with the original experiment's null finding at its single-step design). At the largest tested gap (8 filler steps), that same dose-response is r=0.0481, p=0.6021. The memory-gap main-effect tests above show whether widening the gap between cue exposure and the reward signal changes overall spurious reliance, independent of ambiguity level, and whether any such change concentrates in the condition where the cue is most predictive of the latent outcome (extreme_ambiguity).

## Does memory constraint reveal a dose-response that the original design missed?
Supported: False.

## Limitations
Memory_gap is implemented by withholding cue information from the observation during filler phases, not by giving the tabular agent a literal sliding window of raw history to integrate -- a recurrent or belief-state agent (as the hidden_state_superstition and causal_reversal experiments use) could in principle behave differently under the same manipulation. Correlation p-values use scipy's Pearson test on seed-level observations and do not correct for testing multiple metrics and gap values together.

## Next recommended experiment
Repeat this same memory-gap manipulation with a recurrent (GRU/LSTM) DQN agent instead of tabular SARSA(lambda), to test whether a function-approximation agent that must compress the gap into a learned hidden state shows a different relationship between credit-assignment distance and spurious-cue reliance than the tabular agent used here.
