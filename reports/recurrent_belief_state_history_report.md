# Recurrent Belief-State History Experiment

## Research question
neural_recurrent_reversal_report.md found recurrent_dqn (GRU, 5-step history) was not clearly protective relative to the feedforward neural_dqn, and its own recommended next step was: if recurrent DQN reduces persistence, test longer histories and belief-state supervision to separate memory from false latent-state stabilization. Does giving the recurrent agent a longer observation history change how strongly it locks onto the proxy cue through acquisition, reversal, and extinction?

## Why this follows the neural/recurrent reversal experiment
That experiment fixed the recurrent agent's history window at 5 steps. This experiment varies only that window length (`history_length`), holding the environment, phase schedule, and all other agent hyperparameters fixed, to isolate the effect of memory capacity itself.

## Environment and phase schedule
Identical hidden-state POMDP and 3-phase schedule as the original experiment (acquisition proxy probabilities 0.95/0.05, reversal swaps them to 0.05/0.95, extinction decorrelates at 0.50/0.50), scaled down to 400/200/200 episodes per phase (vs. 3000/1500/1500 in the original) to keep 3 history lengths x 5 seeds of GRU training tractable here.

## Agent design
Recurrent DQN: a GRU(64) over the last `history_length` raw observation vectors, trained with the same replay buffer, target network, and optimizer settings as the original experiment's recurrent_dqn (lr=0.001, batch=64, gamma=0.95, target update every 200 steps). history_length in [1, 5, 15] (5 seeds each); history_length=5 is the original experiment's fixed setting, 1 removes effectively all temporal memory (each decision conditions only on the current observation), and 15 is well beyond the original window.

## Results table (final checkpoint of each phase)

| history_length | phase | reward | goal_rate | proxy_dependence | abs_proxy_dependence | proxy_Q_advantage |
|---:|---|---:|---:|---:|---:|---:|
| 1 | acquisition | 0.444 +/- 0.035 | 0.466 +/- 0.035 | 0.784 +/- 0.231 | 0.784 +/- 0.231 | 0.003 +/- 0.012 |
| 1 | reversal | 0.492 +/- 0.049 | 0.514 +/- 0.049 | 0.476 +/- 0.410 | 0.517 +/- 0.360 | -0.002 +/- 0.005 |
| 1 | extinction | 0.472 +/- 0.024 | 0.494 +/- 0.024 | -0.095 +/- 0.533 | 0.446 +/- 0.320 | -0.001 +/- 0.003 |
| 5 | acquisition | 0.444 +/- 0.035 | 0.466 +/- 0.035 | -0.018 +/- 0.194 | 0.171 +/- 0.099 | -0.003 +/- 0.024 |
| 5 | reversal | 0.492 +/- 0.049 | 0.514 +/- 0.049 | -0.104 +/- 0.263 | 0.206 +/- 0.198 | 0.002 +/- 0.005 |
| 5 | extinction | 0.472 +/- 0.024 | 0.494 +/- 0.024 | 0.164 +/- 0.335 | 0.284 +/- 0.246 | -0.002 +/- 0.003 |
| 15 | acquisition | 0.444 +/- 0.035 | 0.466 +/- 0.035 | -0.055 +/- 0.171 | 0.179 +/- 0.034 | -0.007 +/- 0.024 |
| 15 | reversal | 0.492 +/- 0.049 | 0.514 +/- 0.049 | -0.036 +/- 0.371 | 0.325 +/- 0.193 | 0.005 +/- 0.006 |
| 15 | extinction | 0.472 +/- 0.024 | 0.494 +/- 0.024 | 0.108 +/- 0.188 | 0.171 +/- 0.136 | -0.003 +/- 0.009 |

## Persistence index by history length

(superstition_persistence_index = |proxy_dependence| at the extinction midpoint checkpoint, divided by |proxy_dependence| at the acquisition-final checkpoint; NaN when the acquisition-final dependence was too small (<0.05) to normalize by.)

| history_length | acquisition dep (final) | extinction dep (mid) | extinction dep (final) | persistence index |
|---:|---:|---:|---:|---:|
| 1 | 0.784 +/- 0.231 | 0.096 +/- 0.374 | -0.095 +/- 0.533 | 0.555 +/- 0.230 |
| 5 | -0.018 +/- 0.194 | 0.162 +/- 0.191 | 0.164 +/- 0.335 | 0.466 +/- 0.483 |
| 15 | -0.055 +/- 0.171 | 0.178 +/- 0.222 | 0.108 +/- 0.188 | 1.319 +/- 1.041 |

## Interpretation
At history_length=1 (minimal memory), extinction abs_proxy_dependence = 0.446 and persistence index = 0.555. At history_length=15 (long memory), extinction abs_proxy_dependence = 0.171 and persistence index = 1.319. If longer history genuinely lets the agent separate a real latent-state belief from a merely-correlated recent observation, the long-history condition should show lower residual proxy dependence by the end of extinction than the minimal-memory condition. The two metrics disagree here, and the disagreement is itself informative: the raw, unnormalized abs_proxy_dependence at extinction is lower for the long-history agent, which on its own would favor the hypothesis, but the normalized persistence index is far larger and much noisier for history_length=15 because that condition's acquisition-final proxy_dependence sits just above the 0.05 stability threshold (0.055 in absolute value) -- dividing by a denominator that small inflates the ratio and its confidence interval, rather than reflecting a genuinely stronger acquired dependence to begin with. The raw abs_proxy_dependence comparison is the more trustworthy read of this run. That said, with only 5 seeds per condition the point-estimate gap alone is not a significance test: per-seed extinction-final abs_proxy_dependence for history_length=1 vs 15 gives t=1.543, p=0.1613, Cohen's d=1.091.

## Does longer history reduce false latent-state stabilization relative to minimal memory?
Supported (raw extinction abs_proxy_dependence lower AND p<0.05): False. Normalized persistence index agrees with the point-estimate direction: False.

## Limitations
Phase lengths, seed count (5), and evaluation episode count (100) are all reduced relative to the original neural_recurrent_reversal_experiment.py to keep 3 history-length conditions tractable on CPU-only PyTorch within this session; the persistence index here uses the extinction midpoint as a single residual checkpoint rather than an average over many early-extinction evaluations, so it is noisier than the original experiment's version. This experiment does not implement belief-state supervision (an auxiliary loss predicting hidden_reward_state), which the original report also recommended as a way to separate memory capacity from an explicit causal-inference objective; only the memory-capacity half of that recommendation is tested here.

## Next recommended experiment
Add an auxiliary belief-state supervision loss (predicting hidden_reward_state from the GRU's hidden state, trained alongside the Q-learning objective) to directly test whether explicit causal-structure supervision -- rather than memory capacity alone -- is what would be needed to prevent false latent-state stabilization.
