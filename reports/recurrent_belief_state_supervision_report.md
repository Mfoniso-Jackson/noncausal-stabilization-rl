# Recurrent Belief-State Supervision Experiment

## Research question
recurrent_belief_state_history_report.md found that longer observation history alone gave a large but not statistically significant reduction in post-extinction proxy dependence, and named belief-state supervision -- an auxiliary loss predicting the true hidden_reward_state from the agent's hidden state, trained alongside the Q objective -- as the remaining, untested half of the original recommendation. Does adding that explicit supervision signal (not more memory) reduce false latent-state stabilization?

## Why this isolates the supervision signal, not model capacity
Both conditions use the identical GRU(64) + two-head architecture (a Q-head and a belief_head predicting hidden_reward_state), the same history length (5 steps, the original experiment's default), and the same replay/target/optimizer settings. The only difference is whether the belief_head's prediction error (binary cross-entropy against the environment's true hidden_reward_state for that episode, never exposed in the observation) is added to the backward pass with weight 1.0. In the unsupervised condition the belief_head exists but never receives gradient, so it is dead weight with no effect on the shared GRU trunk -- its behavior should be statistically indistinguishable from a plain recurrent_dqn with no second head at all.

## Environment and phase schedule
Identical hidden-state POMDP and 3-phase schedule as recurrent_belief_state_history_experiment.py (acquisition proxy probabilities 0.95/0.05, reversal swaps them to 0.05/0.95, extinction decorrelates at 0.50/0.50), 400/200/200 episodes per phase, 8 seeds per condition.

## Metrics
Same proxy-dependence and persistence-index metrics as the history-length experiment, plus belief_accuracy: the fraction of evaluation episodes at the target phase where the belief_head's predicted probability, thresholded at 0.5, matches the true hidden_reward_state for that episode. belief_accuracy is a sanity check that the auxiliary task is actually being learned when supervised, not just a measure of downstream behavior.

## Results table (final checkpoint of each phase)

| supervised | phase | reward | goal_rate | proxy_dependence | abs_proxy_dependence | belief_accuracy |
|---|---|---:|---:|---:|---:|---:|
| False | acquisition | 0.474 +/- 0.038 | 0.496 +/- 0.038 | -0.075 +/- 0.156 | 0.187 +/- 0.091 | 0.599 +/- 0.147 |
| False | reversal | 0.490 +/- 0.037 | 0.512 +/- 0.037 | -0.114 +/- 0.159 | 0.178 +/- 0.123 | 0.456 +/- 0.105 |
| False | extinction | 0.484 +/- 0.023 | 0.506 +/- 0.023 | 0.091 +/- 0.215 | 0.199 +/- 0.170 | 0.511 +/- 0.021 |
| True | acquisition | 0.474 +/- 0.038 | 0.496 +/- 0.038 | 0.233 +/- 0.233 | 0.374 +/- 0.085 | 0.994 +/- 0.005 |
| True | reversal | 0.490 +/- 0.037 | 0.512 +/- 0.037 | -0.124 +/- 0.185 | 0.193 +/- 0.149 | 0.010 +/- 0.006 |
| True | extinction | 0.484 +/- 0.023 | 0.506 +/- 0.023 | -0.127 +/- 0.226 | 0.250 +/- 0.161 | 0.495 +/- 0.044 |

## Persistence index by condition

| supervised | acquisition dep (final) | extinction dep (mid) | extinction dep (final) | persistence index |
|---|---:|---:|---:|---:|
| False | -0.075 +/- 0.156 | 0.074 +/- 0.143 | 0.091 +/- 0.215 | 0.450 +/- 0.282 |
| True | 0.233 +/- 0.233 | -0.175 +/- 0.229 | -0.127 +/- 0.226 | 0.871 +/- 0.639 |

## Extinction comparison (per-seed abs_proxy_dependence, unsupervised vs supervised)

unsupervised mean = 0.199, supervised mean = 0.250, t = -0.422, p = 0.6793, Cohen's d = -0.226.

## Interpretation
The belief_head reaches 0.994 accuracy by the end of acquisition when supervised, confirming the auxiliary task is being learned rather than acting as an inert extra head. But the moment the phase schedule reverses the proxy cue's correlation with the truth, belief accuracy collapses to 0.010 -- not merely to chance (0.5), but to a value close to the reversed cue's own predictive floor (1 - cue_given_reward = 0.05 during reversal), and this is consistent across every one of the 8 seeds from the very first reversal evaluation (per-seed reversal belief_accuracy sits at 0.00-0.02 throughout the whole reversal phase; see the raw results CSV). The simplest explanation is that the belief_head did not learn to infer hidden_reward_state from temporal structure at all -- it learned the literal rule 'predict = proxy cue bit', which is highly accurate under the acquisition schedule and highly *wrong* under the reversed one, in lockstep with the cue's own flipped reliability. That is exactly the same proxy-cue shortcut the Q-network is suspected of using, now made directly visible in an explicitly supervised classifier: giving the network an auxiliary objective correlated with the hidden state did not make it learn anything different from the Q-objective alone, because the single most predictive feature available (the raw proxy bit) is a shortcut for both objectives at once. Extinction abs_proxy_dependence is 0.199 unsupervised vs 0.250 supervised; the significance test above is the primary basis for the behavioral verdict, not the point estimate alone, given the small seed count.

## Does belief-state supervision reduce false latent-state stabilization?
Supported: False. (Auxiliary task learned during acquisition, i.e. belief_accuracy > 0.6: True. Belief head tracks the proxy-cue shortcut rather than causal structure, i.e. reversal belief_accuracy < 0.1: True.)

## Limitations
Only 8 seeds per condition and a single aux-loss weight (1.0, untuned) are used given CPU-only PyTorch training cost in this session; a null or weak result here does not rule out a stronger effect at a different auxiliary-loss weight or with belief supervision applied at every timestep rather than only via the replay-buffer label. Phase lengths are the same reduced scale as recurrent_belief_state_history_experiment.py, not the original experiment's full schedule.

## Next recommended experiment
If belief-state supervision helps, sweep the auxiliary loss weight and test whether its benefit compounds with longer history (combining both manipulations from the original recommendation at once, which this experiment and the history-length experiment each tested only in isolation).
