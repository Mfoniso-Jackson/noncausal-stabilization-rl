# Cue Validity PPO Experiment

## Pre-registration
Design and criteria were written and committed in docs/cue_validity_ppo.md before any of the data below was collected.

## Design summary
ppo (agents.ppo_agent.PPOAgent), cue_validity in [0.5, 0.95], 15 seeds, decorrelate intervention. Same episode budget as the rest of this lineage (3000 acquisition, 1500 post-intervention in 30 blocks of 50); each block is one PPO rollout+update.

## Results

| validity | decorrelate | naive | t | p | Cohen's d | has effect |
|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 0.007 | 0.024 | -6.028 | 0.0000 | -2.278 | False |
| 0.95 | 0.010 | 0.024 | -4.856 | 0.0000 | -1.836 | False |

## Verdict (H4: do different algorithms show different patterns?)
PPO shows the linear_raw/neural_raw pattern (significant effect at >=1 validity level): False.
PPO shows the tabular pattern (null at both levels): True.

## Limitations
PPO's hyperparameters (lr, clip_eps, epochs, entropy_coef, etc.) are agents/ppo_agent.py's defaults, not tuned for this specific task -- a differently-tuned PPO could behave differently. Only one seed-per-rollout-shuffle ordering and one network architecture (1 hidden layer, 32 units) are tested.
