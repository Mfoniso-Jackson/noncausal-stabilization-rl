# Non-Causal Behavioural Stabilization in RL

This project investigates whether reinforcement learning agents develop persistent dependencies on non-causal environmental cues under uncertainty, and whether such dependencies persist after intervention-based cue removal.

## Key Hypothesis

Increasing uncertainty leads to stable non-causal behavioural fixation in RL policies.

## Experiments

- Gridworld with non-causal cues
- PPO and DQN agents
- Uncertainty manipulation (sparsity, delay, noise)
- Intervention tests (cue removal, relocation, randomization)

## Metrics

- CDS (Cue Dependence Score)
- SPI (Persistence Index)
- Entropy
- Visitation bias

## Run

```bash
python experiments/train.py --config config.yaml
