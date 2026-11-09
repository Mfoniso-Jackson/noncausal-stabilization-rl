# PPO Seed Top-Up (n=30): Total-Episode-Matched Comparison

## Pre-registration
Design and criteria were written and committed in docs/cue_validity_ppo.md ("Seed top-up") before this data was collected. 15 new seeds (15-29) added to both arms, combined with the original 15 (0-14) for n=30 per arm per validity level.

## Results

| validity | n | decorrelate | naive (total-matched) | t | p | Cohen's d | bootstrap diff CI | has effect |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.5 | 30 | 0.015 | 0.015 | 0.287 | 0.7749 | 0.075 | [-0.003, 0.004] | False |
| 0.95 | 30 | 0.017 | 0.015 | 1.162 | 0.2498 | 0.305 | [-0.001, 0.005] | False |

## Verdict (H4, n=30 seed top-up)
PPO shows the linear_raw/neural_raw pattern with the full n=30 seed count: False.
