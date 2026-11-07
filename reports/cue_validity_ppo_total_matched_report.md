# PPO Total-Episode-Matched Retest

## Pre-registration
Design and criteria were written and committed in docs/cue_validity_ppo.md ("Total-episode-matched retest") before this data was collected. Reuses the already-collected decorrelate arm (2300 total episodes); only naive is rerun, now also trained for 2300 total episodes entirely in the decorrelated environment.

## Results

| validity | decorrelate | naive (total-matched) | t | p | Cohen's d | bootstrap diff CI | has effect |
|---:|---:|---:|---:|---:|---:|---|---|
| 0.5 | 0.016 | 0.014 | 0.858 | 0.3980 | 0.324 | [-0.003, 0.008] | False |
| 0.95 | 0.018 | 0.014 | 1.422 | 0.1662 | 0.537 | [-0.001, 0.009] | False |

## Verdict (H4, total-episode-matched, methodologically corrected test)
PPO shows the linear_raw/neural_raw pattern once total episode count is properly matched: False.
