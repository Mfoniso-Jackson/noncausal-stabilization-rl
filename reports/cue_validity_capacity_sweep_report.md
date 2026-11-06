# Cue Validity Capacity Sweep

## Pre-registration
Design and criteria were written and committed in docs/cue_validity_capacity_sweep.md before any of the data below was collected.

## Design summary
neural_raw only, hidden_size in [8, 16, 32, 64, 128], cue_validity in [0.5, 0.95], 15 seeds, decorrelate intervention only.

## Capacity curve

| width | validity | decorrelate | naive | t | p | Cohen's d | has effect |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 0.5 | 0.035 | 0.011 | 4.888 | 0.0000 | 1.847 | True |
| 8 | 0.95 | 0.054 | 0.011 | 5.902 | 0.0000 | 2.231 | True |
| 16 | 0.5 | 0.025 | 0.010 | 3.522 | 0.0015 | 1.331 | True |
| 16 | 0.95 | 0.030 | 0.010 | 5.263 | 0.0000 | 1.989 | True |
| 32 | 0.5 | 0.017 | 0.009 | 2.783 | 0.0095 | 1.052 | True |
| 32 | 0.95 | 0.026 | 0.009 | 4.293 | 0.0002 | 1.623 | True |
| 64 | 0.5 | 0.015 | 0.009 | 1.523 | 0.1389 | 0.576 | False |
| 64 | 0.95 | 0.020 | 0.009 | 2.600 | 0.0147 | 0.983 | True |
| 128 | 0.5 | 0.012 | 0.010 | 0.619 | 0.5406 | 0.234 | False |
| 128 | 0.95 | 0.017 | 0.010 | 1.655 | 0.1091 | 0.625 | False |

Cohen's d by width at validity=0.5: [(8, 1.847421), (16, 1.33123), (32, 1.051772), (64, 0.575736), (128, 0.234127)]. Roughly monotonic non-increasing (allowing noise): True.

Cohen's d by width at validity=0.95: [(8, 2.230562), (16, 1.989122), (32, 1.622643), (64, 0.982557), (128, 0.625447)]. Roughly monotonic non-increasing (allowing noise): True.
