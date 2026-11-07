# Cue Validity Replay Buffer Size Sweep

## Pre-registration
Design and criteria were written and committed in docs/cue_validity_replay_buffer_sweep.md before any of the data below was collected.

## Design summary
linear_raw only, replay_size in [500, 2000, 8000, 20000], cue_validity in [0.5, 0.95], 15 seeds, decorrelate intervention only.

## Results

| buffer size | validity | decorrelate | naive | t | p | Cohen's d | has effect |
|---:|---:|---:|---:|---:|---:|---:|---|
| 500 | 0.5 | 0.034 | 0.008 | 15.542 | 0.0000 | 5.874 | True |
| 500 | 0.95 | 0.058 | 0.008 | 29.076 | 0.0000 | 10.990 | True |
| 2000 | 0.5 | 0.038 | 0.007 | 18.459 | 0.0000 | 6.977 | True |
| 2000 | 0.95 | 0.066 | 0.007 | 35.125 | 0.0000 | 13.276 | True |
| 8000 | 0.5 | 0.062 | 0.009 | 32.225 | 0.0000 | 12.180 | True |
| 8000 | 0.95 | 0.110 | 0.009 | 62.412 | 0.0000 | 23.589 | True |
| 20000 | 0.5 | 0.084 | 0.009 | 40.969 | 0.0000 | 15.485 | True |
| 20000 | 0.95 | 0.151 | 0.009 | 93.355 | 0.0000 | 35.285 | True |

## Verdict
**Staleness exclusion holds across all tested buffer sizes: True.**
