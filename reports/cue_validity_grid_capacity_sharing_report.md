# Grid Environment Capacity-Sharing vs. Step-Count Test

## Pre-registration
Design and criteria were written and committed in docs/cue_validity_grid_capacity_sharing.md before any of the data below was collected.

## Design summary
neural_raw, auto_navigate=True (step count held exactly constant vs. the already-null normal-navigation 1500-episode condition), cue_validity in [0.5, 0.95], 15 seeds, decorrelate intervention, original 1500-episode intervention budget.

## Results

| validity | decorrelate | naive | t | p | Cohen's d | has effect |
|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 0.008 | 0.008 | 0.187 | 0.8530 | 0.071 | False |
| 0.95 | 0.013 | 0.008 | 1.900 | 0.0678 | 0.718 | False |

## Verdict
**Capacity-sharing hypothesis supported: False.**
(Comparison: normal navigation at this same 1500-episode budget was null; the 800-episode step-matched normal-navigation condition showed a significant effect.)
