# neural_raw Extension to Cross-Environment Replication

## Pre-registration
Design and success criterion were written and committed in docs/cue_validity_cross_environment.md ("Extension: neural_raw in the grid environment") before this data was collected.

## Results

| validity | decorrelate | naive | t | p | Cohen's d | has effect |
|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 0.008 | 0.008 | 0.002 | 0.9986 | 0.001 | False |
| 0.95 | 0.010 | 0.008 | 0.992 | 0.3299 | 0.375 | False |

## Floor control (cue_validity=0.0)
decorrelate=0.006, naive=0.008, t=-1.363, p=0.1839, d=-0.515, no spurious effect: True.

## Verdict
**neural_raw extends the graded pattern to the grid environment: False.**
(tabular and linear_raw results, for comparison, are unchanged from cue_validity_cross_environment_report.md: linear_raw showed p<0.0001, d=6.12/5.44 at the same two validity levels.)
