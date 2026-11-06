# Step-Count-Matched Retest of neural_raw in Grid Environment

## Pre-registration
docs/cue_validity_cross_environment.md, "Step-count-matched retest" -- written before this data was collected. Flagged in advance as a weak-hypothesis check, not the leading explanation.

## Design
neural_raw only, 800-episode intervention budget in 16 blocks of 50 (vs. 1500/30 in the main extension), chosen to approximately match total training-call count against the original environment's 1500-episode budget.

## Results

| validity | decorrelate | naive | t | p | Cohen's d | has effect |
|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 0.027 | 0.011 | 3.010 | 0.0055 | 1.138 | True |
| 0.95 | 0.031 | 0.011 | 3.465 | 0.0017 | 1.309 | True |

## Verdict
**Step-count hypothesis supported: True.**
