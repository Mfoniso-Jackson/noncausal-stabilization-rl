# Cue Validity Uncertainty Sweep

## Pre-registration
Design and criteria were written and committed in docs/cue_validity_uncertainty_sweep.md before any of the data below was collected. reward_noise is an environment property only (env/cue_validity_env.py); no agent setting is ever a function of it. Holm-Bonferroni correction applied across the full 6-cell family (3 noise levels x 2 validity levels).

## Results

| reward_noise | validity | decorrelate | naive | t | raw p | holm p | Cohen's d | has effect |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.0 | 0.0 | 0.0081 | 0.0094 | -0.782 | 0.4406 | 1.0000 | -0.296 | False |
| 0.0 | 0.95 | 0.1099 | 0.0094 | 62.412 | 0.0000 | 0.0000 | 23.589 | True |
| 0.3 | 0.0 | 0.0060 | 0.0059 | 0.068 | 0.9463 | 1.0000 | 0.026 | False |
| 0.3 | 0.95 | 0.0731 | 0.0059 | 42.958 | 0.0000 | 0.0000 | 16.237 | True |
| 0.6 | 0.0 | 0.0032 | 0.0035 | -0.318 | 0.7532 | 1.0000 | -0.120 | False |
| 0.6 | 0.95 | 0.0393 | 0.0035 | 31.417 | 0.0000 | 0.0000 | 11.875 | True |

## Verdict
Primary pre-registered comparison (v=0.95, Cohen's d at noise=0.6 vs. noise=0.0): 11.875 vs. 23.589. Supported (uncertainty increases persistence): False.
Floor control (v=0.0) remains null at every noise level, as required for interpretability: True.
