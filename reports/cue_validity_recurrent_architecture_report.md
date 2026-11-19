# Cue Validity Recurrent Architecture Comparison

## Pre-registration
Design and criteria were written and committed in docs/cue_validity_recurrent_architecture.md before any of the data below was collected. linear_raw and neural_raw rerun fresh in this script for full internal comparability; recurrent_gru uses hyperparameters matched to neural_raw wherever architecturally possible. Holm-Bonferroni correction applied across the full 6-cell family.

## Results

| condition | validity | decorrelate | naive | t | raw p | holm p | Cohen's d | has effect |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| linear_raw | 0.5 | 0.0619 | 0.0094 | 32.225 | 0.0000 | 0.0000 | 12.180 | True |
| linear_raw | 0.95 | 0.1099 | 0.0094 | 62.412 | 0.0000 | 0.0000 | 23.589 | True |
| neural_raw | 0.5 | 0.0171 | 0.0085 | 2.783 | 0.0095 | 0.0110 | 1.052 | True |
| neural_raw | 0.95 | 0.0265 | 0.0085 | 4.293 | 0.0002 | 0.0006 | 1.623 | True |
| recurrent_gru | 0.5 | 0.0405 | 0.0229 | 3.008 | 0.0055 | 0.0110 | 1.137 | True |
| recurrent_gru | 0.95 | 0.0548 | 0.0229 | 5.376 | 0.0000 | 0.0000 | 2.032 | True |

## Verdict
recurrent_gru shows the linear_raw/neural_raw pattern: True.
