# Cue Validity Shuffled/Placebo Control

## Pre-registration
Design and criteria were written and committed in docs/cue_validity_shuffle_placebo.md before any of the data below was collected.

## Design summary
linear_raw only, cue_validity in [0.5, 0.95], 15 seeds. Shuffle arm replaces decorrelate()'s fresh i.i.d. resampling with a permutation of the realized acquisition cue sequence. Compared against the already-committed decorrelate arm from cue_validity_proxy_dependence_experiment.py.

## Results

| validity | shuffle gap | decorrelate gap | t | p | Cohen's d | placebo concern ruled out |
|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 0.054 | 0.054 | -0.197 | 0.8452 | -0.074 | True |
| 0.95 | 0.104 | 0.102 | 0.819 | 0.4195 | 0.310 | True |

## Verdict
**Placebo-mechanism concern ruled out at both validity levels: True.** (Shuffle and decorrelate gaps are statistically indistinguishable, meaning decorrelate()'s own resampling mechanism is not itself contributing to the measured persistence effect beyond the mere removal of cue information.)
