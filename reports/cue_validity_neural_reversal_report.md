# Cue Validity Neural Reversal Experiment

## Pre-registration
Design, metrics, and success/failure criteria were written and committed in docs/cue_validity_neural_reversal.md before any of the data below was collected.

## Design summary
cue_validity in [0.5, 0.95], condition in ['linear_interaction_only', 'neural_raw', 'neural_interaction_only'], 15 seeds. Same agents, feature encodings, and episode budget as the nonlinear experiment; env.reverse() replaces env.decorrelate() as the intervention. Primary comparison (pre-registered): per-seed fraction of required adaptation achieved, reversal vs. the already-collected decorrelation data from results/cue_validity_nonlinear_summaries.csv.

## Results table (mean across seeds, signed proxy_dependence)

| condition | validity | acquisition dep | reversed final dep | naive-reversed final dep |
|---|---:|---:|---:|---:|
| linear_interaction_only | 0.5 | 0.112 | -0.033 | -0.080 |
| linear_interaction_only | 0.95 | 0.207 | -0.091 | -0.157 |
| neural_raw | 0.5 | 0.100 | -0.085 | -0.093 |
| neural_raw | 0.95 | 0.191 | -0.165 | -0.186 |
| neural_interaction_only | 0.5 | 0.101 | -0.087 | -0.098 |
| neural_interaction_only | 0.95 | 0.193 | -0.166 | -0.190 |

## Primary (pre-registered) test: fraction of required adaptation achieved, reversal vs. decorrelation

| condition | validity | reversal fraction adapted | decorrelation fraction adapted | t | p | Cohen's d | asymmetry supported |
|---|---:|---:|---:|---:|---:|---:|---|
| linear_interaction_only | 0.5 | 0.765 | 0.671 | 2.393 | 0.0237 | 0.904 | True |
| linear_interaction_only | 0.95 | 0.827 | 0.757 | 2.490 | 0.0190 | 0.941 | True |
| neural_raw | 0.5 | 0.966 | 0.898 | 1.416 | 0.1678 | 0.535 | False |
| neural_raw | 0.95 | 0.946 | 0.894 | 1.948 | 0.0615 | 0.736 | False |
| neural_interaction_only | 0.5 | 0.950 | 1.012 | -0.841 | 0.4075 | -0.318 | False |
| neural_interaction_only | 0.95 | 0.939 | 0.950 | -0.301 | 0.7657 | -0.114 | False |

## Secondary, reported for comparability only: raw |reversed final - naive-reversed final| gap
Not used to decide support for the mechanism (see pre-registration: reversal and decorrelation have different-distance fully-adapted targets, so raw gaps are not a fair speed comparison).

| condition | validity | raw reversal gap |
|---|---:|---:|
| linear_interaction_only | 0.5 | 0.050 |
| linear_interaction_only | 0.95 | 0.071 |
| neural_raw | 0.5 | 0.018 |
| neural_raw | 0.95 | 0.021 |
| neural_interaction_only | 0.5 | 0.019 |
| neural_interaction_only | 0.95 | 0.024 |

## Verdict
Gradient-asymmetry mechanism supported (reversal fraction-adapted significantly larger than decorrelation fraction-adapted) in at least one condition: True.
- linear_interaction_only: True
- neural_raw: False
- neural_interaction_only: False

## Limitations
neural_interaction_only showed no significant decorrelate-vs-naive effect in the nonlinear experiment, so there is no persistent dependency for reversal to extinguish faster than there; its fraction-adapted comparison is reported for completeness but a null or noisy result there is expected, not a failure of the mechanism. The naive-reversed control, like both parent experiments' naive controls, starts from a fresh agent rather than one with unrelated prior training, so it is a lower bound on how much prior history matters. This experiment and the nonlinear experiment it reuses decorrelation data from were run as separate processes, not interleaved, so any uncontrolled environmental difference between the two runs would not be caught by this design.
