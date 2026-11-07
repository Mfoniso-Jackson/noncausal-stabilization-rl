# Cue Validity Optimizer Ablation (SGD vs. Adam)

## Pre-registration
Design, metrics, and success/failure criteria were written and committed in docs/cue_validity_optimizer_ablation.md before any of the data below was collected. This experiment is load-bearing for the publication plan's own Go/No-Go criteria (Section 12): an SGD effect loss where Adam showed one is grounds to abandon or pivot the neural portion of the claim.

## Primary test: does the decorrelate-vs-naive effect survive under SGD (no momentum, same learning rate)?

| condition | validity | SGD decorrelate | SGD naive | gap | t | p | Cohen's d | effect under SGD |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| neural_raw | 0.5 | 0.015 | 0.007 | 0.008 | 2.512 | 0.0180 | 0.949 | True |
| neural_raw | 0.95 | 0.020 | 0.007 | 0.013 | 4.140 | 0.0003 | 1.565 | True |
| neural_interaction_only | 0.5 | 0.024 | 0.015 | 0.010 | 1.927 | 0.0642 | 0.728 | False |
| neural_interaction_only | 0.95 | 0.036 | 0.015 | 0.021 | 3.328 | 0.0025 | 1.258 | True |

## Secondary: SGD vs. Adam decorrelate-arm magnitude (direct comparison, not the primary verdict)

| condition | validity | SGD decorrelate mean | Adam decorrelate mean | t | p | Cohen's d |
|---|---:|---:|---:|---:|---:|---:|
| neural_raw | 0.5 | 0.015 | 0.017 | -0.708 | 0.4848 | -0.268 |
| neural_raw | 0.95 | 0.020 | 0.026 | -1.425 | 0.1653 | -0.538 |
| neural_interaction_only | 0.5 | 0.024 | 0.012 | 2.693 | 0.0118 | 1.018 |
| neural_interaction_only | 0.95 | 0.036 | 0.021 | 2.427 | 0.0219 | 0.917 |

## Verdict
neural_raw retains a significant decorrelate-vs-naive effect under SGD at both validity levels: True.
neural_raw loses the effect under SGD at at least one validity level: False.
neural_interaction_only had no significant Adam effect to begin with; its SGD result is reported above but is not informative for the momentum question either way.

## Limitations
The learning rate (1e-3) was not re-tuned for SGD, deliberately, to isolate momentum as the single changed variable -- this means a differently-tuned SGD configuration could behave differently, and this experiment cannot rule that out. Only one architecture and one momentum value (0.0, i.e. plain SGD) were tested, not a momentum sweep.
