# Cue Validity Nonlinear & Interaction-Only Representation Experiment

## Pre-registration
Design, metrics, and success/failure criteria were written and committed in docs/cue_validity_nonlinear.md before any of the data below was collected.

## Design summary
cue_validity in [0.5, 0.95], 15 seeds for the new conditions ['linear_interaction_only', 'neural_raw', 'neural_interaction_only']. tabular/linear_raw/linear_orthogonal are not re-run; their already-committed results from the proxy-dependence experiment are reused directly for comparison.

## Results table (new conditions, mean across seeds)

| condition | validity | acquisition dep | final decorrelate dep | final naive dep |
|---|---:|---:|---:|---:|
| linear_interaction_only | 0.5 | 0.112 | 0.046 | 0.012 |
| linear_interaction_only | 0.95 | 0.207 | 0.060 | 0.012 |
| neural_raw | 0.5 | 0.100 | 0.017 | 0.006 |
| neural_raw | 0.95 | 0.191 | 0.026 | 0.006 |
| neural_interaction_only | 0.5 | 0.101 | 0.012 | 0.009 |
| neural_interaction_only | 0.95 | 0.193 | 0.020 | 0.009 |

## Decorrelate-vs-naive gap, all conditions (new + reused prior results)

| condition | validity | decorrelate | naive | gap | t | p | Cohen's d | has effect |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tabular (prior) | 0.5 | 0.017 | 0.017 | -0.001 | -0.128 | 0.8989 | -0.048 | False |
| tabular (prior) | 0.95 | 0.017 | 0.017 | -0.001 | -0.128 | 0.8989 | -0.048 | False |
| linear_raw (prior) | 0.5 | 0.062 | 0.009 | 0.052 | 32.225 | 0.0000 | 12.180 | True |
| linear_raw (prior) | 0.95 | 0.110 | 0.009 | 0.100 | 62.412 | 0.0000 | 23.589 | True |
| linear_orthogonal (prior) | 0.5 | 0.050 | 0.007 | 0.043 | 21.271 | 0.0000 | 8.040 | True |
| linear_orthogonal (prior) | 0.95 | 0.098 | 0.007 | 0.091 | 47.163 | 0.0000 | 17.826 | True |
| linear_interaction_only | 0.5 | 0.046 | 0.013 | 0.033 | 10.124 | 0.0000 | 3.827 | True |
| linear_interaction_only | 0.95 | 0.060 | 0.013 | 0.047 | 14.227 | 0.0000 | 5.377 | True |
| neural_raw | 0.5 | 0.017 | 0.009 | 0.009 | 2.783 | 0.0095 | 1.052 | True |
| neural_raw | 0.95 | 0.026 | 0.009 | 0.018 | 4.293 | 0.0002 | 1.623 | True |
| neural_interaction_only | 0.5 | 0.012 | 0.012 | 0.000 | 0.028 | 0.9780 | 0.011 | False |
| neural_interaction_only | 0.95 | 0.021 | 0.012 | 0.008 | 1.939 | 0.0627 | 0.733 | False |

## Coordinate-dependence test (raw vs interaction-only gap, per architecture)

| architecture | validity | raw gap | interaction-only gap | t | p | Cohen's d | coordinate-dependent |
|---|---:|---:|---:|---:|---:|---:|---|
| linear | 0.5 | 0.052 | 0.033 | 9.763 | 0.0000 | 3.690 | True |
| linear | 0.95 | 0.100 | 0.047 | 25.529 | 0.0000 | 9.649 | True |
| neural | 0.5 | 0.009 | 0.000 | 2.446 | 0.0210 | 0.924 | True |
| neural | 0.95 | 0.018 | 0.008 | 1.948 | 0.0615 | 0.736 | False |

## Interpretation: a graded effect, not an all-or-nothing artifact
Mean decorrelate-vs-naive gap across the two validity levels: linear_raw=0.076, linear_interaction_only=0.040, neural_raw=0.013, neural_interaction_only=0.004. Both factors tested here -- removing direct coordinate access (interaction-only encoding) and increasing representational flexibility (linear to nonlinear) -- shrink the effect individually (linear_interaction_only and neural_raw both still show a significant effect, smaller than linear_raw's), and the effect is statistically indistinguishable from the naive-control floor only when both factors are combined (neural_interaction_only). This is a more precise picture than 'the effect is a brittle linear-representation artifact' (this repository's own conclusion from a different environment, in orthogonalized_proxy_feature_experiment.py / nonlinear_interaction_proxy_experiment.py): here persistence appears to be a graded property of model capacity and feature accessibility, attenuating smoothly as either increases, rather than a binary property of linearity alone.

## Verdict
neural_raw shows a decorrelate-vs-naive effect at at least one validity level: True.
neural_interaction_only shows a decorrelate-vs-naive effect at at least one validity level: False.
The effect is coordinate-dependent (requires a raw, easily-accessible cue coordinate) for at least one architecture: True.

## Limitations
15 seeds per new condition, matching the proxy-dependence experiment. A single MLP architecture (2 hidden layers, 32 units) and a single interaction-only encoding are tested, not a sweep, so the 'graded' claim above rests on one point along each of two axes (capacity, accessibility), not a full curve. The neural agent's hyperparameters were chosen by reusing this repository's established recurrent-agent defaults rather than tuned for this specific task. neural_interaction_only's p-value at validity=0.95 (0.0627) is close to the significance threshold with 15 seeds; a larger seed count could plausibly move this cell either way.
