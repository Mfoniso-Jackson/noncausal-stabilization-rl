# Grid Environment: neural_raw Reversal and neural_interaction_only Extension

## Pre-registration
Design and criteria were written and committed in docs/cue_validity_grid_neural_reversal_interaction.md before any of the data below was collected. Uses the already-established step-matched 800-episode budget for neural_raw in this environment, applied from the start. Holm-Bonferroni correction applied within each 2-cell family.

## Reversal (neural_raw): fraction-of-required-adaptation, reversal vs. step-matched decorrelation

| validity | reversal frac. adapted | decorrelation frac. adapted | t | raw p | holm p | Cohen's d | asymmetry supported |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 1.350 | 0.898 | 2.938 | 0.0065 | 0.0131 | 1.110 | True |
| 0.95 | 1.166 | 0.894 | 2.721 | 0.0111 | 0.0131 | 1.028 | True |

## Interaction-only (neural_interaction_only): own decorrelate-vs-naive effect, and coordinate-dependence test vs. neural_raw (step-matched)

| validity | decorrelate | naive | own raw p | own holm p | has own effect | coord raw p | coord holm p | coordinate dependent |
|---:|---:|---:|---:|---:|---|---:|---:|---|
| 0.5 | 0.015 | 0.033 | 0.0097 | 0.0194 | False | 0.0000 | 0.0000 | True |
| 0.95 | 0.036 | 0.033 | 0.7390 | 0.7390 | False | 0.0613 | 0.0613 | False |

## Verdict
Reversal: neural_raw gradient-asymmetry mechanism supported in the grid environment (after correction): True.
Interaction-only: neural_interaction_only shows its own decorrelate-vs-naive effect (after correction): False.
Interaction-only: coordinate-dependent relative to neural_raw (after correction): True.
