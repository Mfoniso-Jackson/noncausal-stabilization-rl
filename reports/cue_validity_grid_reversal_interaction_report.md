# Grid Environment: Reversal and Interaction-Only Extension

## Pre-registration
Design and criteria were written and committed in docs/cue_validity_grid_reversal_interaction.md before any of the data below was collected. Holm-Bonferroni correction applied within each 2-cell family from the start.

## Reversal (linear_raw): fraction-of-required-adaptation, reversal vs. decorrelation

| validity | reversal frac. adapted | decorrelation frac. adapted | t | raw p | holm p | Cohen's d | asymmetry supported |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 0.975 | 0.467 | 15.449 | 0.0000 | 0.0000 | 5.839 | True |
| 0.95 | 0.876 | 0.419 | 5.536 | 0.0000 | 0.0000 | 2.093 | True |

## Interaction-only (linear_interaction_only): own decorrelate-vs-naive effect, and coordinate-dependence test vs. linear_raw

| validity | decorrelate | naive | own raw p | own holm p | has own effect | coord raw p | coord holm p | coordinate dependent |
|---:|---:|---:|---:|---:|---|---:|---:|---|
| 0.5 | 0.042 | 0.028 | 0.0442 | 0.0627 | False | 0.0000 | 0.0000 | True |
| 0.95 | 0.013 | 0.028 | 0.0313 | 0.0627 | False | 0.0000 | 0.0000 | True |

## Verdict
Reversal: gradient-asymmetry mechanism supported in the grid environment (after correction): True.
Interaction-only: linear_interaction_only shows its own decorrelate-vs-naive effect (after correction): False.
Interaction-only: coordinate-dependent relative to linear_raw (after correction): True.
