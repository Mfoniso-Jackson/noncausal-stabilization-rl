# PPO Capacity Sweep

## Pre-registration
Design and criteria were written and committed in docs/cue_validity_ppo_capacity_sweep.md before any of the data below was collected. Width 32 reuses already-collected results; widths 16 and 64 are new. Holm-Bonferroni correction applied across the full 6-cell family.

## Capacity curve (total-episode-matched comparison)

| width | validity | decorrelate | naive | t | raw p | holm p | Cohen's d | has effect |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 0.5 | 0.029 | 0.028 | 0.252 | 0.8026 | 1.0000 | 0.095 | False |
| 16 | 0.95 | 0.030 | 0.028 | 0.542 | 0.5922 | 1.0000 | 0.205 | False |
| 32 | 0.5 | 0.016 | 0.014 | 0.858 | 0.3980 | 1.0000 | 0.324 | False |
| 32 | 0.95 | 0.018 | 0.014 | 1.422 | 0.1662 | 0.9970 | 0.537 | False |
| 64 | 0.5 | 0.009 | 0.010 | -0.775 | 0.4448 | 1.0000 | -0.293 | False |
| 64 | 0.95 | 0.009 | 0.010 | -0.336 | 0.7396 | 1.0000 | -0.127 | False |

## Verdict
Any width shows a significant effect after correction: False.
