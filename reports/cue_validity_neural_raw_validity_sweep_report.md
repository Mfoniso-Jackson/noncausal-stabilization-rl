# neural_raw Validity Sweep

## Pre-registration
Design, metrics, and success/failure criteria were written and committed in docs/cue_validity_neural_raw_validity_sweep.md before any of the new data below was collected.

## Design summary
condition=neural_raw only. New cue_validity levels [0.6, 0.75, 0.9] (15 seeds each, all four arms freshly run). Existing levels [0.5, 0.95] reused from already-committed results (15 decorrelation seeds, 15 or 30 reversal seeds per the seed top-up).

## Five-point curve

| validity | source | n decorr. seeds | n reversal seeds | decorrelation frac. adapted | reversal frac. adapted | headroom | asymmetry effect | t | p | Cohen's d | asymmetry supported |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5 | reused | 15 | 30 | 0.898 | 0.944 | 0.102 | 0.046 | 1.227 | 0.2264 | 0.372 | False |
| 0.6 | new | 15 | 15 | 0.889 | 0.960 | 0.111 | 0.070 | 1.773 | 0.0871 | 0.670 | False |
| 0.75 | new | 15 | 15 | 0.887 | 0.956 | 0.113 | 0.069 | 1.992 | 0.0562 | 0.753 | False |
| 0.9 | new | 15 | 15 | 0.885 | 0.947 | 0.115 | 0.062 | 2.118 | 0.0431 | 0.801 | True |
| 0.95 | reused | 15 | 30 | 0.894 | 0.947 | 0.106 | 0.054 | 2.554 | 0.0143 | 0.733 | True |

## Primary (pre-registered) test: headroom vs. asymmetry effect, Pearson correlation across the 5 levels
r = 0.8413, p = 0.0740 (n=5 points; explicitly pre-registered as underpowered and descriptive, not confirmatory, regardless of outcome).
Headroom hypothesis supported (r > 0, p < 0.05): False.
