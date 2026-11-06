# Multiple-Comparison Correction: Analysis Output

Policy and interpretation: docs/multiple_comparison_correction.md.

## MVE (criterion 1, decorrelate-vs-naive) (n=2)

| cell | raw p | holm-adjusted p | survives (alpha=0.05) |
|---|---:|---:|---|
| v0.5 | 0.774348 | 1.000000 | False |
| v0.95 | 0.911307 | 1.000000 | False |

## Proxy-dependence (criterion 1) (n=6)

| cell | raw p | holm-adjusted p | survives (alpha=0.05) |
|---|---:|---:|---|
| tabular_v0.5 | 0.898872 | 1.000000 | False |
| tabular_v0.95 | 0.898872 | 1.000000 | False |
| linear_raw_v0.5 | 0.000000 | 0.000000 | True |
| linear_raw_v0.95 | 0.000000 | 0.000000 | True |
| linear_orthogonal_v0.5 | 0.000000 | 0.000000 | True |
| linear_orthogonal_v0.95 | 0.000000 | 0.000000 | True |

## Reversal (pre-registered raw-gap) (n=6)

| cell | raw p | holm-adjusted p | survives (alpha=0.05) |
|---|---:|---:|---|
| tabular_v0.5 | 0.852279 | 1.000000 | False |
| tabular_v0.95 | 0.888875 | 1.000000 | False |
| linear_raw_v0.5 | 0.005962 | 0.017886 | True |
| linear_raw_v0.95 | 0.001830 | 0.007320 | True |
| linear_orthogonal_v0.5 | 0.001191 | 0.005955 | True |
| linear_orthogonal_v0.95 | 0.000045 | 0.000270 | True |

## Nonlinear (coordinate-dependence test) (n=4)

| cell | raw p | holm-adjusted p | survives (alpha=0.05) |
|---|---:|---:|---|
| linear_v0.5 | 0.000000 | 0.000000 | True |
| linear_v0.95 | 0.000000 | 0.000000 | True |
| neural_v0.5 | 0.020994 | 0.041988 | True |
| neural_v0.95 | 0.061480 | 0.061480 | False |

## Neural reversal (pre-registered normalized, neural_raw via topup) (n=6)

| cell | raw p | holm-adjusted p | survives (alpha=0.05) |
|---|---:|---:|---|
| linear_interaction_only_v0.5 | 0.023655 | 0.094995 | False |
| linear_interaction_only_v0.95 | 0.018999 | 0.094995 | False |
| neural_interaction_only_v0.5 | 0.407455 | 0.814910 | False |
| neural_interaction_only_v0.95 | 0.765711 | 0.814910 | False |
| neural_raw_topup_v0.5 | 0.226355 | 0.679065 | False |
| neural_raw_topup_v0.95 | 0.014269 | 0.085614 | False |

## Optimizer ablation (primary) (n=4)

| cell | raw p | holm-adjusted p | survives (alpha=0.05) |
|---|---:|---:|---|
| neural_raw_v0.5 | 0.018046 | 0.036092 | True |
| neural_raw_v0.95 | 0.000288 | 0.001152 | True |
| neural_interaction_only_v0.5 | 0.064163 | 0.064163 | False |
| neural_interaction_only_v0.95 | 0.002459 | 0.007377 | True |

## Replay buffer sweep (n=8)

| cell | raw p | holm-adjusted p | survives (alpha=0.05) |
|---|---:|---:|---|
| buf500_v0.5 | 0.000000 | 0.000000 | True |
| buf500_v0.95 | 0.000000 | 0.000000 | True |
| buf2000_v0.5 | 0.000000 | 0.000000 | True |
| buf2000_v0.95 | 0.000000 | 0.000000 | True |
| buf8000_v0.5 | 0.000000 | 0.000000 | True |
| buf8000_v0.95 | 0.000000 | 0.000000 | True |
| buf20000_v0.5 | 0.000000 | 0.000000 | True |
| buf20000_v0.95 | 0.000000 | 0.000000 | True |

## Capacity sweep (n=10)

| cell | raw p | holm-adjusted p | survives (alpha=0.05) |
|---|---:|---:|---|
| w8_v0.5 | 0.000038 | 0.000304 | True |
| w8_v0.95 | 0.000002 | 0.000020 | True |
| w16_v0.5 | 0.001488 | 0.008928 | True |
| w16_v0.95 | 0.000014 | 0.000126 | True |
| w32_v0.5 | 0.009544 | 0.047720 | True |
| w32_v0.95 | 0.000191 | 0.001337 | True |
| w64_v0.5 | 0.138911 | 0.327411 | False |
| w64_v0.95 | 0.014730 | 0.058920 | False |
| w128_v0.5 | 0.540633 | 0.540633 | False |
| w128_v0.95 | 0.109137 | 0.327411 | False |

## Shuffle/placebo control (excluded from discovery family; success = non-significance)

| validity | raw p (uncorrected; success criterion is p>=0.05) |
|---:|---:|
| 0.5 | 0.845214 |
| 0.95 | 0.419522 |
