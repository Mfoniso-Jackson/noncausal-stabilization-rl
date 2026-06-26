# Experimental Pipeline Summary

Trend test: linear monotonic model compared against concave quadratic inverted-U model.

| condition | uncertainty | reward 95% CI | goal_rate 95% CI | CDS 95% CI | SPI 95% CI |
|---|---:|---:|---:|---:|---:|
| very_low | 0.000 | 0.820 +/- 0.000 | 1.000 +/- 0.000 | 0.039 +/- 0.031 | 0.300 +/- 0.167 |
| low | 0.256 | 0.327 +/- 0.111 | 0.989 +/- 0.012 | 0.047 +/- 0.031 | 0.333 +/- 0.171 |
| medium | 0.567 | -0.282 +/- 0.242 | 0.876 +/- 0.106 | 0.059 +/- 0.038 | 0.433 +/- 0.180 |
| high | 0.818 | -0.956 +/- 0.267 | 0.636 +/- 0.155 | 0.038 +/- 0.029 | 0.292 +/- 0.163 |
| very_high | 0.967 | -1.290 +/- 0.237 | 0.478 +/- 0.155 | 0.110 +/- 0.079 | 0.318 +/- 0.165 |
| baseline |  | 0.820 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| cue_baseline |  | 0.820 +/- 0.000 | 1.000 +/- 0.000 | 0.039 +/- 0.031 | 0.300 +/- 0.167 |

| metric | preferred | evidence | peak uncertainty | delta AIC | monotonic pattern |
|---|---|---|---:|---:|---|
| CDS | tie | inconclusive | 0.256 | -1.134 | non_monotonic |
| SPI | tie | inconclusive | 0.490 | -1.121 | non_monotonic |
