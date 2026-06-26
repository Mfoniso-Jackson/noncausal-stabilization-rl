# Factorial Credit-Assignment Ambiguity Report

2x2 design: reward delay low/high crossed with reward sparsity low/high.

| condition | reward | goal_rate | CDS | SPI |
|---|---:|---:|---:|---:|
| low_delay_low_sparsity | 0.820 +/- 0.000 | 1.000 +/- 0.000 | 0.039 +/- 0.031 | 0.300 +/- 0.167 |
| high_delay_low_sparsity | -0.026 +/- 0.470 | 0.700 +/- 0.167 | 0.107 +/- 0.089 | 0.400 +/- 0.178 |
| low_delay_high_sparsity | -0.137 +/- 0.127 | 0.967 +/- 0.065 | 0.069 +/- 0.039 | 0.567 +/- 0.180 |
| high_delay_high_sparsity | -1.105 +/- 0.348 | 0.467 +/- 0.182 | 0.119 +/- 0.072 | 0.433 +/- 0.180 |

| metric | effect | estimate | 95% CI | std. effect | p approx |
|---|---|---:|---:|---:|---:|
| CDS | delay_main | 0.068 | +/- 0.088 | 0.388 | 0.1331 |
| CDS | sparsity_main | 0.030 | +/- 0.088 | 0.174 | 0.4998 |
| CDS | delay_x_sparsity | -0.019 | +/- 0.125 | -0.107 | 0.7702 |
| SPI | delay_main | 0.100 | +/- 0.250 | 0.203 | 0.4324 |
| SPI | sparsity_main | 0.267 | +/- 0.250 | 0.541 | 0.0363 |
| SPI | delay_x_sparsity | -0.233 | +/- 0.353 | -0.473 | 0.1952 |

High delay + high sparsity has largest CDS: True.
High delay + high sparsity has largest SPI: False.
High delay + high sparsity remains learnable: False.
Supports credit-assignment ambiguity: False.
