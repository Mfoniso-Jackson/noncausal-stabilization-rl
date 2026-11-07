# PPO Retest: Fixed 800-Episode Acquisition Budget

## Pre-registration
Design and criteria were written and committed in docs/cue_validity_ppo.md ("Properly-powered retest") before this data was collected, following a diagnostic (not pre-registered) run that found PPO's proxy_dependence follows an inverted-U trajectory, not monotonic growth, driven by policy entropy's collapse toward convergence.

## Results

| validity | decorrelate | naive | t | p | Cohen's d | bootstrap diff CI | has effect |
|---:|---:|---:|---:|---:|---:|---|---|
| 0.5 | 0.016 | 0.024 | -2.105 | 0.0444 | -0.796 | [-0.014, -0.001] | False |
| 0.95 | 0.018 | 0.024 | -1.790 | 0.0842 | -0.677 | [-0.012, 0.000] | False |

## Acquisition-phase diagnostics (mean across seeds, at episode 800)

| validity | acquisition proxy_dependence | acquisition entropy |
|---:|---:|---:|
| 0.5 | 0.0709 | 0.6824 |
| 0.95 | 0.0917 | 0.6763 |

## Naive-arm entropy (diagnoses whether the naive arm is confounded by its own entropy-collapse trajectory)

| validity | naive final proxy_dependence | naive final entropy |
|---:|---:|---:|
| 0.5 | 0.0234 | 0.1588 |
| 0.95 | 0.0234 | 0.1588 |

## Verdict (H4, properly-powered retest)
PPO shows the linear_raw/neural_raw pattern at the fixed, diagnostic-informed acquisition budget: False.
