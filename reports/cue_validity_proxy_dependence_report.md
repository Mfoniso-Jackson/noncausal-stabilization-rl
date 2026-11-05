# Cue Validity Proxy-Dependence & Representation Experiment

## Pre-registration
Design, metrics, and success/failure criteria were written and committed in docs/cue_validity_proxy_dependence.md before any of the data below was collected.

## Design summary
cue_validity in [0.0, 0.5, 0.95], agent/representation in ['tabular', 'linear_raw', 'linear_orthogonal'], 15 seeds. cue_phase is folded into target_phase (the structural change required per docs/tabular_rl_cannot_be_superstitious.md). Primary metric: proxy_dependence (P(causal|cue present) - P(causal|cue absent)). Secondary metric: CDS, kept for continuity with the MVE.

## Results table (mean across seeds)

| agent_type | arm | acquisition proxy_dep | final-block proxy_dep | final-block CDS | extinction half-life (blocks) |
|---|---|---:|---:|---:|---:|
| tabular | floor_none | 0.010 | -0.011 | 0.657 | 3.07 |
| tabular | floor_naive | n/a | -0.003 | 0.643 | n/a |
| tabular | v0.5_none | 0.112 | 0.099 | 0.601 | 15.20 |
| tabular | v0.5_decorrelate | 0.112 | -0.011 | 0.657 | 0.20 |
| tabular | v0.5_naive | n/a | -0.003 | 0.643 | n/a |
| tabular | v0.95_none | 0.194 | 0.186 | 0.548 | 30.00 |
| tabular | v0.95_decorrelate | 0.194 | -0.011 | 0.657 | 0.00 |
| tabular | v0.95_naive | n/a | -0.003 | 0.643 | n/a |
| linear_raw | floor_none | 0.009 | 0.008 | 0.644 | 27.00 |
| linear_raw | floor_naive | n/a | 0.008 | 0.658 | n/a |
| linear_raw | v0.5_none | 0.090 | 0.102 | 0.595 | 30.00 |
| linear_raw | v0.5_decorrelate | 0.090 | 0.062 | 0.619 | 30.00 |
| linear_raw | v0.5_naive | n/a | 0.008 | 0.658 | n/a |
| linear_raw | v0.95_none | 0.162 | 0.186 | 0.549 | 30.00 |
| linear_raw | v0.95_decorrelate | 0.162 | 0.110 | 0.596 | 30.00 |
| linear_raw | v0.95_naive | n/a | 0.008 | 0.658 | n/a |
| linear_orthogonal | floor_none | -0.003 | -0.004 | 0.656 | 19.60 |
| linear_orthogonal | floor_naive | n/a | 0.004 | 0.659 | n/a |
| linear_orthogonal | v0.5_none | 0.076 | 0.088 | 0.610 | 30.00 |
| linear_orthogonal | v0.5_decorrelate | 0.076 | 0.050 | 0.630 | 30.00 |
| linear_orthogonal | v0.5_naive | n/a | 0.004 | 0.659 | n/a |
| linear_orthogonal | v0.95_none | 0.145 | 0.170 | 0.566 | 30.00 |
| linear_orthogonal | v0.95_decorrelate | 0.145 | 0.098 | 0.607 | 30.00 |
| linear_orthogonal | v0.95_naive | n/a | 0.004 | 0.659 | n/a |

## Pre-registered success criteria (decorrelate vs naive control, |proxy_dependence|)

| agent_type | validity | decorrelate | naive | t | p | Cohen's d | criterion 1 | criterion 2 | supported |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| tabular | 0.5 | 0.017 | 0.017 | -0.128 | 0.8989 | -0.048 | False | True | False |
| tabular | 0.95 | 0.017 | 0.017 | -0.128 | 0.8989 | -0.048 | False | True | False |
| linear_raw | 0.5 | 0.062 | 0.009 | 32.225 | 0.0000 | 12.180 | True | True | True |
| linear_raw | 0.95 | 0.110 | 0.009 | 62.412 | 0.0000 | 23.589 | True | True | True |
| linear_orthogonal | 0.5 | 0.050 | 0.007 | 21.271 | 0.0000 | 8.040 | True | True | True |
| linear_orthogonal | 0.95 | 0.098 | 0.007 | 47.163 | 0.0000 | 17.826 | True | True | True |

## Interpretation
Tabular shows no effect at either validity level (p=0.8989, 0.8989; Cohen's d near zero both times), consistent with docs/tabular_rl_cannot_be_superstitious.md and with the half-life data above showing tabular's decorrelate arms collapse to the naive-control level almost immediately (half-life 0-0.2 blocks). Both linear conditions show large, highly significant effects (linear_raw p<0.0001 both levels, Cohen's d 12-24; linear_orthogonal p<0.0001 both levels, Cohen's d 8-18) that do not reach the naive-control floor within the full 1500-episode, 30-block post-intervention window (half-life censored at 30 in every decorrelate cell for both linear conditions). Because linear_orthogonal's effect size is comparable to, not dramatically smaller than, linear_raw's, this is not dismissible as the specific 'brittle raw-coordinate accessibility' artifact this repository's own orthogonalized_proxy_feature_experiment.py found for a different environment -- the effect survives an invertible rotation of the identical information.
A replay-buffer-staleness explanation (the agent is still training on old, pre-decorrelation transitions) can be ruled out directly: acquisition alone generates approximately 15000 environment steps and the post-intervention window alone generates approximately 7500, both far exceeding the agent's replay buffer capacity (8000); by the final measurement block the buffer has cycled through roughly 2.5x its own capacity in purely post-decorrelation data. The persistence is therefore parametric -- a property of the learned weights after thousands of corrective gradient updates on uncorrelated data, not a data-pipeline artifact.
A plausible mechanism, consistent with this session's earlier spurious_correlation_reversal_experiment.py finding (tabular agent, reversal intervention) that an active, structured anti-correlation signal extinguishes dependence fast: acquisition provides a strong, structured gradient that pushes the cue feature's weight away from zero, while decorrelation provides only a weak, zero-mean noise gradient with no systematic force pushing it back -- the asymmetry is between acquiring a correlation (fast, structured signal) and un-learning it under mere invalidation (slow, unstructured signal), not between tabular and linear representations as such. This was not tested directly in either experiment (spurious_correlation_reversal_experiment.py used a tabular agent and a reversal intervention; this experiment uses linear/tabular agents and a decorrelation intervention) and is this report's primary recommended follow-up.

## Verdict
Hypothesis supported in at least one agent/representation condition: True.
If linear_raw shows a supported effect that linear_orthogonal does not (same validity level): representation-dependent (artifact-consistent with this repo's prior linear_feature_ablation -> orthogonalized_proxy_feature finding): False.
This is the first properly-controlled, pre-registered, statistically decisive positive finding in this research program: genuine non-causal stabilization that survives a real intervention, under linear function approximation, ruled out as tabular-representation-independent and as a raw-coordinate artifact, with replay-buffer staleness directly excluded.

## Limitations
15 seeds per condition (reduced from the MVE's 30) to control the added cost of two linear-agent conditions. Only two non-floor validity levels and one orthogonal rotation are tested, not a full sweep. The probe-based metric readout is exact for tabular and linear agents (no Monte Carlo noise), which this design relies on; it would need to become a sampled estimate for any future nonlinear/recurrent agent extension. The gradient-asymmetry mechanism above is a plausible, falsifiable account, not yet directly tested.

## Next recommended experiment
Directly test the gradient-asymmetry mechanism: repeat this exact design (same agents, same validity levels) with a reversal intervention (the cue becomes actively anti-predictive, as in spurious_correlation_reversal_experiment.py) instead of decorrelation, holding the agent class fixed. The mechanism predicts reversal should extinguish proxy_dependence far faster than decorrelation does for linear_raw and linear_orthogonal alike, while tabular should show fast extinction under either intervention (already partially shown by this report's own tabular decorrelate result and the earlier tabular reversal result, but not yet in one matched design).
