# Recurrent Belief-State Interaction Experiment

## Research question
recurrent_belief_state_supervision_report.md found no significant benefit from belief-state supervision at history_length=5 and aux_weight=1.0, and recommended sweeping the auxiliary loss weight and testing whether any benefit compounds with longer history. This experiment crosses history_length in {1, 5} with aux_weight in {0.0 (unsupervised), 4.0 (strong supervision)} in a 2x2 design, to test both the main effect of a much stronger supervision weight and its interaction with memory capacity.

## Design
Same hidden-state POMDP, 3-phase acquisition/reversal/extinction schedule (400/200/200 episodes), and two-head GRU(64) architecture as the two prior recurrent experiments. 2 history lengths x 2 aux weights x 5 seeds = 20 total runs.

## Results table (final checkpoint of each phase)

| history_length | aux_weight | phase | abs_proxy_dependence | belief_accuracy |
|---:|---:|---|---:|---:|
| 1 | 0.0 | acquisition | 0.784 +/- 0.231 | 0.490 +/- 0.048 |
| 1 | 0.0 | reversal | 0.517 +/- 0.360 | 0.490 +/- 0.050 |
| 1 | 0.0 | extinction | 0.446 +/- 0.320 | 0.502 +/- 0.024 |
| 1 | 4.0 | acquisition | 0.722 +/- 0.207 | 0.968 +/- 0.022 |
| 1 | 4.0 | reversal | 0.316 +/- 0.289 | 0.056 +/- 0.021 |
| 1 | 4.0 | extinction | 0.606 +/- 0.250 | 0.502 +/- 0.047 |
| 5 | 0.0 | acquisition | 0.171 +/- 0.099 | 0.576 +/- 0.210 |
| 5 | 0.0 | reversal | 0.206 +/- 0.198 | 0.424 +/- 0.164 |
| 5 | 0.0 | extinction | 0.284 +/- 0.246 | 0.502 +/- 0.024 |
| 5 | 4.0 | acquisition | 0.450 +/- 0.178 | 0.992 +/- 0.007 |
| 5 | 4.0 | reversal | 0.163 +/- 0.064 | 0.016 +/- 0.010 |
| 5 | 4.0 | extinction | 0.257 +/- 0.208 | 0.508 +/- 0.031 |

## Extinction comparison: unsupervised vs strong supervision, by history length

| history_length | none mean | strong mean | gap (none - strong) | t | p | Cohen's d |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.446 | 0.606 | -0.161 | -0.776 | 0.4598 | -0.549 |
| 5 | 0.284 | 0.257 | 0.027 | 0.162 | 0.8752 | 0.115 |

## Interaction

Gap (none - strong) at history_length=1: -0.161. Gap at history_length=5: 0.027. Difference (interaction): 0.188.

## Interpretation
A positive gap means the unsupervised condition has higher residual extinction dependence than the strongly-supervised one, i.e. supervision helped; a larger positive gap at the longer history length would be the 'compounds with memory' pattern the original recommendation hypothesized. None of the four cells' comparisons reach significance at this seed count (see p-values above); any difference in the gaps between history lengths should be read as suggestive at best with 5 seeds per cell. The belief-head shortcut-learning pattern from recurrent_belief_state_supervision_experiment.py replicates cleanly at both history lengths and this stronger aux_weight=4.0: acquisition belief_accuracy reaches 0.968 at history_length=1 and 0.992 at history_length=5, then collapses to 0.056 and 0.016 respectively the moment the proxy cue reverses -- both close to the reversed cue's own predictive floor, confirming across two independent experiments that the belief head tracks the proxy-cue shortcut rather than anything causal, regardless of history length or supervision strength.

## Does a stronger auxiliary weight help, and does any benefit compound with longer history?
Any cell individually significant (p<0.05): False. Benefit appears to compound with longer history (larger positive gap at the longer history length): False.

## Limitations
Only 5 seeds per cell (a 2x2 design, 20 runs total) given CPU-only PyTorch training cost; this is underpowered to detect anything but a large interaction effect. Only two aux_weight values (0 and 4.0) are tested rather than a full sweep, and two history lengths (1 and 5) rather than the three tested in recurrent_belief_state_history_experiment.py, to keep the factorial design tractable.

## Next recommended experiment
If any cell shows a promising (even if non-significant) effect, rerun that specific history_length x aux_weight combination alone with substantially more seeds to get a properly powered estimate, rather than continuing to expand the factorial design.
