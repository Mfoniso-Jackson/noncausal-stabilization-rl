# Cue Validity Reversal Experiment

## Pre-registration
Design, metrics, and success/failure criteria were written and committed in docs/cue_validity_reversal.md before any of the data below was collected.

## Design summary
cue_validity in [0.5, 0.95], agent/representation in ['tabular', 'linear_raw', 'linear_orthogonal'], 15 seeds. Same environment and agents as the decorrelation experiment; env.reverse() replaces env.decorrelate() as the intervention. Primary comparison: this experiment's per-seed |reversed_final - naive_reversed_final| gap against the already-collected decorrelation experiment's |decorrelate_final - naive_final| gap, same agent type and validity.

## Results table (mean across seeds, signed proxy_dependence)

| agent_type | validity | acquisition dep | reversed final dep | naive-reversed final dep |
|---|---:|---:|---:|---:|
| tabular | 0.5 | 0.112 | -0.100 | -0.102 |
| tabular | 0.95 | 0.194 | -0.196 | -0.186 |
| linear_raw | 0.5 | 0.090 | 0.021 | -0.050 |
| linear_raw | 0.95 | 0.162 | 0.033 | -0.100 |
| linear_orthogonal | 0.5 | 0.076 | 0.010 | -0.056 |
| linear_orthogonal | 0.95 | 0.145 | 0.025 | -0.110 |

## Gradient-asymmetry test: reversal gap vs decorrelation gap (|pretrained final - naive final|)

| agent_type | validity | reversal gap | decorrelation gap | t | p | Cohen's d | asymmetry supported |
|---|---:|---:|---:|---:|---:|---:|---|
| tabular | 0.5 | 0.022 | 0.023 | -0.188 | 0.8523 | -0.071 | False |
| tabular | 0.95 | 0.022 | 0.023 | -0.141 | 0.8889 | -0.053 | False |
| linear_raw | 0.5 | 0.071 | 0.054 | 2.976 | 0.0060 | 1.125 | False |
| linear_raw | 0.95 | 0.132 | 0.102 | 3.443 | 0.0018 | 1.301 | False |
| linear_orthogonal | 0.5 | 0.067 | 0.046 | 3.607 | 0.0012 | 1.363 | False |
| linear_orthogonal | 0.95 | 0.135 | 0.094 | 4.824 | 0.0000 | 1.823 | False |

## Verdict (pre-registered criterion: raw |pretrained final - naive final| gap)
Gradient-asymmetry mechanism supported (reversal gap significantly smaller than decorrelation gap) in at least one function-approximation condition: False.
Tabular gaps stayed small under both interventions, as expected (not a test of the asymmetry mechanism): True.

## Post-hoc analysis: the raw-gap comparison conflates distance with speed
Reversal's fully-adapted target is a full sign flip from the acquisition value; decorrelation's is approximately zero -- a strictly smaller distance from the same starting point. The raw gap above is therefore not a fair comparison of adaptation *speed*: an agent could be adapting faster under reversal in relative terms while still showing a larger absolute residual gap, simply because it had further to travel in the same episode budget. The table below instead reports, per seed, the fraction of the required change (acquisition value to the naive control's fully-adapted value) actually achieved by the final block -- a comparison of rates, not distances. This was not part of the pre-registered criteria in docs/cue_validity_reversal.md and is reported as exploratory, not confirmatory.

| agent_type | validity | reversal fraction adapted | decorrelation fraction adapted | t | p | Cohen's d | reversal faster |
|---|---:|---:|---:|---:|---:|---:|---|
| tabular | 0.5 | 1.009 | 1.093 | -1.109 | 0.2770 | -0.419 | False |
| tabular | 0.95 | 1.029 | 1.050 | -0.476 | 0.6379 | -0.180 | False |
| linear_raw | 0.5 | 0.509 | 0.349 | 6.354 | 0.0000 | 2.402 | True |
| linear_raw | 0.95 | 0.504 | 0.341 | 7.953 | 0.0000 | 3.006 | True |
| linear_orthogonal | 0.5 | 0.512 | 0.368 | 4.236 | 0.0002 | 1.601 | True |
| linear_orthogonal | 0.95 | 0.482 | 0.336 | 6.459 | 0.0000 | 2.441 | True |

## Revised verdict (post-hoc, normalized comparison)
Under the normalized (rate, not distance) comparison, reversal shows a significantly larger fraction of the required adaptation achieved than decorrelation, consistently across both linear representations and both validity levels: True. Tabular reaches close to full adaptation (fraction close to 1.0) under both interventions, as expected. This reverses the pre-registered raw-gap verdict above and is consistent with the gradient-asymmetry mechanism this experiment set out to test -- reversal's active, structured corrective signal does produce proportionally faster unlearning of the old association than decorrelation's passive, zero-mean signal, for both linear-representation conditions, even though reversal's absolute residual gap is larger because it had a longer distance to cover in the same episode budget.

## Limitations
The naive-reversed control, like the decorrelation experiment's naive control, starts from a zero-initialized/fresh agent rather than an agent with unrelated prior training, so it is a lower bound on how much prior history matters, not a full dissociation of 'any training' from 'this specific correlation's training'. The comparison against the decorrelation experiment's results reuses that experiment's seeds and episode budget exactly, but the two experiments were run as separate processes, not interleaved, so any uncontrolled environmental difference between the two runs (e.g. library versions) would not be caught by this design. The pre-registered raw-gap criterion and the post-hoc normalized criterion disagree, which is itself the main methodological lesson of this report: a 'gap to naive control' metric should be normalized by the required distance whenever the fully-adapted target differs between conditions being compared, and this was not anticipated when docs/cue_validity_reversal.md was written.
