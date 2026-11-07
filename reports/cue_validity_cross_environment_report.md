# Cue Validity Cross-Environment Replication

## Pre-registration
Design, metrics, and success/failure criteria were written and committed in docs/cue_validity_cross_environment.md before any of the data below was collected. This experiment is the explicit Go/No-Go gate named in the publication plan's Section 12.

## Design summary
Environment: CueValidityGridEnv (grid_size=5), a structurally different task from CueValidityEnv -- real multi-step 2D navigation to a decision cell, not a phase countdown that ignores the agent's action. condition in ['tabular', 'linear_raw'], cue_validity in [0.0, 0.5, 0.95], 15 seeds. 5000 acquisition episodes, 1500 post-intervention episodes in 30 blocks of 50.

## Results table (mean across seeds)

| condition | validity | acquisition dep | decorrelate final dep | naive final dep |
|---|---:|---:|---:|---:|
| tabular | 0.0 | -0.002 | 0.000 | 0.000 |
| tabular | 0.5 | 0.102 | 0.001 | 0.000 |
| tabular | 0.95 | 0.191 | -0.004 | 0.000 |
| linear_raw | 0.0 | 0.012 | 0.014 | 0.008 |
| linear_raw | 0.5 | 0.097 | 0.056 | 0.008 |
| linear_raw | 0.95 | 0.175 | 0.106 | 0.008 |

## Primary test: decorrelate-vs-naive gap, non-floor validity levels

| condition | validity | decorrelate | naive | t | p | Cohen's d | has effect |
|---|---:|---:|---:|---:|---:|---:|---|
| tabular | 0.5 | 0.028 | 0.021 | 1.173 | 0.2507 | 0.443 | False |
| tabular | 0.95 | 0.028 | 0.021 | 1.281 | 0.2105 | 0.484 | False |
| linear_raw | 0.5 | 0.056 | 0.011 | 16.200 | 0.0000 | 6.123 | True |
| linear_raw | 0.95 | 0.106 | 0.011 | 14.404 | 0.0000 | 5.444 | True |

## Floor control (cue_validity=0.0): comparison methodology must not manufacture a spurious gap

| condition | decorrelate | naive | t | p | Cohen's d | no spurious effect |
|---|---:|---:|---:|---:|---:|---|
| tabular | 0.027 | 0.021 | 0.949 | 0.3505 | 0.359 | True |
| linear_raw | 0.014 | 0.011 | 1.072 | 0.2927 | 0.405 | True |

## Verdict (per pre-registered criteria in docs/cue_validity_cross_environment.md)
Criterion 1 (linear_raw shows a significant effect at >=1 non-floor validity level): True.
Criterion 2 (tabular shows no significant effect at either non-floor validity level): True.
Criterion 3 (floor control shows no spurious effect for either condition): True.
**Pattern confirmed (all three criteria met): True.**

## Limitations
This replication tests only the decorrelate intervention with tabular and linear_raw agents, by pre-registered design -- reversal, neural agents, and interaction-only encoding in this environment are explicitly deferred to a follow-up, contingent on this core pair replicating. A single grid size (5x5) and a single start/decision-cell layout are tested, not a sweep. The episode budget (5000 acquisition) was increased from the 3000 used elsewhere in this lineage to accommodate the added navigation-learning requirement; this was decided from a pre-run timing/learnability smoke test, not tuned post-hoc on this experiment's own persistence results.
