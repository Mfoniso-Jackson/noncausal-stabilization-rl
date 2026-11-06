# neural_raw Validity Sweep: Pre-Registration

Written and committed before any results from this experiment exist.
Direct continuation of the post-hoc observation flagged, but explicitly
not confirmed, in `docs/cue_validity_neural_reversal.md`'s seed top-up
addendum: read that section first, along with
`reports/cue_validity_neural_raw_topup_report.md`.

## What this tests

The seed top-up found that `neural_raw` shows the gradient-asymmetry
effect (reversal adapts a significantly larger fraction of the required
change than decorrelation) at `cue_validity=0.95` but not at
`cue_validity=0.5`, even at 30 seeds. A plausible but un-pre-registered
reading was offered: validity may set how much headroom remains below the
adaptation ceiling (lower validity -> weaker original association -> less
residual persistence under decorrelation -> less room for reversal to show
a larger fraction), so the asymmetry becomes detectable only once validity
is high enough to leave meaningful headroom. This experiment tests that
specific claim directly, for `neural_raw` only, by filling in the gap
between the two already-tested validity levels.

## Design

- Condition: **`neural_raw` only**. `linear_interaction_only` already
  showed the asymmetry at both tested levels and `neural_interaction_only`
  showed no persistence to begin with; neither motivates a validity sweep
  here.
- New `cue_validity` levels: **0.6, 0.75, 0.9** (3 new points between the
  two already-run levels). Combined with the existing 0.5 and 0.95 points,
  this gives a 5-point curve.
- For each new level: all four arms already defined for `neural_raw` in
  this lineage, reusing the exact same code --
  `experiments/cue_validity_nonlinear_experiment.py`'s `run_pretrained_arm`
  / `run_naive_arm` (decorrelate intervention, for the persistence/headroom
  side) and `experiments/cue_validity_neural_reversal_experiment.py`'s
  `run_reversed_arm` / `run_naive_reversed_arm` (reversal intervention).
- 15 seeds (0-14) per arm per new validity level, matching this lineage's
  standard seed count. Same episode budget as every prior experiment in
  this lineage (3000 acquisition, 1500 post-intervention in 30 blocks of
  50).
- The existing 0.5 and 0.95 points are **not re-run**: they reuse the
  already-committed, most-powered data -- `neural_raw`'s decorrelate/naive
  arms from `results/cue_validity_nonlinear_summaries.csv` (15 seeds) and
  its reversed/naive_reversed arms from the merged 30-seed
  `results/cue_validity_neural_raw_topup_summaries.csv` -- directly, giving
  a 5-point curve with uneven per-point seed counts (15, 15, 15, 15, 30)
  that is disclosed as such in the report rather than smoothed over.

## Metrics

Per validity level:
- **Headroom**: `1 - decorrelation_fraction_adapted` (how far decorrelation
  alone falls short of full adaptation; 0 means decorrelation already
  fully adapts, leaving nothing for reversal to improve on).
- **Asymmetry effect**: `reversal_fraction_adapted - decorrelation_fraction_adapted`
  (signed; positive means reversal adapts a larger fraction than
  decorrelation, consistent with the mechanism).
- Both computed exactly as in `cue_validity_neural_reversal_experiment.py`
  (`per_seed_fraction_adapted_reversal` / `_decorrelation`), then averaged
  per seed-matched level.

## Success / failure criteria (fixed before data collection)

**Primary** (directly testing the headroom hypothesis): the Pearson
correlation, across the 5 validity levels, between headroom and asymmetry
effect. The headroom hypothesis is **supported** if r > 0 with p < 0.05;
**not supported** otherwise.

This is explicitly flagged in advance as a severely underpowered test --
n=5 points, each itself an aggregate of a noisy 15-or-30-seed estimate --
and is reported as exploratory/descriptive regardless of outcome, not as a
confirmatory statistical result. A non-significant or even negative-r
result is a valid, reportable outcome and would mean the apparent
validity-dependence seen at the two original endpoints does not hold up as
a clean monotonic relationship once the gap between them is examined.

**Secondary** (reported for transparency, not for deciding the primary
hypothesis): the same per-level asymmetry significance test used
throughout this lineage (`compare_groups`, p < 0.05), reported at each of
the 3 new validity levels, to see whether any of them individually cross
the significance threshold and, if so, whether that lines up with where
headroom is largest.

## What would change this plan after the fact

Nothing in the metrics, conditions, or criteria above will change after
results are inspected, following the same discipline as every prior
pre-registration in this lineage.
