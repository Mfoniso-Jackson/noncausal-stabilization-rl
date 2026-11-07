# Cue Validity Capacity Sweep: Pre-Registration

Written and committed before any results from this experiment exist.
Item from the publication plan's Section 6: `cue_validity_nonlinear_experiment.py`
tested exactly one MLP architecture (2 hidden layers, 32 units); this
turns that single data point into a proper capacity curve.

## What this tests

Whether `neural_raw`'s decorrelate-vs-naive persistence gap (already
confirmed significant at the one architecture tested) scales smoothly
with model capacity, consistent with the "graded, not binary" framing
this lineage uses throughout, or whether it is a non-monotonic/threshold
function of width that the single-architecture test could not reveal.

## Design

- Condition: **`neural_raw` only**, decorrelate intervention only (not
  reversal), matching the scope of the original nonlinear experiment this
  extends.
- New agent wrapper, `CapacityAgentWrapper` (defined in the experiment
  script): identical to
  `cue_validity_nonlinear_experiment.py::NeuralAgentWrapper`/`MLPQNet` in
  every respect except `hidden_size`, which is swept over **{8, 16, 32,
  64, 128}** (32 is the already-tested value, included here for direct
  continuity with the existing result rather than re-running it
  separately).
- `cue_validity` in {0.5, 0.95}, 15 seeds, same episode budget (3000
  acquisition, 1500 post-intervention in 30 blocks of 50) as the
  nonlinear experiment.
- Arms: `decorrelate` and `naive`, same methodology as every prior
  experiment in this lineage.

## Metrics

`proxy_dependence`, read via the same exact-probe method.

## Success / failure criteria (fixed before data collection)

Report, for each of the 5 widths at each validity level, whether the
decorrelate-vs-naive gap is significant (`compare_groups`, p < 0.05,
Cohen's d). The capacity relationship is characterized as:
- **monotonic non-increasing** if the gap's magnitude (or at minimum its
  Cohen's d) does not increase as width increases from 8 to 128 (allowing
  ordinary seed-to-seed noise, i.e. judged by eye against the ci95, not a
  strict inequality at every adjacent pair);
- **non-monotonic/threshold** if it is not monotonic -- e.g. peaks at an
  intermediate width, or shows a sharp drop between two adjacent widths
  rather than a smooth decline;
- **flat** if effect size is statistically indistinguishable across all
  5 widths.

All three are valid, reportable outcomes; none would overturn any
existing finding, since this is a within-`neural_raw` characterization,
not a new comparison against `tabular`/`linear_raw`.

## What would change this plan after the fact

Nothing above will change after results are inspected, following the
same discipline as every prior pre-registration in this lineage.
