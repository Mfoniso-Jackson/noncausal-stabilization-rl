# Cue Validity Neural Reversal Experiment: Pre-Registration

Written and committed before any results from this experiment exist. Direct
continuation of two prior experiments in this lineage: read
`docs/cue_validity_reversal.md` (the gradient-asymmetry mechanism and the
reversal intervention) and `docs/cue_validity_nonlinear.md` (the neural
agent, `interaction_only` feature encoding, and the graded-persistence
result) first, along with both their reports.

## What this tests

Two things are known so far, from two separate experiments that never
overlapped:

1. The reversal experiment found that, for `linear_raw` and
   `linear_orthogonal` agents, an active reversal of the cue-state
   correlation produces faster proportional adaptation than a passive
   decorrelation (the gradient-asymmetry mechanism), once the comparison is
   normalized by the distance each intervention requires the agent to
   travel. `tabular` agents adapt fully under both interventions.
2. The nonlinear experiment found that decorrelate-vs-naive persistence is
   a *graded* property: `linear_interaction_only` and `neural_raw` each
   still show a significant, smaller effect than `linear_raw`; only
   `neural_interaction_only` (removing direct coordinate access **and**
   adding nonlinearity together) is statistically indistinguishable from
   the naive-control floor.

Neither experiment tested the other's manipulation: the reversal
experiment never used a neural agent or the interaction-only encoding, and
the nonlinear experiment never used reversal, only decorrelation. This
experiment closes that gap: does the gradient-asymmetry mechanism (reversal
adapts faster than decorrelation) hold for the three representations
introduced in the nonlinear experiment -- `linear_interaction_only`,
`neural_raw`, `neural_interaction_only` -- or is it specific to the raw
linear representations where it was first observed?

## Design

- Environment and intervention: `env/cue_validity_env.py::CueValidityEnv`,
  `cue_phase = target_phase = 4`, `env.reverse()` (not `decorrelate()`) as
  the post-acquisition intervention, mirroring
  `experiments/cue_validity_reversal_experiment.py` exactly.
- Conditions (3, all reused verbatim from
  `experiments/cue_validity_nonlinear_experiment.py`, which is imported
  rather than redefined): `linear_interaction_only` (dependency-free linear
  agent over 8 interaction-only features), `neural_raw` (2-hidden-layer
  ReLU MLP over the 6 raw features), `neural_interaction_only` (same MLP
  over the 8 interaction-only features). `tabular`, `linear_raw`, and
  `linear_orthogonal` are **not re-run** under reversal here; they were
  already tested in `cue_validity_reversal_experiment.py` and that
  comparison is not repeated.
- `cue_validity` in {0.5, 0.95}, 15 seeds, same episode budget as both
  prior experiments (3000 acquisition, 1500 post-intervention in 30 blocks
  of 50).
- For each (condition, validity): a **reversed** arm (acquisition, then
  `env.reverse()`, then continued training) and a **naive-reversed**
  control arm (fresh agent, trained only in an already-reversed environment
  for the same post-intervention budget), reusing
  `cue_validity_reversal_experiment.py`'s naive-control methodology exactly,
  but with `cue_validity_nonlinear_experiment.py`'s `make_agent`,
  `features_for`, `train_episode`, and `read_metrics` functions (so the
  agents, feature encodings, and metric-reading method are byte-identical
  to the nonlinear experiment's, differing only in the intervention).

## Metrics

Unchanged: `proxy_dependence` (primary, signed -- acquisition should be
positive, full adaptation to reversal means the sign flips negative) and
`CDS` (secondary), read via the same exact-probe method already used by
both parent experiments.

## Success / failure criteria (fixed before data collection)

Applying the general lesson learned in `docs/cue_validity_reversal.md`'s
addendum prospectively rather than repeating the same mistake: a raw
`|reversed_final - naive_reversed_final|` gap is not a fair comparison
against a decorrelation gap, because reversal's fully-adapted target (a
full sign flip) is a strictly larger distance from the acquisition value
than decorrelation's (~0). The **primary** criterion here is therefore the
normalized one from the start: per seed, the fraction of the required
change (acquisition value to the naive-reversed control's fully-adapted
value) achieved by the final block, compared against the same fraction
computed from the already-collected decorrelation data for the same
condition and validity, in `results/cue_validity_nonlinear_summaries.csv`.

The gradient-asymmetry mechanism is **supported**, per condition, if the
reversal fraction-adapted is significantly larger (independent two-sample
t-test across the 15 seeds of each experiment, p < 0.05, Cohen's d
reported) than the decorrelation fraction-adapted, at the same validity
level. It is **not supported** if the fractions are statistically
indistinguishable or if the decorrelation fraction is larger.

The raw `|gap|` (reversed vs. naive-reversed) is also computed and reported
for comparability with `cue_validity_reversal_experiment.py`'s table, but
is explicitly secondary and not used to decide support for the mechanism.

A condition with no decorrelation effect to begin with
(`neural_interaction_only` was statistically indistinguishable from the
naive-control floor under decorrelation) has no meaningful asymmetry to
test -- there is no persistent dependency for reversal to extinguish faster
than. This is noted as an expected, not a failed, outcome for that
condition specifically, and the fraction-adapted computation is still
reported for it for completeness.

A null result (no condition shows a significant asymmetry) is a valid,
reportable outcome.

## What would change this plan after the fact

Nothing in the metrics, conditions, or criteria above will change after
results are inspected, following the same discipline as the three prior
pre-registrations in this lineage.

## Addendum (added after the run below; the sections above are unchanged)

Result, by the pre-registered fraction-adapted criterion:
`linear_interaction_only` shows the asymmetry (reversal fraction-adapted
0.765/0.827 vs. decorrelation's 0.671/0.757, p=0.024/0.019, Cohen's d
0.90/0.94, both validity levels) -- the mechanism replicates in a third
linear representation. `neural_raw` and `neural_interaction_only` do not
reach significance (p=0.17/0.06 and p=0.41/0.77).

Inspecting the per-condition fractions shows why, and it is a ceiling
effect, not evidence against the mechanism: both neural conditions are
already adapting to **0.89-1.01** of the required change under
decorrelation alone (vs. `linear_interaction_only`'s 0.67-0.76), leaving
little headroom for reversal to show a detectably larger fraction via this
normalized metric. This is consistent with, not contrary to, the parent
nonlinear experiment's own finding that neural representations carry a
*smaller* decorrelate-vs-naive persistence effect than linear ones in the
first place -- there is less residual dependency left for reversal to
extinguish faster. `neural_raw`'s direction is still consistent with the
mechanism (reversal fraction numerically higher at both validity levels,
0.946-0.966 vs. 0.894-0.898, borderline at v=0.95, p=0.0615) but
underpowered at 15 seeds given how little variance remains once both arms
are this close to full adaptation. `neural_interaction_only` shows no
directional trend either way (p=0.41/0.77), exactly as the pre-registration
anticipated for a condition with no decorrelate-vs-naive effect to begin
with.

Net picture across both experiments in this sub-lineage: the
gradient-asymmetry mechanism and the graded-persistence finding are not
independent -- they interact. Representations with more room below the
adaptation ceiling (linear, raw-coordinate-accessible) show both larger
persistence under decorrelation *and* a clearer reversal-vs-decorrelation
asymmetry; representations closer to the ceiling (neural, or
interaction-only) show smaller persistence and a correspondingly harder
asymmetry to detect with the same seed count, not an absence of the
underlying mechanism. This addendum does not change any criterion above;
the verdict stands exactly as the pre-registered test computed it
(supported for `linear_interaction_only` only). A seed-count increase for
`neural_raw` specifically (closest to significance) is the natural
follow-up noted here but not undertaken as part of this experiment, per
the discipline of this lineage.
