# Cue Validity Nonlinear & Interaction-Only Representation: Pre-Registration

Written and committed before any results from this experiment exist.
Direct continuation of `docs/cue_validity_proxy_dependence.md`; read that
and `reports/cue_validity_proxy_dependence_report.md` first.

## What this tests

The proxy-dependence experiment found a large, persistent `proxy_dependence`
effect under the decorrelate intervention for linear agents (raw and
orthogonally-rotated features), absent under tabular representation. It
did not test whether the effect is specific to *linear* function
approximation, or whether it also appears under nonlinear (neural)
function approximation, and it did not test whether the effect requires a
raw, directly-accessible cue coordinate at all (the orthogonal rotation
tested coordinate rotation, not coordinate removal).

This experiment directly mirrors this repository's own established
methodology for exactly this question
(`linear_feature_ablation_experiment.py` ->
`orthogonalized_proxy_feature_experiment.py` ->
`nonlinear_interaction_proxy_experiment.py`): add an interaction-only
feature encoding with no raw cue coordinate at all (the cue appears only
multiplied with other features), and test it with both a linear agent and
a small neural network, crossed with the already-tested raw encoding for
the neural agent as a control.

## Design

- Environment, intervention, and validity levels unchanged:
  `env/cue_validity_env.py::CueValidityEnv`, `cue_phase = target_phase`,
  decorrelate intervention only (not reversal -- this experiment is about
  representation, not intervention type), `cue_validity` in {0.5, 0.95}.
- New feature encoding, `interaction_only`: the same base features as
  `observation_vector()` (phase_norm, target_flag, pre_target_flag, sin,
  cos) but with the raw `cue_bit` coordinate replaced by three interaction
  terms (`cue_signed * phase_norm`, `cue_signed * sin`, `cue_signed * cos`,
  where `cue_signed = 2*cue_bit - 1`) -- 8 features total, no term that is
  purely the cue bit alone, mirroring
  `orthogonalized_proxy_feature_experiment.py`'s `proxy_interaction_only`
  condition exactly.
- New agent/representation conditions (3, in addition to the three already
  reported): `linear_interaction_only` (the same dependency-free linear
  agent as before, over the interaction-only features), `neural_raw` (a
  small 2-layer ReLU MLP, PyTorch, over the raw 6 features), and
  `neural_interaction_only` (the same MLP, over the interaction-only
  features).
- `tabular`, `linear_raw`, and `linear_orthogonal` are **not re-run**; this
  experiment's comparisons reuse the already-committed
  `results/cue_validity_proxy_dependence_summaries.csv` for those three
  conditions directly.
- 15 seeds per new condition, matching the proxy-dependence experiment
  (a timing check confirmed a full-scale neural run completes in ~9
  seconds, so there is no cost reason to reduce seed count here). Same
  episode budget (3000
  acquisition, 1500 post-intervention in 30 blocks of 50) as the
  proxy-dependence experiment, for comparability.

## Metrics

Unchanged: `proxy_dependence` (primary) and `CDS` (secondary), read via
the same exact-probe method, extended to work against a feature-vector
function (linear or neural) instead of a tabular Q-table lookup, exactly
as the proxy-dependence experiment's linear agents were already probed.

## Success / failure criteria (fixed before data collection)

Mirroring `orthogonalized_proxy_feature_experiment.py`'s own criterion for
calling an effect representation-dependent: for each of `linear_raw`
(already reported) vs `linear_interaction_only`, and `neural_raw` vs
`neural_interaction_only`, the effect is **coordinate-dependent** (requires
an easily accessible raw coordinate) if the interaction-only condition's
decorrelate-vs-naive-control gap is significantly smaller (independent
two-sample t-test, p < 0.05, Cohen's d reported) than the raw condition's
gap, at the same validity level. It is **coordinate-independent** (the
information content alone is sufficient, regardless of how it is encoded)
if the gaps are statistically indistinguishable. Separately,
`neural_raw` vs `linear_raw` tests whether nonlinearity changes the size
of the effect at all (expected: comparable or larger, since a neural
network can always approximate a linear function).

A null result (no condition shows a decorrelate-vs-naive gap) is a valid,
reportable outcome, as is a result where the interaction-only encoding
kills the effect for both linear and neural agents (which would mean the
proxy-dependence experiment's positive finding was itself coordinate-
dependent, the same conclusion this repository's own
`orthogonalized_proxy_feature_experiment.py` ->
`nonlinear_interaction_proxy_experiment.py` chain reached for a different
environment).

## What would change this plan after the fact

Nothing in the metrics, conditions, or criteria above will change after
results are inspected, following the same discipline as the three prior
pre-registrations in this lineage.
