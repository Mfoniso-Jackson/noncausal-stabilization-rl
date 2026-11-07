# Cue Validity Shuffled/Placebo Control: Pre-Registration

Written and committed before any results from this experiment exist.
Item from the publication plan's Section 6 ("Recommended additions
before publication"): a stronger null than the `cue_validity=0.0` floor
control, which rules out any artifact specific to `decorrelate()`'s
mechanism (drawing a fresh i.i.d. Bernoulli(0.5) cue each post-
intervention episode) rather than to the absence of true cue information
per se.

## What this tests

`decorrelate()` makes the cue uninformative going forward by resampling
it fresh at p=0.5 every episode. This is mathematically uninformative
about `hidden_state` in expectation, but it is not the only way to
construct a sequence with zero cue-state correlation: a **permutation**
of the real, already-realized acquisition-phase cue sequence (same
empirical marginal frequency as the real high-validity cue stream, same
exact pseudo-random values that were actually used, just randomly
reassigned to episodes) also has zero correlation with `hidden_state` by
construction, without relying on a fresh random draw's own small-sample
behavior. If a pretrained agent's post-intervention persistence gap under
this **shuffle** control is statistically indistinguishable from its gap
under ordinary `decorrelate()`, that rules out "something about
`decorrelate()`'s specific resampling mechanism, not the mere absence of
cue information, drives the measured gap" as an alternative explanation.

## Design

- Condition: **`linear_raw` only** (the clearest, most-replicated effect
  in this lineage; this control's purpose is methodological, not to
  re-establish the finding across representations).
- New environment subclass, `ShuffleDecorrelateEnv(CueValidityEnv)`
  (defined in the experiment script, not `env/cue_validity_env.py` --
  no change to the shared environment module): logs the realized
  `cue_present` value at every acquisition-phase `reset()`, then, once
  `start_shuffle_phase(seed)` is called, overrides `cue_present` on every
  subsequent `reset()` by popping from a freshly-shuffled permutation of
  that logged sequence (reshuffling and refilling when exhausted) instead
  of drawing a new value via `validity_to_probs`. Everything else
  (`step()`, `observe()`, reward mechanics) is inherited unchanged.
- `cue_validity` in {0.5, 0.95}, 15 seeds, same episode budget as
  `cue_validity_proxy_dependence_experiment.py` (3000 acquisition, 1500
  post-intervention in 30 blocks of 50).
- Two arms: **shuffle** (acquisition, then shuffle-phase using that
  seed's own realized acquisition cue log) and **naive_shuffle** (fresh
  agent, trained for the same post-intervention budget directly against
  a shuffle-phase env whose pool comes from a separate acquisition run
  with the same validity and a different seed, so naive training never
  sees its own evaluation-time pool).
- `decorrelate`/`naive` arms are **not re-run**; this experiment reuses
  `results/cue_validity_proxy_dependence_summaries.csv`'s already-
  committed `linear_raw` results directly as the comparison baseline.

## Metrics

`proxy_dependence`, read via the same exact-probe method used throughout
this lineage.

## Success / failure criteria (fixed before data collection)

The placebo-mechanism concern is **ruled out** if the shuffle arm's
per-seed `|final - naive_shuffle_final|` gap is statistically
indistinguishable (`compare_groups`, p >= 0.05) from the already-
committed decorrelate arm's `|decorrelate_final - naive_final|` gap, at
each validity level. It is **not ruled out** (meaning `decorrelate()`'s
specific mechanism may itself be contributing to the measured effect,
beyond the mere absence of cue information) if the gaps differ
significantly. A null or small shuffle-arm gap that is *smaller* than
decorrelate's would be the more concerning direction and is reported as
such if found, not minimized.

## What would change this plan after the fact

Nothing above will change after results are inspected, following the
same discipline as every prior pre-registration in this lineage.
