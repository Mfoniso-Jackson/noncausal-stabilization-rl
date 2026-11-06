# Cue Validity Replay Buffer Size Sweep: Pre-Registration

Written and committed before any results from this experiment exist.
Item from the publication plan's Section 6: the replay-buffer-capacity
calculation elsewhere in this lineage ruled out data staleness as an
explanation for persistence at the one buffer size tested
(`REPLAY_BUFFER_SIZE=8000`); this sweep checks that exclusion holds
across other buffer sizes too.

## What this tests

If the decorrelate-vs-naive gap appeared only at a specific replay
buffer size, that would suggest the effect is about how much stale
(pre-decorrelation) experience survives in the buffer during the
post-intervention period, rather than genuine weight-level persistence.
A much smaller buffer flushes stale transitions far faster (fully
replaced within a handful of episodes); a much larger buffer retains
them far longer. If the effect holds across a wide range of buffer
sizes, staleness in this specific sense is not the explanation.

## Design

- Condition: **`linear_raw` only** (cheapest, most-established effect;
  this is a robustness check, not a new representation comparison).
- New agent wrapper, `ReplayBufferAgentWrapper` (defined in the
  experiment script): identical to
  `cue_validity_proxy_dependence_experiment.py::LinearAgentWrapper` in
  every respect except `replay_size`, swept over **{500, 2000, 8000,
  20000}** (8000 is the value used everywhere else in this lineage,
  included here for direct continuity rather than re-running it
  separately).
- `cue_validity` in {0.5, 0.95}, 15 seeds, same episode budget (3000
  acquisition, 1500 post-intervention in 30 blocks of 50).
- Arms: `decorrelate` and `naive`, same methodology as every prior
  experiment.

## Metrics

`proxy_dependence`, read via the same exact-probe method.

## Success / failure criteria (fixed before data collection)

The staleness exclusion **holds across buffer sizes** if the
decorrelate-vs-naive gap is significant (`compare_groups`, p < 0.05,
Cohen's d) at every one of the 4 buffer sizes, at both validity levels.
It **does not hold** if any buffer size shows no significant effect --
that buffer size would need to be flagged and investigated, not
smoothed over. A monotonic trend in effect size with buffer size (larger
buffer -> larger effect, or the reverse) is reported descriptively but
does not by itself overturn the exclusion as long as every size clears
the significance bar.

## What would change this plan after the fact

Nothing above will change after results are inspected, following the
same discipline as every prior pre-registration in this lineage.
