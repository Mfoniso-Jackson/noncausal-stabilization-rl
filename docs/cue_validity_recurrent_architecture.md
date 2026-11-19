# Cue Validity Recurrent Architecture Comparison: Pre-Registration

Written and committed before any results from this experiment exist.
Direct response to the "Research State Audit" (this session) item 6:
retest a GRU-based recurrent architecture, which previously existed only
in `experiments/neural_recurrent_reversal_experiment.py` under
hyperparameters that did not match that script's own `linear_dqn` baseline
(verified directly: `linear_dqn` used `lr=0.015, batch_size=32`;
`neural_dqn`/`recurrent_dqn` used `lr=1e-3, batch_size=64,
replay_buffer_size=50000, target_update=200`), and on the pre-correction
environment design (separate `cue_phase`/`target_phase`, shown in
`docs/tabular_rl_cannot_be_superstitious.md` to structurally prevent any
agent from showing the effect for reasons unrelated to architecture).

## What this tests

Whether a GRU-based recurrent agent shows a different magnitude or pattern
of non-causal persistence than `linear_raw`/`neural_raw`, when every
hyperparameter that can be held constant across architectures is held
constant, on this session's corrected environment design, using this
project's established naive-control and Holm-Bonferroni methodology. This
extends H4 ("different representations and algorithms exhibit different
patterns") to a fourth architecture, the one previously tested only under
confounded conditions.

## Design

- Environment: `env/cue_validity_env.py::CueValidityEnv`, `cue_phase =
  target_phase = 4`, identical to every condition in
  `cue_validity_proxy_dependence_experiment.py` /
  `cue_validity_nonlinear_experiment.py`.
- Agent/representation conditions (3): `linear_raw` and `neural_raw` are
  rerun fresh in this script (not pulled from prior CSVs) so every
  condition in this experiment's own comparison comes from the exact same
  script, environment instantiation, and seed stream; this avoids any
  cross-script version drift as a possible confound. `recurrent_gru` is new.
- **Matched hyperparameters, held identical across `neural_raw` and
  `recurrent_gru`** (the one prior script's confound this experiment exists
  to fix): `gamma=0.95`, `lr=1e-3` (Adam), `batch_size=32`,
  `replay_buffer_size=8000`, `target_update_interval=200`,
  `train_every_steps=4`, hidden size 32, same epsilon schedule (1.0 to 0.04,
  annealed over acquisition; fixed 0.04 during intervention), same episode
  budget (3000 acquisition, 1500 intervention), same 15 seeds. These are
  exactly `cue_validity_nonlinear_experiment.py`'s established `neural_raw`
  settings, carried over unchanged.
- **The one necessary architecture-specific addition**: `recurrent_gru`
  processes a rolling window of the last 5 per-step observation vectors
  (`sequence_length=5`, matching the precedent in
  `neural_recurrent_reversal_experiment.py`) through an `nn.GRU`, rather
  than the single current-step vector an MLP takes. This is a structural
  requirement of using a recurrent network at all, not a tunable
  hyperparameter being set differently by choice; it is stated here
  explicitly so it is not mistaken for an unexamined mismatch the way the
  prior script's optimizer settings were.
- `tabular` is not rerun here; its null result is already extremely
  well-established across this entire project (most recently
  `cue_validity_proxy_dependence_criteria.csv`) and is referenced by
  citation, not retested, to keep this experiment's family focused on the
  actually-new question (does recurrence change the function-approximation
  picture).
- `cue_validity ∈ {0.5, 0.95}` (the two already-established, best-powered
  validity levels; no new floor arm, since the floor result is already
  established by `cue_validity_proxy_dependence_experiment.py` and is not
  specific to architecture).
- Arms: `decorrelate` (acquisition, then `env.decorrelate()`, continued
  training) and `naive` (fresh agent, decorrelated environment only, same
  post-intervention budget), the same naive-control methodology used
  throughout this project.

## Metrics

`proxy_dependence` (primary), `cue_dependence_score` (secondary), via the
same exact-probe method already validated in this lineage. For
`recurrent_gru`, the probe constructs the same rolling-window input the
agent was trained on (the most recent `sequence_length` observations,
padded by repeating the probed observation when fewer than `sequence_length`
steps have occurred, matching `neural_recurrent_reversal_experiment.py`'s
own precedent for this).

## Success / failure criteria (fixed before data collection)

Each of the 12 cells (3 agent types x 2 validity levels x... collapsed to
6 decorrelate-vs-naive comparisons, one per agent type x validity) tested
individually (`compare_groups`, p < 0.05, decorrelate magnitude greater
than naive), Holm-Bonferroni corrected across all 6 as one family. `recurrent_gru`
is judged against the same bar as `linear_raw`/`neural_raw`: significant
and correction-surviving counts as showing the pattern; non-significant
counts as not. No directional prediction is made in advance for whether
recurrence should increase, decrease, or leave unchanged the magnitude
relative to `neural_raw`, since the corrected environment gives the GRU no
functional need for memory (the cue is already folded into the decision
state) -- this experiment treats "recurrence given no functional need for
it" as the question itself, not as a confound to control away.

## What would change this plan after the fact

Nothing above will change after results are inspected, following the same
discipline as every prior pre-registration in this project.

## Result

**`recurrent_gru` shows the pattern cleanly, and with a slightly larger
effect size than `neural_raw` at both validity levels.** All 6 cells
survive Holm-Bonferroni correction (worst case holm p = 0.011).
`linear_raw`'s rerun reproduces its originally-published numbers (from
`cue_validity_proxy_dependence_experiment.py`) to the fourth decimal place
(d = 12.180046 vs. 12.180046, d = 23.589352 vs. 23.589352), confirming the
rerun is a faithful, deterministic reproduction and that this experiment's
`neural_raw`/`recurrent_gru` comparison is directly comparable to it.

| condition | v=0.5 Cohen's d | v=0.95 Cohen's d |
|---|---:|---:|
| `linear_raw` | 12.18 | 23.59 |
| `recurrent_gru` | 1.14 | 2.03 |
| `neural_raw` | 1.05 | 1.62 |

**One caveat worth stating plainly rather than overlooking**:
`recurrent_gru`'s naive-control floor itself is elevated relative to
`neural_raw`'s (0.0229 vs. 0.0085 at v=0.5; same at v=0.95 since the naive
arm never sees `validity`, matching the established pattern seen
elsewhere in this project where the naive arm's trajectory depends only
on the decorrelated environment and total training, not on the nominal
validity parameter). Both arms use the identical training protocol within
this experiment (same epsilon schedule, same episode budget), so this is
not the unequal-exploration-schedule artifact found and fixed in
`docs/cue_validity_grid_neural_reversal_interaction.md`'s addendum; it is
more likely an intrinsic property of the GRU's windowed input giving it a
higher resting `proxy_dependence` reading even with no real persistence to
measure, plausibly because repeating the 5-step window for every probe
compounds whatever small, architecture-specific instability a GRU head has
relative to a plain MLP head. The decorrelate-vs-naive *gap* is still
genuinely larger for `recurrent_gru` than for `neural_raw` at both
levels, not merely a uniform upward shift masking an unchanged gap, but
this elevated floor is relevant context for how confidently "recurrence
modestly increases persistence relative to a plain MLP" should be read,
and is reported rather than smoothed over.

**Net reading**: recurrence is not a disqualifying or synergistic factor
for this phenomenon; a GRU-based agent shows the same qualitative pattern
as every other function-approximation agent tested in this project, at a
magnitude between `neural_raw`'s and `linear_raw`'s, closer to
`neural_raw`'s. This extends H4 to a fourth architecture and closes the
"recurrent architecture only tested under confounded conditions" gap
identified in the Research State Audit and
`docs/symbolic_stabilization_audit.md`.
