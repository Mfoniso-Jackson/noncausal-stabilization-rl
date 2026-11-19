# Multi-Symbol Cue Validity Environment: Design Rationale

This is a design/capability document, not an experiment pre-registration:
no experiment has been run on this environment yet. It documents the
environment itself, built and tested, in response to
`docs/symbolic_stabilization_audit.md` Section 4's proposed computational
definition of symbolic stabilization, which this project's existing
environments could not express.

## Why a new class, not a change to `CueValidityEnv`

`CueValidityEnv` is constructed directly by dozens of already-validated
experiments across this project. Modifying it to support multiple symbols
would risk regressing every one of them for no benefit (none of them need
more than one symbol). `env/multi_symbol_cue_validity_env.py::MultiSymbolCueValidityEnv`
is a new, independent class; `CueValidityEnv` is untouched.

## What the single-cue design could not express

`docs/symbolic_stabilization_audit.md` Section 3 distinguished "non-causal
stabilization" (already proven: persistence scales with a cue's own
validity) from a stricter "symbolic stabilization" claim: does persistence
depend on *which symbol* was invalidated, not just on that symbol's own
validity? Section 4 set the minimum bar: at least two independently-
trackable symbols, each with its own independently-set validity, present
concurrently in the same environment and the same agent's training. A
single-cue environment cannot even pose this question, regardless of how
it is analyzed.

## Design

- One shared hidden state (`hidden_state`, Bernoulli(0.5), never observed),
  identical in meaning to `CueValidityEnv`'s: the causal mechanism (one
  causal action, correct iff `hidden_state == 1`) is unchanged. Only the
  *non-causal* side changes: instead of one cue, there are `N >= 1`
  independent per-episode Bernoulli cues, each with its own validity
  `cue_validities[i]`, via the same `validity_to_probs` formula
  `CueValidityEnv` uses (duplicated rather than imported, so the two
  environments have zero coupling).
- Each symbol has its own fixed, dedicated, never-rewarding action
  (`cue_action(i)`), the same role `ACTION_CUE` plays for the single-cue
  case, generalized to N of them. Action space:
  `(ACTION_CAUSAL,) + (cue_action(0), ..., cue_action(N-1)) + neutral_actions`
  (two neutral actions, same convention as `CueValidityEnv`).
- **The one genuinely new capability**: `decorrelate(symbol_id)` and
  `reverse(symbol_id)` intervene on *one* symbol, leaving every other
  symbol's validity and intervention state untouched (`symbol_id=None`
  intervenes on all symbols at once, recovering `CueValidityEnv`'s
  single-symbol behavior as the `N=1` special case). This is what makes a
  "symbolic stabilization" experiment possible: set two symbols to the
  same validity during acquisition, decorrelate only one of them, and ask
  whether that symbol's own `proxy_dependence` differs from what its
  validity alone would predict (already established, single-cue) once the
  *other* symbol's continued validity is accounted for.
- `reward_noise` is carried over unchanged from `CueValidityEnv` (same
  mechanism: withholds an earned causal reward with the stated
  probability), so this environment can also serve as the natural next
  step flagged in `docs/cue_validity_uncertainty_sweep.md`'s own result
  ("whether other forms of uncertainty... produce a different pattern"),
  without having to build that capability twice.
- `observation_vector()` generalizes `CueValidityEnv`'s 6-feature encoding
  to `N + 5` features (`phase_norm`, `N` cue bits, `target_flag`,
  `pre_target_flag`, `sin_phase`, `cos_phase`); `feature_dim` exposes this
  directly rather than requiring callers to compute it.

## Metric reuse: zero new metric code

`metrics/cue_validity.py::proxy_dependence` and `cue_dependence_score` are
used completely unmodified, confirmed directly in
`tests/multi_symbol_cue_validity_env_test.py::test_proxy_dependence_and_cds_reused_unmodified_per_symbol`
with a hand-built Q-function, not just asserted: both functions take
explicit action arguments and make no assumption about total action
count. Probing symbol `i`'s own effect means calling them once with
`cue_action = cue_action(i)` and every other action (every other symbol's
cue action, plus the two neutral actions) folded into `neutral_actions`
for that specific probe. The exact-probe methodology already validated
throughout this project (fix `phase = target_phase`, set each symbol's
`cue_present` bit directly, read Q-values) extends unchanged: just set
`N` bits instead of one before probing.

## Testing

`tests/multi_symbol_cue_validity_env_test.py`, 15 tests: per-symbol
validity independence (including a 3-symbol case with three different
validities), that `decorrelate`/`reverse` on one symbol leaves every other
symbol's statistics untouched, that `symbol_id=None` affects all symbols,
boundary/error handling (zero symbols, out-of-range symbol_id), that no
cue action ever earns reward regardless of symbol count or validity, that
causal reward depends only on `hidden_state`, observation vector shape
and cue-bit correctness (including that bits are zero outside
`cue_phase`), `reward_noise` carrying over correctly, and the metric-reuse
confirmation above. All pass; full existing test suite re-run alongside
them with zero regressions.

## What this document does not do

It does not run a symbolic-stabilization experiment. The natural next
step -- an actual pre-registered persistence-by-symbol-identity test,
following this project's standing discipline -- is proposed but not
started, consistent with `docs/symbolic_stabilization_audit.md`'s own
stated preference for not adding capability ahead of a concretely
scoped experiment that will use it.
