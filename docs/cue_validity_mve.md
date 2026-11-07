# Cue-Validity MVE: Pre-Registration

Written and committed before any data from this experiment exists, per the
research brief's leakage-prevention requirement. If anything below needs to
change after seeing results, that change will be made in a separate,
dated commit that says so explicitly, rather than edited in place.

## Research question

Does a reinforcement-learning agent develop a behavioural dependency on a
cue that is predictive but never causal during acquisition, and does that
dependency persist, more than a matched floor control's, after an
intervention decorrelates the cue from the hidden state it used to track?

## Why this design, and what it deliberately does not test

The full audit (see conversation history / repo `reports/` for prior
experiments) found that every existing experiment in this repository
re-defines "CDS" and "SPI" ad hoc, inconsistently, per script, and that the
three pre-existing "headline" pipeline scripts (`run_pipeline.py`,
`run_mechanism_isolation.py`, `run_factorial_credit_assignment.py`) share an
exploration-bias confound (`agents/dqn_agent.py`'s `cue_exploration_bias`,
set as a deterministic function of the manipulated uncertainty variable)
that makes their "uncertainty increases cue dependence" findings close to
circular. This MVE is built as a new, shared module
(`env/cue_validity_env.py`, `metrics/cue_validity.py`) specifically to avoid
repeating either problem: one environment, one CDS/SPI definition, no
agent-side bias tied to the manipulated variable.

This MVE tests the **decorrelate** intervention only (the cue's predictive
value is withdrawn; P(cue=1) reset to 0.5 regardless of hidden state,
nothing else changes). It deliberately does not test reversal (already
covered by `experiments/spurious_correlation_reversal_experiment.py`) or
removal of the cue-associated action (already covered by
`experiments/distractor_intervention_experiment.py`). It uses a tabular
SARSA(lambda) agent only; a function-approximation follow-up is explicitly
out of scope here and is the first item in "Planned follow-ups" below.

Four prior interventions already run in this repository this session
(distractor removal, spurious-action reversal, reward-delay manipulation,
memory-gap/belief-state supervision) all found **no evidence of
intervention-resistant persistence**. The prior for this experiment is
therefore that the effect is probably absent or small, not that it is
probably present. This experiment is designed to be sensitive enough to
detect a real but modest effect, not merely to confirm a large one it
already expects to find.

## Design

- Environment: `env/cue_validity_env.py::CueValidityEnv`. Hidden state ~
  Bernoulli(0.5) per episode, never observed. Reward only if the causal
  action is chosen at `target_phase` and hidden_state == 1. A single cue is
  shown once, at `cue_phase`, with P(cue=1 | hidden_state) set by
  `cue_validity` via `validity_to_probs`. The cue and the cue-associated
  action never affect reward, transitions, or action availability, in any
  condition, at any point -- this is fixed for the whole experiment and is
  not altered by the intervention.
- Conditions (5 total): `cue_validity` in {0.0 (floor control), 0.5, 0.95},
  crossed with intervention in {none, decorrelate} for the two non-zero
  validity levels (the floor control has nothing to decorrelate). 30 seeds
  each.
- Acquisition: 3000 episodes at the assigned `cue_validity`. Intervention
  conditions then call `env.decorrelate()` and continue training online for
  1500 further episodes, in 50-episode blocks, exactly as in
  `spurious_correlation_reversal_experiment.py` and
  `ambiguous_causality_memory_experiment.py` earlier this session.
- Agent: tabular SARSA(lambda), identical hyperparameters to the other
  tabular experiments in this repository this session (alpha=0.08,
  gamma=0.95, lambda=0.80, epsilon 1.0 -> 0.04).

## Metrics (single definitions, `metrics/cue_validity.py`)

- **CDS** (Cue Dependence Score): at the cue-exposed decision state, the
  fraction of the agent's action-value mass (softmax over available
  actions) placed on ACTION_CUE plus the neutral actions, i.e. the
  non-causal mass -- identical in spirit to the SPI definitions already
  validated in this session's spurious-correlation and ambiguous-causality
  experiments, kept here as one canonical, tested function instead of a
  per-script reimplementation.
- **SPI** (persistence index): post-intervention block CDS divided by the
  pre-intervention (acquisition-end) CDS for that seed, clamped to
  [0, inf); also reported as an extinction half-life in blocks (first
  post-intervention block whose CDS falls to half the acquisition-end
  value), matching the convention already used successfully in this
  session's reversal and memory-gap experiments.
- Both are implemented once, unit-tested in `tests/cue_validity_metrics_test.py`
  and `tests/cue_validity_env_test.py`, and imported by the experiment
  script -- not redefined there.

## Success / failure criteria (fixed before data collection)

The hypothesis is **supported** by this MVE only if, for at least one
non-zero validity level:
1. The decorrelate condition's post-intervention CDS is significantly
   higher (independent two-sample t-test on per-seed values, p < 0.05,
   with the Cohen's d reported) than a **matched naive-history-free
   control** run fresh in the already-decorrelated environment for the
   same number of post-intervention episodes (isolating "prior acquisition
   history matters" from "this is just how fast any agent extinguishes a
   useless cue in this environment," exactly as in
   `distractor_intervention_experiment.py`'s design), AND
2. the floor control (`cue_validity=0.0`) does not itself show the same
   gap (ruling out an artifact of the environment or metric alone).

The hypothesis is **not supported** if neither validity level clears both
bars, or if the floor control shows a comparable gap (indicating a metric
or environment artifact rather than a genuine validity-dependent effect).

A **null or small result here is a valid, reportable outcome** and will be
written up as such, not re-run with different parameters until a positive
result appears.

## What would change this plan after the fact

Nothing in the metric definitions, success criteria, or condition list
above will be changed after results are inspected. If a bug is found in
`env/cue_validity_env.py` or `metrics/cue_validity.py` itself (not in the
interpretation of their output), the fix will be committed separately with
an explicit note that re-running was required, and the pre-registered
criteria above will still apply to the corrected run.

## Addendum (added after the run below; the sections above are unchanged)

The run reported in `reports/cue_validity_mve_report.md` found that
acquisition-end CDS never differs from the exact tie-breaking floor at any
`cue_validity` level, including 0.95 -- a structural property of this
environment (`cue_phase` and `target_phase` are separate decision points,
so no action chosen at `cue_phase` can affect the bootstrapped return), not
a bug in the code relative to its own stated invariants. The
pre-registered success/failure criteria above were evaluated exactly as
written and are unaffected; the hypothesis is not supported by this run,
and more specifically the intervention-persistence question could not be
meaningfully tested at all under this particular environment structure.
See the report's "Why acquisition itself shows no differentiation" section
for the full mechanistic account. This addendum does not change any
criterion above, and no part of this document was edited after the run's
results were inspected.

## Correction to the addendum above

The structural fix proposed above (fold `cue_phase` into `target_phase`)
was checked empirically and does not work, for a different and more
general reason than the one diagnosed above: see
`docs/tabular_rl_cannot_be_superstitious.md`. Folding the cue into the
decision state makes CDS *decrease* as `cue_validity` increases (checked:
0.58 / 0.55 / 0.53 at validity 0.0 / 0.5 / 0.95), because the cue-associated
action never yields reward at any validity level, so a more informative
cue only sharpens the (correct) preference for the causal action -- this
tests rational cue exploitation, not non-causal dependence. The note above
generalizes this into a structural claim about tabular RL as a whole and
corrects the report's "Next recommended experiment" section accordingly
(superseded by the function-approximation / state-aliasing direction in
that note, not the fold proposed here). This correction is itself dated
and additive, consistent with this document's own rule against editing
settled criteria after the fact.

## Planned follow-ups (explicitly out of scope for this MVE)

1. Repeat with a function-approximation agent (DQN, reusing this session's
   PyTorch infrastructure) once the tabular result is in.
2. If decorrelation shows an effect, test whether it compounds with lower
   cue_validity (a full validity x intervention interaction rather than
   just two points).
3. Reconcile with the four already-completed Family-C interventions in a
   combined write-up once this MVE's result is known.
