# Symbolic Stabilization: Research Architecture Audit

Written in response to a research directive proposing "symbolic stabilization"
as the next phase of this project. Per that directive's own Section 8
("Before changing code: inspect, trace, identify weaknesses"), this is an
audit and a proposal, not an implementation. No code has been changed.

## 0. First finding: the directive's own preliminary-results table does not exist in this repository

The directive cites a table:

| condition | CDS |
|---|---:|
| Baseline | 0 |
| Medium | ~0.116 |
| High | ~0.108 |

No file in this repository contains these exact numbers under these exact
condition labels. This was checked directly (`grep` across every `.md` and
`.csv` file), not assumed. The closest real data is in two files that predate
this session's work:

- `outputs/factorial_report.md`: `high_delay_low_sparsity` CDS = 0.107 ± 0.089,
  `high_delay_high_sparsity` CDS = 0.119 ± 0.072, `low_delay_low_sparsity`
  ("baseline") CDS = 0.039 ± 0.031 (**not** 0).
- `outputs/experiment_pipeline_summary.md`: a `baseline` row with CDS =
  0.000 ± 0.000 exactly (this part of the directive's table does match), but
  its `medium` (0.059) and `high` (0.038) rows do not match "~0.116" / "~0.108"
  at all, and `high < medium` there, the opposite of a monotonic story.

**This matters beyond pedantry.** The directive's Section 9 explicitly
instructs "do not optimize metrics simply to produce positive results" and
"do not assume uncertainty increases non-causal stabilization." Starting from
a results table that does not match any committed data would be building the
next phase of this research program on an unverified premise, exactly the
failure mode this project has spent this entire session building tooling
to prevent. The real, committed numbers are reported in full below instead.

## 1. What already exists: a second, earlier experimental lineage

This repository contains two largely separate experimental lineages. This
session's work (documented in `docs/research_report.md`) built and
exhaustively tested the `CueValidityEnv`/`CueValidityGridEnv` lineage: one
binary cue, one hidden state, swept validity, naive/floor/placebo controls,
pre-registration, Holm-Bonferroni correction throughout.

A second, earlier lineage already existed before this session and was not
previously audited in full: `env/grid_cue_env.py` (`GridCueEnv`), driven by
`experiments/run_pipeline.py`, `experiments/run_mechanism_isolation.py`, and
`experiments/run_factorial_credit_assignment.py`, with outputs in
`outputs/*.md` and `runs/*.md`. This is the actual source of `CDS` and `SPI`
as general-purpose, shared metric names (`metrics/cue_validity.py`'s own
docstring: *"every other experiment script in this repository reimplements
its own ad hoc version of 'CDS' and 'SPI'... these definitions are meant to
be imported, not copied"* — this session's `CueValidityEnv` lineage already
inherited and cleaned these two metric names from this older lineage).

### 1.1 `GridCueEnv`'s cue system

`env/grid_cue_env.py` + `env/cues.py`: a 2-D grid with `num_cues` (default 5
in these experiments) randomly-placed cue markers, each contributing one bit
to the state vector. This is closer to "distractor markers scattered in
space" than genuine discrete symbols with trackable identity (a "symbol" in
the directive's sense, e.g. color A vs. color B, each independently
trackable) — there is no `symbol_id`, label, or identity field distinguishing
one cue from another beyond grid position. Grep confirms: the string
"symbolic" does not appear anywhere in this repository's code. No explicit
symbolic representation layer exists, implicitly or explicitly, in either
lineage.

### 1.2 SPI: already defined, already in this session's manuscript under a different name

`metrics/cue_validity.py::persistence_index` **is** SPI (its docstring says so
explicitly: `"""SPI: post-intervention CDS relative to pre-intervention
CDS."""`). This session's manuscript (`docs/research_report.md`, Section 6)
already documents this exact function under the name `persistence_index`,
with its real limitation stated: it returns `None` (not a silently-wrong
zero) when the pre-intervention baseline CDS is too small to normalize by,
and the manuscript notes it is "descriptive; not used as a primary decision
criterion in this report" — because `proxy_dependence`, not a CDS-ratio
metric, is what this session's entire positive result (Cohen's d up to 23.6)
is built on. SPI/CDS and `proxy_dependence` are not interchangeable, and the
directive is right to ask that these be reviewed before being treated as
equivalent to anything this session already validated.

### 1.3 CDS's known structural limitation applies here too

Section 10.1 of this session's manuscript already proved, against
`distractor_action_experiment.py`'s own numbers, that CDS has a hard
structural ceiling under tabular TD learning: a never-rewarded action cannot
acquire stable value, so CDS cannot exceed the softmax tie-breaking floor
`(k-1)/k` regardless of true dependence. `docs/tabular_rl_cannot_be_superstitious.md`
(written earlier in this session, before the work summarized in the
manuscript) independently derives the same conclusion from first principles
and traces it through three more pre-existing scripts
(`linear_feature_ablation_experiment.py`, `orthogonalized_proxy_feature_experiment.py`,
`nonlinear_interaction_proxy_experiment.py`). Every CDS value in
`outputs/*.md` (0.04-0.12 range) is small enough to be plausibly within
tie-breaking-floor noise for a `DQNAgent`, not necessarily evidence of real
dependence. This needs to be checked against a `cue_validity=0`-style floor
control before any of these numbers are treated as a positive result, and
the directive's own Section 7 agrees ("Review the existing CDS and SPI
metrics before proposing replacements... establish appropriate controls").

## 2. The decisive finding: the existing uncertainty-sweep pipeline does not test RQ3, because it is confounded by construction

This is the single most important finding in this audit, and it directly
answers the directive's Section 8 question 7 ("Determine whether the current
experiments genuinely test the central hypothesis").

`experiments/run_pipeline.py::train_agent` does this:

```python
cue_bias = cue_bias_from_uncertainty(env_config)
agent = DQNAgent(
    env.state_size, env.action_size, seed=seed,
    cue_exploration_bias=cue_bias,
    cue_regularization=0.025 * (1.0 - cue_bias),
)
```

`cue_bias_from_uncertainty` computes a value from 0 to 0.85 that **is**
the experiment's own definition of "uncertainty" (a blend of sparsity, delay,
and noise). That same number is then fed directly into the agent as two
active training-time biases (`agents/dqn_agent.py`):

- `cue_exploration_bias`: on each action selection, with probability equal to
  this value, the agent is forced to act on the cue-associated action,
  regardless of its own learned policy.
- `cue_regularization`: every training step, this shrinks the weights at the
  non-core feature indices (`[4, state_size)`, i.e. the cue-related features)
  for **every** action's weight row, by a factor proportional to
  `(1 - cue_bias)` — so regularization pressure against cue-weight growth is
  *strongest* exactly when the experiment's own "uncertainty" label is
  *lowest*, and weakest when uncertainty is highest.

**This means the pipeline does not measure whether uncertainty causes an
agent to rely more on a non-causal cue as an emergent property of learning
under uncertainty.** It mechanically forces more cue-directed behavior and
relaxes the regularization that would suppress cue-weight growth, in direct,
monotonic proportion to the experimenter's own uncertainty label, before the
agent's own learning dynamics get a chance to produce or fail to produce the
effect. Any observed CDS/SPI-vs-uncertainty relationship from this pipeline,
positive, negative, or null, is not interpretable as evidence about RQ3
either way. This is exactly the design flaw the directive's Section 9 warns
against ("do not confuse ordinary reinforcement learning convergence... with
symbolic stabilization") — except the confound here is more direct than
ordinary convergence dynamics: it is a built-in behavioral bias scaled to the
independent variable itself.

(This also fully explains why `outputs/*.md`'s own trend tests mostly return
`tie / inconclusive`: a confound that forces behavior in one direction while
also disabling the regularization that would otherwise suppress it does not
reliably produce a clean monotonic *or* inverted-U signal; the one nominally
positive result, `noise` group SPI showing `inverted_u` with delta AIC
+3.551, is not evidence of anything given this confound, and was never
corrected for the 8 metric x group comparisons run alongside it.)

**RQ3 ("does uncertainty increase non-causal stabilization") has not yet
been tested by anything in this repository, in either lineage, without this confound.**

## 3. Definitions (per the directive's Section 2 request: not interchangeable)

- **Non-causal stabilization** (this session's operational definition,
  `docs/research_report.md` Section 1.2): persistent, post-intervention
  reliance on a cue that was predictive during training but is not causally
  related to reward, measured by `proxy_dependence` surviving naive-control
  and floor-control comparisons. This is the definition this session's
  manuscript actually tested and found, representation-dependently.
- **Symbolic association**: a learned mapping from a discrete, identifiable
  cue (a "symbol," in the directive's sense) to an expected outcome. Neither
  existing lineage currently has symbols with trackable identity; `GridCueEnv`
  has multiple spatial cue markers but no symbol-identity field.
- **Learned heuristic**: a behavioral rule ("prefer action X when symbol A is
  present") that may or may not be explicitly represented; in both existing
  lineages it is implicit in the agent's learned Q-values/policy, never an
  explicit, inspectable rule object.
- **Internal representation**: whatever the agent's parameters (Q-table
  entries, linear weights, network activations) encode; already measured
  indirectly via `proxy_dependence`/CDS, never directly inspected as a
  symbolic object in either lineage.
- **Policy-level behavioral pattern**: the observable action distribution;
  this is what `proxy_dependence` and CDS actually measure, one level removed
  from any claim about an internal symbolic representation.
- **Causal model**: an agent's (implicit) representation of which features
  causally determine reward; this project has never built or tested for one
  directly, only inferred its absence or presence from behavior under
  intervention.
- **Non-causal correlation**: the actual ground-truth statistical structure
  the environment designer builds in (e.g. `CueValidityEnv`'s swept `v`); a
  property of the environment, not of the agent.

**Symbolic stabilization**, as a candidate new term, would need to mean
something stricter than non-causal stabilization in general: persistent
reliance specifically on a *discrete, identifiable* cue (not a continuous or
single-bit correlate), ideally with an explicit, inspectable internal
representation of that cue's identity and estimated reliability, rather than
reliance inferred only from aggregate action statistics. Under that
definition, **nothing in this repository currently tests symbolic
stabilization** as distinct from non-causal stabilization in general, because
no existing cue carries a trackable identity and no existing metric reads
an internal representation directly rather than inferring it from the policy.

## 4. Proposed computational definition of symbolic stabilization

A cue system with `N >= 2` discrete, independently-trackable symbols
(`symbol_id in {0, ..., N-1}`), each with its own ground-truth validity
`v_i` (potentially different per symbol, so some symbols are genuinely
informative and others are not, within the same environment and the same
agent's training). A genuinely symbolic-stabilization-specific claim becomes
testable only once there is more than one symbol to compare: does an agent's
persistence differ *by symbol identity*, independent of that symbol's own
validity? If persistence scales only with each symbol's own `v_i` (as
`CueValidityEnv`'s single-cue results already show it does for validity, via
the capacity dose-response and coordinate-access findings), that is
non-causal stabilization, already established. Symbolic stabilization as a
*distinct* phenomenon would require persistence to depend on something about
symbol identity or representation (e.g. which symbol was encountered first,
how symbols are encoded, cross-symbol interference) beyond what each
symbol's own validity predicts. This has never been tested in this
repository; it requires at least two concurrently-present cues with
independently-set validities, which neither `CueValidityEnv` nor `GridCueEnv`
currently has (`GridCueEnv`'s 5 cues do not currently have independent,
experimenter-set validities; they are spatial markers, not validity-swept
symbols).

## 5. Prioritized roadmap against the directive's RQ1-RQ6

| RQ | Status |
|---|---|
| RQ1 (emergence) | Answered for the single-cue case (`docs/research_report.md` Sections 7.1-7.4): emerges under function approximation, not tabular, graded by capacity and coordinate access. Not yet tested for multi-symbol emergence. |
| RQ2 (stabilization/persistence) | Answered for the single-cue case (Sections 7.2-7.3, 7.9): persists through intervention under function approximation; the reversal-vs-decorrelation asymmetry (H3) is the clearest existing mechanism result. |
| RQ3 (uncertainty) | **Not yet validly tested anywhere in this repository** (Section 2 above). This is the most important open item and the most natural next experiment. |
| RQ4 (symbolic representation as an observable mechanism) | Not yet addressed. No explicit symbolic layer exists in either lineage (Section 1.1, 3). |
| RQ5 (revision) | Partially addressed: reversal (H3) is a form of forced revision; whether an agent can *spontaneously* detect that a heuristic has become unreliable, without an environment-forced intervention, has not been tested. |
| RQ6 (generalization) | Partially addressed: `docs/research_report.md` Section 7.7 (cross-environment replication) and Section 13.2 item 2 (proposed third environment) already cover this for non-causal stabilization generally; not yet tested for anything symbol-specific. |

## 6. The strongest next experiment

Combining the two clearest gaps (Section 2's RQ3 confound, Section 4's
multi-symbol requirement), the single strongest next experiment is **a
properly controlled uncertainty sweep, built on this session's validated
`CueValidityEnv`/`linear_raw` infrastructure rather than the confounded
`GridCueEnv` pipeline**, with:

- Uncertainty manipulated as an actual environment property (e.g. reward
  noise or outcome stochasticity added to the causal action's payoff), never
  as an agent hyperparameter. No agent setting may be a function of the
  uncertainty condition; this is the one non-negotiable fix to the existing
  design.
- The existing naive-control, floor-control, and Holm-Bonferroni-corrected
  methodology applied unchanged, so this experiment is held to the same
  standard as everything already in `docs/research_report.md`.
- `proxy_dependence` as the primary metric (not CDS/SPI alone, given
  Section 1.3's floor concern), with CDS/SPI reported alongside for
  continuity with the directive's framing.
- A pre-registered success criterion fixed before data collection, per this
  project's standing discipline.

This single experiment would be the first valid test of RQ3 in this
repository's history. A genuine multi-symbol environment (Section 4) is the
right second step once RQ3 has a clean answer, since it is a larger design
change and should not be built on top of an unresolved confound in the
simpler case.

## 7. What this audit recommends against, for now

- Building an explicit symbolic representation layer before RQ3 is cleanly
  answered: the directive's own Section 6 says "do not introduce unnecessary
  complexity before establishing a measurable experimental advantage," and
  RQ3 is answerable with the existing single-cue infrastructure plus one
  design fix.
- Treating any `outputs/*.md` number as a confirmed result: all of it is
  confounded by Section 2's finding and should be treated as uninterpretable
  pilot data, not evidence, until rerun without the confound.
- Treating CDS/SPI as primary metrics going forward without floor controls:
  `proxy_dependence` already does the job these were meant to do, more
  reliably, and is what this session's actual positive results are built on.

No code has been changed. This is the assessment requested before further
implementation or architectural decisions.
