# Non-Causal Stabilization in Reinforcement Learning

This repository investigates whether reinforcement learning agents develop,
maintain, and revise behavioural dependencies on cues that are predictive of
reward but not causally relevant to it, and what that reveals about an
agent's reliability under distribution shift.

**Current, evidence-scoped finding:** the effect is real, replicated across
two structurally different environments and multiple agent architectures,
but **representation-dependent, not universal**. Tabular agents cannot show
it at all (a structural property of tabular TD learning, not a tuning
failure). Linear and neural function-approximation agents do show it,
graded by model capacity and by how directly the cue is encoded. See
[Key Findings](#key-findings) below and `docs/` for the full,
pre-registered evidence trail.

## Research Question

Under function approximation (and, to a capacity- and
accessibility-dependent degree, under nonlinear function approximation),
does an RL agent's reliance on a predictive-but-non-causal cue persist,
beyond what a naive learner shows, after an intervention that invalidates
the cue — and does this depend on (a) representation (tabular, linear raw,
linear orthogonal, linear interaction-only, neural raw, neural
interaction-only, PPO), (b) intervention type (decorrelation vs. reversal),
and (c) environment structure?

This replaces an earlier, broader framing ("do RL agents in general develop
persistent non-causal dependencies") with one scoped to where the evidence
actually supports a claim.

## Key Findings

| Hypothesis | Status |
|---|---|
| H1: Non-causal, predictive cues produce measurable behavioural dependence | **Supported** under linear function approximation (p<0.0001, Cohen's d 8–24), **absent** under tabular representation |
| H2: Cue validity / uncertainty modulates persistence | **Tested, not supported** for a dose-response relationship after multiple-comparison correction — a graded capacity/accessibility relationship was found instead (see H4 row) |
| H3: An active correction (reversal) produces faster adaptation than a passive one (decorrelation) | **Supported** for `linear_raw`/`linear_orthogonal` (p<0.001 after correction), replicated in a second, structurally different environment for both `linear_raw` and `neural_raw` (p<0.0001, p=0.0131 after correction respectively). An extension to three additional representations in the *original* environment did **not** survive multiple-comparison correction — reported as a genuine, dated correction, not hidden. |
| H4: Different representations/algorithms show different patterns | **Supported**, in two distinct ways. (1) *Representation*: effect size is a clean, monotonic function of model capacity (Cohen's d 1.85→0.23 across MLP widths 8–128) and of cue-coordinate accessibility. (2) *Algorithm class, mechanistically*: PPO's `proxy_dependence` (read from policy logits) necessarily collapses toward zero as its policy converges, because nothing in the policy-gradient objective preserves cue-conditioned logit differences once the dominant action's advantage saturates — unlike DQN's Q-values, a regression target that keeps tracking the cue-conditioned expected-return difference indefinitely. After correcting for this (a calibrated acquisition budget, then a total-episode-matched naive control — PPO's own entropy-collapse dynamics confound the standard DQN-style comparison in a way DQN itself doesn't face), PPO shows a small, direction-consistent-with-persistence effect (Cohen's d 0.3–0.5) that 15 seeds cannot distinguish from noise — reported as underpowered, not negative. See `docs/cue_validity_ppo.md` for the full three-stage correction. |

Every claim above is backed by a pre-registered design (metrics and success
criteria committed **before** data collection), naive controls (to separate
"this agent's history matters" from "this is just how slowly any agent
adapts"), a floor control (`cue_validity=0`, to rule out the comparison
methodology itself manufacturing a gap), and — as of this pass — a stated
Holm-Bonferroni correction applied within each experiment's own comparison
family (`docs/multiple_comparison_correction.md`). Where correction
overturned an earlier-reported result, that is stated as a dated,
*additive* correction in the original document — nothing is edited away
after the fact.

## Repository Structure

```
env/            Environments (see below)
agents/         DQNAgent (linear/neural, dependency-free), PPOAgent
                (actor-critic + GAE + clipped surrogate), base_agent.py
metrics/        CDS, SPI, proxy_dependence — shared, tested definitions
experiments/    One self-contained, pre-registered script per experiment
docs/           Pre-registrations (design + criteria, committed before
                data collection) and retrospective analyses
reports/        Generated markdown reports, one per experiment
results/        Generated CSVs (summaries, per-block data, criteria)
analysis/       Cross-experiment statistical tooling (e.g. the
                multiple-comparison correction pass)
tests/          Unit tests for environments, metrics, and agents
```

Each `experiments/*.py` script is runnable standalone
(`python3 experiments/<name>.py`) and writes its own `results/*.csv` and
`reports/*.md`. There is no shared training/eval driver by design: every
experiment's design, metrics, and success criteria are specific enough that
a generic config-driven runner would hide, not clarify, what each one
actually tests.

### Environments

- **`env/cue_validity_env.py::CueValidityEnv`** — a fixed-horizon POMDP. A
  per-episode hidden state is never observed; a cue correlates with it at a
  swept `cue_validity ∈ [0,1]`; one designated action pays off only when the
  hidden state is 1; a cue-associated action never pays off. `decorrelate()`
  and `reverse()` apply interventions mid-training without touching anything
  else. The primary environment for this whole research line.
- **`env/cue_validity_grid_env.py::CueValidityGridEnv`** — the same causal
  mechanism embedded in real multi-step 2D grid navigation, used to test
  whether the pattern replicates in a structurally different task (it
  requires actually learning to navigate to the decision point, unlike the
  phase-based environment, where pre-decision actions are ignored
  entirely).
- **`env/grid_cue_env.py`, `env/gridworld.py`** — earlier prototypes,
  superseded by the two above; kept for reference, not used in any current
  experiment.

### Agents

- **Tabular SARSA(λ)** (`experiments/cue_validity_mve_experiment.py::SarsaLambdaAgent`)
  — a plain Q-table; structurally cannot show this phenomenon (see
  `docs/tabular_rl_cannot_be_superstitious.md`: a rarely-rewarded action
  cannot acquire stable value with no parameter sharing across states).
- **`agents/dqn_agent.py::DQNAgent`** — a dependency-free linear function
  approximator with a replay buffer and target network, used as `linear_raw`
  throughout. A separate small-MLP wrapper (`NeuralAgentWrapper` /
  `MLPQNet`, defined in `experiments/cue_validity_nonlinear_experiment.py`
  and reused by later experiments) provides the nonlinear `neural_raw` /
  `neural_interaction_only` conditions, with PyTorch for autodiff.
- **`agents/ppo_agent.py::PPOAgent`** — a real actor-critic PPO
  implementation (GAE, clipped surrogate objective, entropy bonus); unit
  tested including a learnability sanity check
  (`tests/ppo_agent_test.py`), replacing an earlier non-functional stub
  that returned uniform-random actions and never learned.

### Metrics (`metrics/cue_validity.py`)

- **`proxy_dependence`** = `P(causal action | cue=1) − P(causal action |
  cue=0)`. The primary metric throughout this repository. Does not require
  the non-causal action to acquire value, so it is meaningful for tabular
  agents too (and empirically is ~0 there). Limitation: a raw gap between a
  pretrained and a naive-control agent must be normalized by the distance
  to the fully-adapted target whenever that target differs between
  conditions (discovered empirically — see `docs/cue_validity_reversal.md`).
- **`cue_dependence_score` (CDS)** = softmax mass on the non-causal
  (cue-associated + neutral) actions at a cue-exposed decision state.
  **Structural limitation, proven not just asserted:** under tabular TD
  learning, CDS cannot exceed the softmax tie-breaking floor
  `(k−1)/k` for `k` available actions, regardless of true dependence — this
  repository's own `distractor_action_experiment.py` result ("Credit
  Diffusion Hypothesis: Supported: True") is an exact instance of this
  artifact, verified to three decimal places against its own action-count
  formula (`docs/family_c_reconciliation.md`). Always compare against a
  `cue_validity=0` floor control at the same action count, never against
  zero in isolation.
- **`persistence_index` / extinction half-life** — secondary descriptive
  measures (post/pre ratio; blocks until decay to half-baseline), used
  where relevant, not as primary decision criteria.

Every metric has unit tests in `tests/cue_validity_metrics_test.py`.

## Statistical Methodology

- Independent two-sample t-tests with Cohen's d reported alongside every
  p-value.
- 15–30 seeds per cell.
- Pre-registration: design, metrics, and success/failure criteria
  committed to `docs/` **before** the corresponding experiment is run.
- Dated, additive addenda for any post-hoc correction or follow-up finding
  — the original criteria are never edited away, only appended to.
- Holm-Bonferroni correction applied within each experiment's own
  pre-registered comparison family (`docs/multiple_comparison_correction.md`,
  `analysis/multiple_comparison_correction.py`), with the policy and its
  rationale for the family boundary stated explicitly.

## What Was Already Known to Be Broken (Audited, Not Modified)

Three earlier pipeline families predate this research line and are
preserved for reference, not relied on:

- **Family A** (`experiments/train.py`, `experiments/ablations.py`) —
  non-functional, 3 independent stacked bugs confirmed by execution.
- **Family B** (`experiments/run_pipeline.py`,
  `run_mechanism_isolation.py`, `run_factorial_credit_assignment.py`) —
  functional, but agent exploration bias is set as a deterministic
  function of the manipulated uncertainty variable, making any
  "uncertainty increases dependence" finding from this family close to
  circular. (Cached outputs in `runs/mechanism_eval_cache/` are from this
  family; they are not evidence for or against any hypothesis above.)
- **Family C** (~14 self-contained scripts: distractor, spurious
  correlation, ambiguous causality, hidden-state, causal-reversal, the
  linear/orthogonal/nonlinear proxy-feature chain) — contrary to an earlier
  summary of this repository, **most of these are not clean nulls**; see
  `docs/family_c_reconciliation.md` for the full correction, including an
  exact mechanistic diagnosis for all four genuine nulls: two from
  tabular representation (one with an added "cue separated from the
  decision" design flaw), two from a more basic fact — their reward is
  fully deterministic, with no hidden stochastic variable for their
  "cue" to carry any rational information about.

## Running Experiments and Tests

```bash
pip install -r requirements.txt

# Run the full test suite
pytest tests/ -q

# Run any single experiment (writes to results/ and reports/)
python3 experiments/cue_validity_proxy_dependence_experiment.py
python3 experiments/cue_validity_ppo_experiment.py

# Re-run the cross-experiment statistical correction
python3 analysis/multiple_comparison_correction.py
```

Each experiment prints per-seed progress to stdout and writes its own
`results/*_summaries.csv` (raw per-seed data), `results/*_criteria.csv`
(the pre-registered test's outcome), and `reports/*.md` (a readable
summary). Reading an experiment's corresponding `docs/*.md` file first is
the fastest way to understand what it tests and why.

## Relevance Beyond GridWorld

The scientific claim in this repository is scoped to the environments
tested — it does not establish equivalent behaviour in LLMs or physical
agents. The evaluation *principles* are intended to transfer: pre-registered
success criteria before seeing results, naive and floor controls to rule
out trivial explanations, replication in a structurally different
environment before generalizing a claim, and a stated multiple-comparison
policy rather than ad hoc significance-chasing. These are the same
discipline a behavioural-robustness or distribution-shift evaluation of a
larger model would need, whatever the model class.

## Open Items

- A larger-seed PPO rerun: with the budget and naive-control confounds
  fixed (`docs/cue_validity_ppo.md`), PPO shows a small,
  direction-consistent-with-persistence effect (Cohen's d 0.3–0.5 at the
  default width) that 15 seeds cannot resolve from noise. A capacity
  sweep (`docs/cue_validity_ppo_capacity_sweep.md`) found this is not a
  monotonic capacity curve like DQN's — effect size peaks at the default
  width and reverses sign at a wider network, plausibly a
  capacity-dependent entropy-collapse-speed effect, but 15 seeds per
  width cannot distinguish that from noise. A seed top-up (the standard
  escalation used elsewhere in this lineage) would settle both.
- `neural_interaction_only`'s own decorrelate-vs-naive effect in the grid
  environment shows an unexplained reversed direction at v=0.5 (naive >
  decorrelate, significant but wrong-signed) — the coordinate-dependence
  conclusion there is unaffected, but this specific result is not
  diagnosed (`docs/cue_validity_grid_neural_reversal_interaction.md`).
