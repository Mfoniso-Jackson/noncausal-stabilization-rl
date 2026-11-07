# Cue Validity MVE

## Pre-registration
This experiment's design, metrics, and success/failure criteria were written and committed in docs/cue_validity_mve.md before any of the data below was collected. Nothing in this report's criteria was changed after seeing results.

## Design summary
cue_validity in [0.0, 0.5, 0.95], 30 seeds. For validity 0.5 and 0.95: an acquisition-only arm (no intervention, 3000 episodes), a decorrelate arm (3000 acquisition + 1500 post-decorrelation episodes in 30 blocks of 50), and a naive-control arm (fresh agent, 1500 episodes directly in the already-decorrelated environment). For validity 0.0 (floor): an undisturbed arm and its own naive-control arm, as a check that the comparison methodology itself does not manufacture a spurious gap when there is no real cue information to begin with.

## Results table (mean across seeds)

| arm | acquisition CDS | final-block CDS | extinction half-life (blocks) | final reward | final goal_rate |
|---|---:|---:|---:|---:|---:|
| floor_none | 0.752 | 0.750 | 30.00 | 0.494 | 0.504 |
| floor_naive | n/a | 0.750 | n/a | 0.494 | 0.504 |
| v0.5_none | 0.737 | 0.748 | 30.00 | 0.494 | 0.504 |
| v0.5_decorrelate | 0.737 | 0.752 | 30.00 | 0.494 | 0.504 |
| v0.5_naive | n/a | 0.750 | n/a | 0.494 | 0.504 |
| v0.95_none | 0.752 | 0.748 | 30.00 | 0.494 | 0.504 |
| v0.95_decorrelate | 0.752 | 0.751 | 30.00 | 0.494 | 0.504 |
| v0.95_naive | n/a | 0.750 | n/a | 0.494 | 0.504 |

## Pre-registered success criteria, criterion 1 (decorrelate vs naive control, per validity level)

| validity | decorrelate final CDS | naive final CDS | t | p | Cohen's d | criterion 1 met |
|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 0.752 | 0.750 | 0.288 | 0.7743 | 0.076 | False |
| 0.95 | 0.751 | 0.750 | 0.112 | 0.9113 | 0.029 | False |

## Pre-registered success criteria, criterion 2 (floor control must show no comparable gap)

floor_none final CDS = 0.750, floor_naive final CDS = 0.750, t = -0.006, p = 0.9953, Cohen's d = -0.002. Criterion 2 met (no significant gap at the floor): True.

## Verdict (per pre-registered criteria in docs/cue_validity_mve.md)
Hypothesis supported: False.

## Why acquisition itself shows no differentiation (mechanistic diagnosis, not just a null result)
Acquisition-end CDS is statistically indistinguishable from the exact 4-way tie-breaking value (0.75) at every cue_validity level, including 0.95: mean 0.752 (seed range 0.699 - 0.770). This is a stronger finding than 'no effect after intervention' -- it means no validity-dependent preference ever forms during acquisition in the first place, so the intervention-persistence question this MVE set out to test is moot for this specific environment structure. The mechanism is structural, not a training-budget issue: `cue_phase` and `target_phase` are separate decision points, and choosing any action at `cue_phase` leads to the identical next state and the identical immediate reward (step_penalty) regardless of which action is chosen. Under SARSA's bootstrap, Q(cue_phase_state, a) for every action a is pulled toward the same value -- the expected return from target_phase onward -- independent of a, so there is no reward-relevant channel through which cue_validity could ever bias those action-values, at any amount of training. The residual, seed-to-seed spread around 0.75 is ordinary stochastic-approximation noise from the training process itself, not signal: it does not shift with cue_validity, and the floor control (cue_validity=0.0, which carries no information by construction) shows the same magnitude of spread as the strongest validity level. This is a genuine limitation of this specific environment design, not a bug in the sense of incorrect code -- every invariant in tests/cue_validity_env_test.py (reward never depends on the cue, decorrelation genuinely removes the correlation) holds exactly as intended.

## Limitations
This MVE tests only the decorrelate intervention with a tabular SARSA(lambda) agent, by pre-registered design -- reversal and action-removal interventions, and function-approximation agents, are explicitly deferred (see docs/cue_validity_mve.md, Planned follow-ups). Extinction half-life uses 50-episode blocks; a null half-life result at this resolution does not rule out a faster, sub-block dynamic.

## Correction (added after initial publication of this report; see docs/cue_validity_mve.md's 'Correction to the addendum')
An earlier version of this section recommended folding cue_phase into target_phase as the fix for the diagnosis above. That was checked empirically and does not work, for a more general reason than the one diagnosed above: see docs/tabular_rl_cannot_be_superstitious.md. Folding the cue into the decision state makes CDS *decrease* as cue_validity increases (checked: 0.58 / 0.55 / 0.53 at validity 0.0 / 0.5 / 0.95), because the cue-associated action never yields reward at any validity level, so a more informative cue only sharpens the (correct) preference for the causal action -- that tests rational cue exploitation, not non-causal dependence. More generally: tabular RL cannot exhibit genuine non-causal stabilization at all, because every (state, action) pair gets an independent value estimate with no mechanism for information to leak between states -- a never-rewarded action cannot acquire a stable positive value regardless of how correlated its context is with reward. This is consistent with, and explains, every tabular-agent intervention in this repository (this MVE and the pre-existing scripts alike) finding no persistent effect.

## Next recommended experiment
Per docs/tabular_rl_cannot_be_superstitious.md: the phenomenon requires either genuine state aliasing (two causally different situations indistinguishable to the agent, not merely observed at different times) or function approximation with shared parameters across states. This repository's own linear_feature_ablation_experiment.py -> orthogonalized_proxy_feature_experiment.py -> nonlinear_interaction_proxy_experiment.py chain already tested exactly this boundary and found the one positive signal it produced was a brittle linear-representation artifact, not a robust effect. A credible next experiment needs a state-aliased or function-approximation redesign of this MVE's cue-validity sweep that explicitly rules out that same artifact (e.g. by testing whether an orthogonalized or nonlinear representation of the same validity-swept cue reproduces or kills any observed effect), rather than another tabular variant of this design.
