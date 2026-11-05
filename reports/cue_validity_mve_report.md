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
This MVE tests only the decorrelate intervention with a tabular SARSA(lambda) agent, by pre-registered design -- reversal and action-removal interventions, and function-approximation agents, are explicitly deferred (see docs/cue_validity_mve.md, Planned follow-ups). Extinction half-life uses 50-episode blocks; a null half-life result at this resolution does not rule out a faster, sub-block dynamic. More importantly, per the mechanistic diagnosis above: this design cannot test the intervention-persistence question at all until the cue is observed at (or folded into) the same decision point where the causal and cue-associated actions compete for the terminal reward -- as the pre-existing experiments in this repository's `experiments/` directory do (e.g. `ambiguous_causality_experiment.py`'s PROBE_PHASE = TARGET_PHASE - 1). That structural change is the first item in the next-experiment recommendation below, not an optional refinement.

## Next recommended experiment
Re-run this exact design (same pre-registered metrics, success criteria, and conditions) with one structural change: fold the cue observation into the target_phase decision state itself (cue_phase = target_phase), so the causal and cue-associated actions genuinely compete for the same terminal-reward credit at the same state, giving cue_validity an actual channel through which it could bias behavior. This also serves as a direct, controlled test of a hypothesis raised by this result: that measured CDS/SPI departures from the tie-breaking floor in this repository's other experiments (which do have this competing-at-the-same-state structure) may themselves be largely credit-diffusion/stochastic-approximation drift rather than genuine validity-dependent learning, since a cue_validity=0.0 floor control run through the *same* competing-action structure would be the cleanest available test of that alternative explanation.
