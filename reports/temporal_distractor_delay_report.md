# Temporal Distractor Delay Experiment

## Research question
temporal_distractor_report.md found no near-vs-far proximity gradient in distractor Q-mass or selection rate (near_minus_far was near zero and not reliably positive). Does that null result change if the credit-assignment problem is made explicitly harder by pushing the reward phase further away from the (unchanged) distractor exposure windows?

## Why this follows the temporal distractor experiment
That experiment's own recommended next step was exactly this: hold temporal availability fixed and manipulate reward delay after near versus far distractor exposure, to see whether the (absent) proximity gradient strengthens when credit assignment is made harder.

## Design
The near ([7, 8, 9]), mid ([4, 5, 6]), far ([1, 2, 3]) and neutral distractor windows are at the exact same absolute phases as the original temporally_biased_distractors condition, and the delay=0 condition reproduces that environment exactly. For delay in [0, 3, 6, 10], the reward phase is moved from phase 10 to phase 10 + delay; the 10-step post-target slack is preserved. The phases inserted between the near window (ending at phase 9) and the (now later) reward phase offer no distractor actions at all -- only action 0 is available there, so the only thing that changes with delay is how many credit-assignment steps separate distractor exposure from the eventual reward signal.

## Agent design
Tabular SARSA(lambda), alpha=0.08, gamma=0.95, lambda=0.8, epsilon 1.0->0.04, 2500 training episodes, 100 evaluation episodes, 30 seeds per delay condition.

## Results table

| delay | target phase | reward | goal_rate | CDS | SPI | SPI_near | SPI_mid | SPI_far | SPI near-far | Q_mass near-far |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 0.978 +/- 0.000 | 1.000 +/- 0.000 | 0.500 +/- 0.000 | 0.404 +/- 0.037 | 0.516 | 0.500 | 0.600 | -0.084 +/- 0.128 | -0.000 +/- 0.002 |
| 3 | 13 | 0.972 +/- 0.000 | 1.000 +/- 0.000 | 0.385 +/- 0.000 | 0.344 +/- 0.042 | 0.466 | 0.533 | 0.378 | 0.089 +/- 0.140 | -0.000 +/- 0.002 |
| 6 | 16 | 0.966 +/- 0.000 | 1.000 +/- 0.000 | 0.312 +/- 0.000 | 0.367 +/- 0.049 | 0.478 | 0.511 | 0.478 | 0.000 +/- 0.133 | -0.000 +/- 0.002 |
| 10 | 20 | 0.958 +/- 0.000 | 1.000 +/- 0.000 | 0.250 +/- 0.000 | 0.386 +/- 0.045 | 0.533 | 0.511 | 0.500 | 0.033 +/- 0.151 | -0.000 +/- 0.002 |

## Dose-response tests (n=120 seed-delay observations)

SPI near-minus-far gap vs delay: r = 0.0727, p = 0.4299.
Q-mass near-minus-far gap vs delay: r = 0.0000, p = 0.9999.
Overall SPI (all distractor types) vs delay: r = -0.0207, p = 0.8227.
Overall CDS vs delay: r = -0.9779, p = 0.0000.

## Interpretation
At delay=0 (identical to the original temporally_biased_distractors condition), SPI near-minus-far = -0.084, matching the original report's near-null finding. At the largest tested delay (10 extra steps), SPI near-minus-far = 0.033. If distractor Q-mass and selection rate were driven by eligibility-trace distance from the reward signal, lengthening the pure, distractor-free gap between the near window and the reward should widen the near/far gap, since the relative eligibility-trace discount ratio between the near and far windows is unaffected by this manipulation but the overall magnitude of all distractor credit should shrink and -- if genuinely distance-driven -- shrink unevenly. The trend tests above are the direct answer to whether that happens.

## Does credit-assignment difficulty strengthen the proximity gradient?
Supported: False.

## A metric-construction artifact in CDS, not a real effect
Overall CDS falls sharply and almost perfectly with delay (0.500 at delay=0 down to 0.250 at delay=10, r=-0.9779, p=0.0000), and CDS * target_phase is approximately constant across every row of the results table above. This is not evidence that longer delay reduces superstition: CDS, as defined in this codebase's `evaluate()` pattern (shared with temporal_distractor_experiment.py and distractor_action_experiment.py), averages the distractor Q-mass over *every* pre-target decision step, including steps where no distractor is available at all (which contribute exactly 0). The filler steps this experiment inserts to create delay are always distractor-free, so they add zero-valued terms to that average without changing the fixed number of genuinely distractor-available steps -- mechanically diluting CDS as delay grows, independent of any change in the policy's actual distractor preference when a distractor is present. SPI, which is defined only over decisions where a distractor was actually available, does not show this artifact (its near-flat, non-significant trend above is the metric that should be trusted for this question). This is worth flagging as a general caveat for any future use of this repository's CDS metric on environments with variable-length, non-uniform distractor availability.

## Limitations
Delay is varied only up to 10 extra steps given the tabular agent's horizon and the fixed near/mid/far window positions; much larger delays were not tested. Approximate correlation p-values use scipy's Pearson test on seed-level observations pooled across delay conditions and do not correct for the two primary gap-vs-delay comparisons (SPI and Q-mass) tested together.

## Next recommended experiment
If the gradient remains null even at the largest delay tested here, the next step is to test much larger delays with a function-approximation agent (where generalization across nearby states, rather than tabular independence, could reintroduce a distance-dependent credit leak that this tabular design cannot show).
