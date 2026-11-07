# Spurious Correlation Reversal Experiment

## Research question
Once the spurious action is made explicitly anti-correlated with reward (choosing it poisons the episode's terminal reward even if the causal action is chosen later), does the speed of its extinction depend on how strongly it was reinforced during acquisition?

## Why this follows the spurious correlation experiment
`spurious_correlation_report.md` found that stronger training-time correlation between the spurious action's availability and the reward phase increases its learned Q-advantage and selection rate, and its own recommended next experiment was exactly this reversal test.

## Design
For each of the same 4 correlation-strength conditions and 30 seeds used in the original experiment, a SARSA(lambda) agent is pretrained for 2500 acquisition episodes exactly as before (the spurious action is available near the target phase with condition-specific probability and never affects reward). A frozen-policy baseline spurious action rate is then measured under the original decorrelated evaluation distribution. The same environment instance then enters reversal: the spurious action's availability statistics are unchanged, but choosing it now poisons the episode -- the terminal causal action no longer pays out even if chosen correctly later. The agent keeps learning online (epsilon=0.04, same as the end of acquisition) for 1000 further episodes, binned into 20 blocks of 50, and the fraction of pre-target decisions selecting the spurious action is tracked per block.

## Metrics
Per-seed half-life (in blocks) is the first reversal block whose spurious action rate falls to half of that seed's own pre-reversal baseline rate; a value of 20 means it never reached half within the window (censored). The dose-response test is the Pearson correlation between acquisition correlation strength and per-seed half-life across all 120 (condition x seed) observations, plus an independent two-sample t-test contrasting the strong and no-correlation conditions' half-lives.

## Results table

| condition | strength | baseline SPI(spurious) | half-life (blocks) | final-block SPI(spurious) | n censored / 30 |
|---|---:|---:|---:|---:|---:|
| no_spurious_correlation | 0.10 | 0.220 | 5.57 +/- 1.86 | 0.081 +/- 0.027 | 2 |
| weak_spurious_correlation | 0.60 | 0.238 | 2.17 +/- 0.65 | 0.067 +/- 0.021 | 0 |
| medium_spurious_correlation | 0.75 | 0.230 | 1.97 +/- 0.78 | 0.055 +/- 0.017 | 0 |
| strong_spurious_correlation | 0.90 | 0.239 | 1.33 +/- 0.51 | 0.044 +/- 0.014 | 0 |

## Dose-response test

Pearson r(correlation_strength, half_life_blocks) = -0.4736, p = 0.0000 (n=120 seed-level observations).
Strong vs none half-life: 1.33 vs 5.57 blocks, t = -4.293, p = 0.0001.

## Interpretation
Half-life is monotonically non-decreasing across the 4 conditions in strength order: False. The no-correlation condition's baseline spurious rate (0.220) is the floor set by the evaluation distribution's independent spurious-availability probability rather than any learned preference, so its half-life mostly reflects noise around that floor. The strong condition starts from a similar baseline of 0.239 but its half-life is *shorter*, not longer: the dose-response correlation is negative and significant, the opposite of a persistence hypothesis. A plausible mechanism is that the same acquisition process which made the spurious action's Q-value confidently (if wrongly) elevated also makes the TD error on first contact with the poisoning signal larger in magnitude, so the correction step itself is proportionally bigger and the value is driven down faster -- the stronger the illusory belief, the larger the single prediction-error correction needed to erase it under SARSA(lambda)'s linear update rule. Under this strong, deterministic correction signal, more confident acquisition does not buy the behavior any extra resistance to intervention.

## Does this support dose-dependent persistence of computational superstition under reversal?
Supported: False.

## Limitations
Per-seed half-life estimates are based on 50-episode blocks and relatively rare pre-target decision opportunities (the spurious action is only available at specific phases), so individual seed curves are noisy; the aggregate block-level summary CSV is the more stable signal for inspecting the shape of the extinction curve. The poisoning mechanism is a strong, unambiguous correction (it zeroes the entire episode's reward), which is a best case for fast extinction; a milder anti-correlation (a partial reward penalty rather than full poisoning) might reveal slower extinction that this design does not capture.

## Next recommended experiment
Repeat this reversal test with a milder, probabilistic anti-correlation (the spurious action only sometimes blocks reward) rather than full poisoning, to test whether partial punishment produces slower, more superstition-like extinction than the strong, fully deterministic correction used here.
