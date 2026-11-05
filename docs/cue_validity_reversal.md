# Cue Validity Reversal Experiment: Pre-Registration

Written and committed before any results from this experiment exist. This
is a direct continuation of `docs/cue_validity_proxy_dependence.md`'s
"Next recommended experiment"; read that document and
`reports/cue_validity_proxy_dependence_report.md` first.

## What this tests

The proxy-dependence experiment found that `linear_raw` and
`linear_orthogonal` agents show large, statistically decisive, persistent
`proxy_dependence` after a **decorrelate** intervention (the cue becomes
uninformative), while `tabular` agents do not. That report proposed a
gradient-asymmetry mechanism: acquisition provides a strong, structured
gradient; decorrelation provides only weak, zero-mean noise with no
systematic force correcting it, whereas an active **reversal** (the cue
becomes anti-predictive, as in
`experiments/spurious_correlation_reversal_experiment.py`'s tabular-agent
finding of fast extinction under reversal) should provide exactly that
corrective force and should extinguish the dependency faster.

This experiment holds everything fixed except the intervention type
(`reverse()` instead of `decorrelate()`, via `env/cue_validity_env.py`'s
new `reverse()` method) and directly compares the size of the resulting
"gap to full adaptation" against the already-collected decorrelation
results, rather than re-deriving a new, unrelated verdict.

## Design

- Same environment, same three agent/representation conditions (`tabular`,
  `linear_raw`, `linear_orthogonal`), same `cue_validity` in {0.5, 0.95}
  (the floor condition is not repeated here: reversing a cue with no
  correlation to begin with is not meaningfully different from
  decorrelating it, and that comparison was already made in the prior
  experiment), same 15 seeds, same episode budget (3000 acquisition, 1500
  post-intervention in 30 blocks of 50).
- For each (agent_type, validity): a **reversed** arm (acquisition, then
  `env.reverse()`, then continued training) and a **naive-reversed**
  control arm (fresh agent, trained only in an already-reversed
  environment for the same post-intervention budget) -- mirroring the
  decorrelation experiment's naive-control methodology exactly.

## Metrics

- `proxy_dependence` (signed, not absolute this time): acquisition should
  produce a positive value (cue=1 predicts hidden_state=1, so the policy
  rationally favors the causal action more when cue=1). After reversal,
  the true relationship flips, so a fully-adapted agent's
  `proxy_dependence` should become **negative** with the same approximate
  magnitude -- "adaptation" here means tracking the new sign, not decaying
  to zero.
- **Gap to full adaptation**, per seed: `reversed_arm_final_proxy_dependence
  - naive_reversed_arm_final_proxy_dependence`. A naive agent, trained only
  in the reversed environment, is taken as the standard for "fully
  adapted to the new regime" (it has no old belief to unlearn). A gap near
  zero means the pretrained agent has fully adapted; a gap that stays
  positive means it is still carrying forward the old, now-wrong positive
  association.
- The same gap, computed from the already-collected decorrelation
  experiment's `results/cue_validity_proxy_dependence_summaries.csv`
  (`|decorrelate_arm_final - naive_arm_final|`, since there the fully-
  adapted target is 0, not a flipped sign), is the comparison baseline.

## Success / failure criteria (fixed before data collection)

The gradient-asymmetry mechanism is **supported**, per agent/representation
condition, only if `|reversal gap|` is significantly smaller (independent
two-sample t-test across the 15 seeds of each experiment, p < 0.05, Cohen's
d reported) than `|decorrelation gap|` for the same agent type, at the same
validity level. It is **not supported** if the gaps are statistically
indistinguishable or if the reversal gap is larger. For `tabular`, both
gaps are expected to already be small (per the prior results); the
criterion there is a check that reversal does not somehow make tabular
*worse*, not a test of the asymmetry mechanism itself.

A null or small result is a valid, reportable outcome.

## What would change this plan after the fact

Nothing in the metrics, conditions, or criteria above will change after
results are inspected, following the same discipline as the two prior
pre-registrations in this lineage.
