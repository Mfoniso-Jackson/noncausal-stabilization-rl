# Family C Reconciliation

This is a retrospective analysis of already-collected data from this
repository's ~14 pre-existing, self-contained "Family C" experiment
scripts (the ones the original audit, before this session's cue-validity
work began, characterized as "functional, properly designed, consistently
null for persistence"). It answers the publication plan's Section 5 item
5: explain, under one shared metric, why the pre-existing scripts found
null results while the cue-validity lineage found a positive one. No new
experiments were run for this document -- every number below is quoted
directly from each script's own already-generated `reports/*.md`.

## Headline correction

**The premise that Family C "consistently" found null results for
persistence is wrong for several of its scripts.** At least four
pre-existing scripts already found large, clearly-reported persistent
`proxy_dependence` under linear or shared-parameter function
approximation, with little or none under tabular representation -- the
same qualitative pattern this session's cue-validity lineage later found
independently, with far more rigor (naive controls, pre-registration,
cross-environment replication, an optimizer ablation). This was not
caught by the earlier audit's blanket summary and should be treated as a
correction to that summary, not a footnote.

| Script | Reports | What it actually found |
|---|---|---|
| `causal_reversal_agent_experiment.py` | `causal_reversal_agent_report.md` | `dqn` (the repository's linear function approximator): acquisition `proxy_dependence`=0.848, reversal=-0.881 (full sign flip), **extinction=-0.723** (still large after the cue became fully uninformative). `tabular_sarsa`: -0.092 at the same point. The report's own verdict: **"Generalized proxy-reliance persistence supported: True."** |
| `linear_feature_ablation_experiment.py` | `linear_feature_ablation_report.md` | Extinction absolute `proxy_dependence`: `proxy_only`=0.800, `full_current_features`=0.689, `phase_only`=0.015. Framed as "representation-induced proxy reliance," not centered as a positive finding, but the magnitude is large and clearly representation-dependent. |
| `orthogonalized_proxy_feature_experiment.py` | `orthogonalized_proxy_feature_report.md` | Extinction absolute `proxy_dependence`: `raw`=0.718, `rotated`=0.455, `interaction_only`=0.158, `phase_only`=0.019 -- the same graded, coordinate-accessibility-dependent pattern this session's `cue_validity_nonlinear_experiment.py` found again, independently, months later. |
| `nonlinear_interaction_proxy_experiment.py` | `nonlinear_interaction_proxy_report.md` | `linear/raw`=0.718, `linear/interaction`=0.158, `neural/raw`=0.124, `neural/interaction`=0.070 -- shrinking under either nonlinearity or interaction-only encoding, individually, and most under both together: the exact "graded, not binary" shape this session's own nonlinear experiment reports as its headline finding. |

These four scripts' own authors did not frame their results as "we found
the phenomenon" -- `linear_feature_ablation_report.md` and
`orthogonalized_proxy_feature_report.md` both explicitly interpret their
own positive numbers as evidence the effect is a "representation
artifact," which this repository's later audit (before this session)
appears to have read as "no real effect," propagating into the "Family C:
consistently null" summary. That interpretive move -- finding a real,
reproducible effect and then dismissing it as "just an artifact of
representation" rather than reporting it as a scoped, representation-
dependent phenomenon -- is exactly the gap this session's cue-validity
lineage closes: the same finding, with naive controls to rule out "any
agent adapts this slowly," a floor control, cross-environment
replication, and an optimizer ablation, reported as what it is rather
than dismissed.

## A script whose own pass/fail criterion obscures its own data

`neural_recurrent_reversal_experiment.py`
(`neural_recurrent_reversal_report.md`) reports extinction
`proxy_dependence`: `tabular`=-0.117, **`linear`=-0.718**, `neural`=0.124,
`recurrent`=-0.012, then concludes "Generalized proxy-reliance persistence
supported: **False**." Reading the actual criterion in the script
(line 752): `supports_proxy = max(abs(neural), abs(recurrent)) >
abs(tabular) + 0.20`. This criterion never looks at `linear` at all -- it
only asks whether the *neural or recurrent* conditions exceed tabular by a
fixed margin, and neural's point estimate (0.124) does not clear that bar
against tabular's own noisy ~0.12-0.17. The large `linear` result is
reported in the same table and simply excluded from the pass/fail
question, apparently because it was treated as already established by
`causal_reversal_agent_experiment.py`. This is not a contradiction of the
cue-validity lineage's finding -- `tabular < neural < linear` in
magnitude here is the same ordering this session's own `neural_raw` vs
`linear_raw` comparison found (neural significant but reliably smaller
than linear) -- it is a single, unreplicated, non-significance-tested
point estimate answering a narrower question ("does neural/recurrent
specifically beat tabular by 0.20") than the one this report is asking.
`recurrent`'s near-zero value is consistent with this session's own
explanation for its new recurrent-belief-state experiment ("belief head
tracks the proxy shortcut, not causal structure").

## A positive-looking result that is actually a measurement artifact

`distractor_action_experiment.py` (`distractor_action_report.md`) reports
CDS rising with the number of distractor actions (0.000, 0.667, 0.833,
0.909 at 0, 2, 5, 10 distractors) alongside goal_rate pinned at 1.000
throughout, and concludes "Credit Diffusion Hypothesis: Supported: True."
Checking the numbers against the softmax tie-breaking floor this
repository's own `metrics/cue_validity.py` warns about explicitly
(`(len(available_actions)-1)/len(available_actions)` for a perfectly-tied
Q-function): with 1 causal + N distractor actions, that floor is `N/(N+1)`
-- which gives exactly 0.667 (N=2), 0.833 (N=5), and 0.909 (N=10), an
**exact match to three decimal places** at every reported point, with
reward and goal_rate never moving. This is not credit diffusion; it is
the same CDS floor artifact this repository's own `docs/cue_validity_mve.md`
and `metrics/cue_validity.py` already identify and warn against, appearing
here under a different name ("Credit Diffusion Hypothesis") in a
pre-existing script that did not check for it. It is correctly grouped
with the null results, just not for the reason its own report gives.

## Scripts that found genuine nulls, and what little is known about why

`spurious_correlation_experiment.py` and `temporal_distractor_experiment.py`
both report `goal_rate=1.000` in their flagship condition -- the task was
solved perfectly regardless of any spurious dependency, which would mean
there was no behavioral pressure for a dependency to show up even if one
existed at the value level; this is a plausible explanation but was not
directly tested in either script (no decomposition of Q-value-level vs.
action-selection-level dependence under a ceiling-performance condition),
so it is reported here as a hypothesis, not a confirmed mechanism.
`hidden_state_superstition_experiment.py` and
`ambiguous_causality_experiment.py` report non-ceiling goal rates
(0.405-0.927) with small, non-significant trend metrics; this reconciliation
did not re-derive why in further mechanistic detail -- doing so would
require re-running them with the naive-control and pre-registration
methodology this lineage now uses, which is noted as a possible future
item but was not undertaken here.

## Revised answer to "why did Family C find null results while this line found positive ones?"

The honest answer is: **it mostly didn't.** Four of the ~14 scripts
already found the same core effect (persistent proxy-style dependence
under linear/shared-parameter function approximation, graded by
representation, small-to-absent under tabular), with magnitudes in the
same range this session's cue-validity lineage reports. One script's
own binary verdict obscures a result in its own table that is directionally
consistent with the cue-validity lineage's graded finding. One
apparently-positive result is a measurement artifact unrelated to the
phenomenon (the CDS tie-breaking floor, scaling mechanically with action
count). Only a minority of the family -- two or three scripts, on this
reading -- found results that look like genuine nulls, and the mechanism
there (ceiling performance removing any behavioral pressure to reveal a
dependency, in at least two of them) is a plausible but not confirmed
explanation.

This session's contribution is therefore not "discovering an effect that
Family C uniformly missed," but **doing rigorously, with pre-registration,
naive controls, a floor control, cross-environment replication, and an
optimizer ablation, what several Family C scripts had already glimpsed
and then set aside as an artifact** -- and separately, in
`docs/tabular_rl_cannot_be_superstitious.md`, explaining structurally why
the tabular conditions in both this lineage and Family C consistently
show nothing. The publication plan's Section 4 table should be read with
this correction; the blanket "consistently null for persistence"
characterization of Family C is inaccurate and is corrected here.
