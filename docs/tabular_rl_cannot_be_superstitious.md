# Why tabular RL cannot exhibit genuine non-causal stabilization

This note corrects the "Next recommended experiment" in
`reports/cue_validity_mve_report.md` (fold `cue_phase` into `target_phase`),
which this note shows does not test the hypothesis, and generalizes the
reason why to the whole tabular-agent portion of this research program.

## The claim

In every tabular-agent environment used in this repository -- the
cue-validity MVE and every pre-existing `experiments/*_experiment.py`
script -- the cue-associated / proxy / distractor action never yields
reward, in any state, under any condition. This is true by the explicit
design requirement that the cue is never causal. Given that, a tabular
agent cannot develop a stable, reward-reinforced preference for that
action, because tabular Q-learning and SARSA give every `(state, action)`
pair an independent value estimate: there is no mechanism by which
information about the action's value in one state can bias its value in
another. Any measured CDS/SPI above the state's tie-breaking floor is
necessarily a finite-sample, pre-convergence artifact (eligibility-trace
credit diffusion, optimistic/tied initialization, early-exploration
lock-in) -- not a genuine, asymptotically stable dependency. This predicts,
and is consistent with, every intervention tested with a tabular agent in
this repository finding no persistent effect.

## Empirical check: does folding the cue into the decision state help?

The cue-validity MVE originally measured CDS at a cue-observation phase
*separate* from the decision phase, where no action has any effect on the
transition or reward -- so cue_validity cannot possibly bias those
Q-values (this is also why the MVE's acquisition-end CDS never left the
tie-breaking floor; see the MVE report's own diagnosis section). The
natural-seeming fix is to make the cue part of the decision state itself
(`cue_phase = target_phase`), so the causal and cue-associated actions
compete directly. This was checked directly:

```
validity=0.0  folded-CDS=0.5845   Q[causal | cue=1] = 0.755
validity=0.5  folded-CDS=0.5522   Q[causal | cue=1] = 0.887
validity=0.95 folded-CDS=0.5289   Q[causal | cue=1] = 0.981
```

CDS *decreases* as validity increases -- the opposite of what persistence
on a non-causal cue would predict. The reason is immediate once the two
Q-values are inspected separately: `Q[ACTION_CAUSAL | cue]` rises toward
the correct Bayesian posterior `P(hidden_state=1 | cue)` as validity
increases, while `Q[ACTION_CUE]` stays pinned at the step-penalty floor
regardless of validity, because ACTION_CUE never yields reward at any
validity level. A more informative cue just makes the causal action's
advantage clearer and the softmax more confidently correct -- this is
"exploiting a genuinely informative cue" (the research brief's own
distinction, Section 1), not dependence on a non-causal one. Folding the
cue into the decision state does not test the hypothesis; it tests
something categorically different, and should not be run as a "fixed"
version of this MVE.

## What would actually make the phenomenon structurally possible

For a tabular agent to develop a stable preference for a never-rewarded
action, two distinct `(state, action)` pairs with different true values
would need to be forced to share a single value estimate -- i.e., state
aliasing (partial observability that makes two causally-different
situations look identical to the agent) or function approximation
(parameter sharing across states, so a feature correlated with reward in
training data can acquire nonzero weight even though it never causes
reward). This is not a new design requirement invented here: it is
exactly the design pattern three pre-existing scripts in this repository
already followed, apparently for other reasons, and their results now
read as a consistent test of this exact theoretical boundary rather than
three unrelated findings:

- `linear_feature_ablation_experiment.py` found a linear-DQN agent
  (feature-sharing function approximation) showing a proxy-dependence
  effect that a tabular agent did not.
- `orthogonalized_proxy_feature_experiment.py` found that effect was tied
  to the raw proxy coordinate's specific position in the linear feature
  basis -- an artifact of how the feature happened to be represented, not
  a property of the correlation itself.
- `nonlinear_interaction_proxy_experiment.py` found a neural (nonlinear)
  network could not recover the effect from an interaction-coded
  representation of the same information either.

Read together under this note's theory: the one place a tabular-vs-
function-approximation contrast *could* show a real difference did show
one initially, and closer inspection in this repository's own prior work
found it was an artifact of a specific, brittle linear representation --
consistent with representation-sharing being necessary but not sufficient,
and with no robust instance of the phenomenon surviving scrutiny anywhere
in this project so far, tabular or not.

## Implication for the research program

This does not mean the central hypothesis is false in general -- it means
every tabular-agent test in this repository (the pre-existing scripts and
this session's new ones alike) was testing a regime in which the
phenomenon is structurally impossible, by the mathematics of tabular TD
learning, independent of how carefully each individual environment was
designed. A credible next test needs either genuine partial observability
with state aliasing (two different causal situations that are
indistinguishable to the agent, not just "the cue is observed at a
different time," which is still fully disambiguated by the tabular state
index) or function approximation with shared parameters -- and, given
`linear_feature_ablation_experiment.py`'s chain, specifically needs to
rule out "brittle linear artifact" as the explanation before counting a
positive result as evidence of real stabilization.
