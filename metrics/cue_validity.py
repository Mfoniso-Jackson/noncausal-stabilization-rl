"""Single, shared CDS/SPI definitions for the cue-validity MVE.

See docs/cue_validity_mve.md for the pre-registered design these serve.
This module exists so the experiment script does not redefine these
metrics itself -- every other experiment script in this repository
reimplements its own ad hoc version of "CDS" and "SPI", and the audit that
preceded this module found those versions mutually inconsistent across
scripts. These definitions are meant to be imported, not copied.
"""

import math


def softmax_mass(q_values, available_actions):
    """Softmax action-probability mass over exactly `available_actions`."""
    values = [q_values[a] for a in available_actions]
    max_q = max(values)
    exp_values = [math.exp(v - max_q) for v in values]
    denom = sum(exp_values)
    return {a: ev / denom for a, ev in zip(available_actions, exp_values)}


def cue_dependence_score(q_values, available_actions, causal_action, cue_action, neutral_actions):
    """CDS: softmax mass on the non-causal actions (the cue-associated
    action plus any neutral actions) at a cue-exposed decision state.

    0.0 means the policy places all value on the causal action; a value
    near (len(available_actions) - 1) / len(available_actions) is the
    softmax tie-breaking floor for an untrained/uninformative Q-function,
    not evidence of dependence -- always compare against a cue_validity=0
    control run at the same scale, never against zero in isolation.
    """
    masses = softmax_mass(q_values, available_actions)
    non_causal = [a for a in available_actions if a != causal_action]
    assert set(non_causal) == set([cue_action, *neutral_actions]), "available_actions must be exactly causal + cue + neutral"
    return sum(masses[a] for a in non_causal)


def causal_action_rate(q_values, available_actions, causal_action):
    """Fraction of softmax mass on the causal action -- used as the
    'rational use of the cue' readout: this is allowed and expected to
    vary with cue_validity during acquisition (Bayesian updating on an
    informative cue), and the interesting question is whether it still
    varies with the cue after decorrelate() has made that variation
    irrational."""
    masses = softmax_mass(q_values, available_actions)
    return masses[causal_action]


def proxy_dependence(q_values_cue_present, q_values_cue_absent, available_actions, causal_action):
    """This repository's established operationalization of cue dependence
    (see e.g. experiments/causal_reversal_agent_experiment.py,
    experiments/neural_recurrent_reversal_experiment.py): the difference in
    how likely the policy is to choose the causal action when the cue is
    present versus absent, at the same decision point. Positive values mean
    the policy chooses the causal action more often when the cue is
    present. This is a different, complementary construct to
    cue_dependence_score (CDS): CDS asks whether the policy places value on
    the non-causal action itself (which never pays off, so cannot acquire
    stable value under tabular TD learning); proxy_dependence asks whether
    the policy's correct-action probability is still modulated by the cue,
    which can persist under any agent class as ordinary slow re-convergence
    after a distribution shift, with no requirement for a never-rewarded
    action to acquire value."""
    return causal_action_rate(q_values_cue_present, available_actions, causal_action) - causal_action_rate(
        q_values_cue_absent, available_actions, causal_action
    )


def persistence_index(pre_cds, post_cds):
    """SPI: post-intervention CDS relative to pre-intervention CDS.

    Returns None (not zero, not NaN silently swallowed) when pre_cds is
    too small to normalize by meaningfully -- callers must handle this
    explicitly rather than treat it as a numeric zero.
    """
    if pre_cds <= 1e-9:
        return None
    return max(0.0, post_cds / pre_cds)


def extinction_half_life(block_cds_sequence, baseline_cds):
    """First index in block_cds_sequence whose value falls to half of
    baseline_cds or below; len(block_cds_sequence) if it never does
    (censored), 0 if baseline_cds itself is already negligible."""
    if baseline_cds <= 1e-9:
        return 0
    threshold = 0.5 * baseline_cds
    for index, value in enumerate(block_cds_sequence):
        if value <= threshold:
            return index
    return len(block_cds_sequence)
