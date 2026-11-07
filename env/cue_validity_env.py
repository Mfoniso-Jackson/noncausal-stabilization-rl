"""Shared, pre-registered environment for the cue-validity MVE.

Design rationale (see docs/cue_validity_mve.md for the full pre-registration):

In this whole research program the cue is *never* causal in any condition --
only the hidden-state -> causal-action -> reward mechanism is, and that
mechanism is held fixed across every condition and every intervention. The
single independent variable manipulated during acquisition is how reliably
the cue correlates with the (always-unobserved) hidden reward state,
expressed as one continuous parameter, `cue_validity` in [0, 1]:

    cue_validity = 0.0  ->  cue carries no information (P(cue=1) = 0.5
                             regardless of the hidden state)
    cue_validity = 1.0  ->  cue almost perfectly tracks the hidden state
                             (P(cue=1|state=1) = 0.95, P(cue=1|state=0) = 0.05)

and interpolates linearly between those endpoints. This replaces the
ad-hoc, inconsistently-named (reward_prob, cue_given_reward) pairs used by
the pre-existing one-off experiment scripts in this repository with a single
swept parameter, so the same environment can express every condition in the
MVE's design matrix without a new class per condition.

After an acquisition period, `decorrelate()` invalidates the cue's
predictive value going forward (P(cue=1) reset to 0.5 regardless of hidden
state) while leaving every other part of the environment -- the hidden-state
distribution, the causal reward mechanism, the action space, the transition
structure -- completely unchanged. This is the "decorrelate" intervention
named in the MVE's design: the literal operationalization of "an
intervention invalidates the cue's predictive value" from the research
brief, as distinct from reversing the correlation's sign or removing the
cue-associated action's availability (both already covered by other,
earlier experiments in this repository).
"""

import random

ACTION_CAUSAL = 0
ACTION_CUE = 1
NEUTRAL_ACTIONS = (2, 3)
ACTION_COUNT = 4


def validity_to_probs(cue_validity):
    """Map a single cue_validity in [0, 1] to (p_cue_given_state1, p_cue_given_state0)."""
    cue_validity = max(0.0, min(1.0, cue_validity))
    p_given_1 = 0.5 + 0.45 * cue_validity
    p_given_0 = 0.5 - 0.45 * cue_validity
    return p_given_1, p_given_0


class CueValidityEnv:
    """A fixed-horizon POMDP with one causal action and one cue-correlated
    (but never causal) action/cue pair.

    - `hidden_state` is sampled i.i.d. Bernoulli(0.5) each episode and is
      never included in the observation.
    - The cue is a single per-episode Bernoulli draw, observed only at
      `cue_phase`, with probabilities set by `cue_validity` via
      `validity_to_probs`.
    - Reward is delivered only if action ACTION_CAUSAL is chosen at
      `target_phase` and hidden_state == 1. Neither the cue nor
      ACTION_CUE ever affects reward, transitions, or action availability,
      in any condition -- this is fixed for the lifetime of the environment
      and is not touched by `decorrelate()`.
    """

    def __init__(self, cue_validity, cue_phase=3, target_phase=4, horizon_slack=10, step_penalty=-0.002, seed=0):
        self.cue_validity = cue_validity
        self.cue_phase = cue_phase
        self.target_phase = target_phase
        self.horizon = target_phase + horizon_slack
        self.step_penalty = step_penalty
        self.rng = random.Random(seed)
        self._decorrelated = False
        self._reversed = False
        self.reset()

    def decorrelate(self):
        """Apply the intervention: the cue stops carrying any information
        about hidden_state from this call forward. Nothing else changes."""
        self._decorrelated = True

    def reverse(self):
        """Apply the alternative intervention: the cue's correlation with
        hidden_state flips sign (P(cue=1|state=1) and P(cue=1|state=0) are
        swapped) from this call forward, with the same magnitude it had
        during acquisition. Nothing else changes. Mutually exclusive with
        decorrelate() -- only call one of the two on a given environment."""
        self._reversed = True

    @property
    def is_decorrelated(self):
        return self._decorrelated

    @property
    def is_reversed(self):
        return self._reversed

    def reset(self):
        self.phase = 0
        self.done = False
        self.hidden_state = int(self.rng.random() < 0.5)
        p_given_1, p_given_0 = validity_to_probs(self.cue_validity)
        if self._decorrelated:
            p_cue = 0.5
        elif self._reversed:
            p_cue = p_given_0 if self.hidden_state else p_given_1
        else:
            p_cue = p_given_1 if self.hidden_state else p_given_0
        self.cue_present = int(self.rng.random() < p_cue)
        self.causal_action_taken = False
        self.reward_obtained = False
        return self.observe()

    def observe(self):
        cue_bit = self.cue_present if self.phase == self.cue_phase else 0
        return self.phase * 2 + cue_bit

    def observation_vector(self):
        """Continuous feature encoding for function-approximation agents,
        in place of the discrete observe() state index. Same 6 features
        used by this repository's pre-existing recurrent/linear-DQN
        experiments (phase_norm, cue bit, target flag, pre-target flag,
        sin/cos phase), so a linear or neural agent here is directly
        comparable to the tabular agent's use of observe()."""
        import math

        cue_bit = float(self.cue_present) if self.phase == self.cue_phase else 0.0
        phase_norm = self.phase / self.target_phase
        return [
            phase_norm,
            cue_bit,
            1.0 if self.phase == self.target_phase else 0.0,
            1.0 if self.phase < self.target_phase else 0.0,
            math.sin(phase_norm * math.pi),
            math.cos(phase_norm * math.pi),
        ]

    def available_actions(self):
        return (ACTION_CAUSAL, ACTION_CUE) + NEUTRAL_ACTIONS

    def cue_available(self):
        return self.phase == self.cue_phase

    def step(self, action):
        if self.done:
            raise RuntimeError("step called after episode termination")
        reward = self.step_penalty
        if self.phase == self.target_phase:
            if action == ACTION_CAUSAL:
                self.causal_action_taken = True
                if self.hidden_state == 1:
                    reward += 1.0
                    self.reward_obtained = True
            self.done = True
        else:
            self.phase += 1
            self.done = self.phase >= self.horizon
        return self.observe(), reward, self.done, {
            "causal_action_taken": self.causal_action_taken,
            "reward_obtained": self.reward_obtained,
        }
