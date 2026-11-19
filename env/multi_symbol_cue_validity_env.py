"""Multi-symbol generalization of env/cue_validity_env.py::CueValidityEnv.

Design rationale: docs/multi_symbol_cue_validity_env.md. Built in response
to docs/symbolic_stabilization_audit.md Section 4's proposed computational
definition of symbolic stabilization, which requires at least two
independently-trackable symbols with independently-set validities within
the same environment -- something CueValidityEnv (one cue) cannot express.
A new, separate class rather than a modification of CueValidityEnv, so
every one of this project's existing, already-validated experiments
(dozens of scripts construct CueValidityEnv directly) carries zero risk of
regression.

Every design choice below reuses CueValidityEnv's own conventions
unchanged wherever a unique new one isn't required by having more than one
symbol: one shared hidden state, the same reward mechanism, the same
reward_noise knob, the same validity_to_probs formula per symbol, the same
decorrelate()/reverse() intervention vocabulary (now addressable per
symbol or across all symbols at once). metrics/cue_validity.py's existing
proxy_dependence and cue_dependence_score functions are reused completely
unmodified -- they take explicit action arguments with no assumption about
total action count, so probing one symbol's own Q-value effect just means
calling them once per symbol with that symbol's own cue-action id and
every other action (all other symbols' cue actions, plus neutral) folded
into "neutral" for that specific probe. No metric code needed to change.
"""

import random

ACTION_CAUSAL = 0


def validity_to_probs(cue_validity):
    """Identical to CueValidityEnv's own function; duplicated rather than
    imported so this module has no dependency on CueValidityEnv at all,
    keeping the two environments fully independent."""
    cue_validity = max(0.0, min(1.0, cue_validity))
    p_given_1 = 0.5 + 0.45 * cue_validity
    p_given_0 = 0.5 - 0.45 * cue_validity
    return p_given_1, p_given_0


class MultiSymbolCueValidityEnv:
    """A fixed-horizon POMDP with one causal action and N independently-
    valid, independently-interveneable symbolic cues, each with its own
    never-rewarding cue-associated action.

    - `hidden_state` is sampled i.i.d. Bernoulli(0.5) each episode and is
      never included in the observation -- shared by every symbol, so
      "symbol i's validity" means P(symbol i's cue agrees with this one,
      shared hidden_state), not N independent hidden states. This keeps
      the causal mechanism (one causal action, correct iff hidden_state==1)
      identical in meaning to CueValidityEnv; only the number and
      independence of the *non-causal* predictors changes.
    - Each symbol's cue is an independent per-episode Bernoulli draw,
      observed only at `cue_phase`, with its own probabilities set by its
      own entry in `cue_validities` via `validity_to_probs`.
    - Reward is delivered only if ACTION_CAUSAL is chosen at `target_phase`
      and hidden_state == 1 (subject to `reward_noise`, identical mechanism
      to CueValidityEnv). No cue, and no cue-associated action for any
      symbol, ever affects reward, transitions, or action availability, in
      any condition -- fixed for the lifetime of the environment.
    - `decorrelate(symbol_id)` and `reverse(symbol_id)` intervene on ONE
      symbol at a time (symbol_id=None intervenes on all of them, matching
      CueValidityEnv's single-symbol behavior as the N=1 special case).
      This is the one genuinely new capability relative to CueValidityEnv:
      it lets an experiment invalidate one symbol's predictive value while
      another, equally-valid symbol remains fully informative, which is
      the minimum requirement for asking whether persistence depends on
      *which* symbol was invalidated, not just on that symbol's own
      validity (docs/symbolic_stabilization_audit.md Section 4).
    """

    def __init__(self, cue_validities, cue_phase=3, target_phase=4, horizon_slack=10, step_penalty=-0.002, reward_noise=0.0, seed=0):
        if len(cue_validities) < 1:
            raise ValueError("MultiSymbolCueValidityEnv requires at least one symbol; use CueValidityEnv for the single-symbol case if preferred")
        self.cue_validities = list(cue_validities)
        self.num_symbols = len(self.cue_validities)
        self.cue_phase = cue_phase
        self.target_phase = target_phase
        self.horizon = target_phase + horizon_slack
        self.step_penalty = step_penalty
        self.reward_noise = reward_noise
        self.rng = random.Random(seed)
        self._decorrelated = [False] * self.num_symbols
        self._reversed = [False] * self.num_symbols
        self.reset()

    # ---- action space ----

    def cue_action(self, symbol_id):
        """The fixed action index associated with symbol `symbol_id`
        (never-rewarding, for any symbol, under any condition)."""
        self._check_symbol_id(symbol_id)
        return 1 + symbol_id

    @property
    def neutral_actions(self):
        """Two neutral actions, same convention as CueValidityEnv (which
        has exactly two), placed after every symbol's cue action."""
        base = 1 + self.num_symbols
        return (base, base + 1)

    def available_actions(self):
        return (ACTION_CAUSAL,) + tuple(self.cue_action(i) for i in range(self.num_symbols)) + self.neutral_actions

    @property
    def action_count(self):
        return 1 + self.num_symbols + 2

    def _check_symbol_id(self, symbol_id):
        if not (0 <= symbol_id < self.num_symbols):
            raise ValueError(f"symbol_id {symbol_id} out of range for {self.num_symbols} symbols")

    # ---- interventions ----

    def decorrelate(self, symbol_id=None):
        """Invalidate one symbol's predictive value (symbol_id given) or
        every symbol's at once (symbol_id=None, matching CueValidityEnv's
        single-symbol decorrelate())."""
        if symbol_id is None:
            self._decorrelated = [True] * self.num_symbols
        else:
            self._check_symbol_id(symbol_id)
            self._decorrelated[symbol_id] = True

    def reverse(self, symbol_id=None):
        """Flip one symbol's correlation sign (symbol_id given) or every
        symbol's at once (symbol_id=None). Mutually exclusive with
        decorrelate() per symbol -- never call both on the same symbol_id."""
        if symbol_id is None:
            self._reversed = [True] * self.num_symbols
        else:
            self._check_symbol_id(symbol_id)
            self._reversed[symbol_id] = True

    def is_decorrelated(self, symbol_id):
        self._check_symbol_id(symbol_id)
        return self._decorrelated[symbol_id]

    def is_reversed(self, symbol_id):
        self._check_symbol_id(symbol_id)
        return self._reversed[symbol_id]

    # ---- episode mechanics ----

    def reset(self):
        self.phase = 0
        self.done = False
        self.hidden_state = int(self.rng.random() < 0.5)
        self.cue_present = []
        for i in range(self.num_symbols):
            p_given_1, p_given_0 = validity_to_probs(self.cue_validities[i])
            if self._decorrelated[i]:
                p_cue = 0.5
            elif self._reversed[i]:
                p_cue = p_given_0 if self.hidden_state else p_given_1
            else:
                p_cue = p_given_1 if self.hidden_state else p_given_0
            self.cue_present.append(int(self.rng.random() < p_cue))
        self.causal_action_taken = False
        self.reward_obtained = False
        return self.observe()

    def observe(self):
        """Tabular-style discrete state index: phase * (2**num_symbols) +
        the symbols' cue bits packed as a binary number (all zero outside
        cue_phase, matching CueValidityEnv's single-bit convention).
        Provided for completeness and parity with CueValidityEnv's own
        interface; function-approximation agents should use
        observation_vector() below, as elsewhere in this project."""
        bits = 0
        if self.phase == self.cue_phase:
            for i, present in enumerate(self.cue_present):
                bits |= present << i
        return self.phase * (2 ** self.num_symbols) + bits

    def observation_vector(self):
        """Continuous feature encoding for function-approximation agents:
        [phase_norm, cue_bit_0, ..., cue_bit_{N-1}, target_flag,
        pre_target_flag, sin_phase, cos_phase] -- the same five
        phase/cue-bit/trig features CueValidityEnv's observation_vector()
        uses, generalized from one cue bit to N."""
        import math

        at_cue_phase = self.phase == self.cue_phase
        cue_bits = [float(present) if at_cue_phase else 0.0 for present in self.cue_present]
        phase_norm = self.phase / self.target_phase
        return (
            [phase_norm]
            + cue_bits
            + [
                1.0 if self.phase == self.target_phase else 0.0,
                1.0 if self.phase < self.target_phase else 0.0,
                math.sin(phase_norm * math.pi),
                math.cos(phase_norm * math.pi),
            ]
        )

    @property
    def feature_dim(self):
        return self.num_symbols + 5

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
                    withheld = self.reward_noise > 0.0 and self.rng.random() < self.reward_noise
                    if not withheld:
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
