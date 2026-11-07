"""A structurally different second environment for cross-environment
replication of the cue-validity persistence pattern.

Design rationale (full pre-registration: docs/cue_validity_cross_environment.md):

Every positive result in this lineage so far comes from one environment
family, `env/cue_validity_env.py::CueValidityEnv`, varied only by parameter
(validity, representation, intervention) -- never by task structure. This
module is a second, structurally different task that reuses the same
causal mechanism (one designated causal action pays off only when an
unobserved, per-episode hidden state is 1; one cue-associated action never
pays off; a per-episode cue correlates with the hidden state at a swept
`cue_validity`; `decorrelate()`/`reverse()` apply the same interventions)
but embeds the decision inside real, multi-step 2D navigation instead of a
linear phase countdown that ignores the agent's actions until the single
decision step. Concretely: `CueValidityEnv.step()` ignores the chosen
action entirely before `target_phase` (it is a pure countdown); here, the
agent must actually navigate a grid to reach the decision cell, so arriving
there at all requires learning real spatial credit assignment, not just
counting steps. The single causal mechanism (cue_validity -> hidden_state
correlation -> one-shot causal/non-causal action choice -> reward) is held
identical in spirit to `CueValidityEnv`, by design: this experiment is
testing whether the *persistence* pattern replicates in a different task
structure, not inventing a new causal story to test.

Reused directly, not reimplemented: `validity_to_probs` and the
decorrelate()/reverse() semantics from `env/cue_validity_env.py`; the
ACTION_CAUSAL=0 / ACTION_CUE=1 / NEUTRAL_ACTIONS=(2,3) action-index
convention, so `metrics/cue_validity.py`'s functions work unchanged; the
step-penalty convention from `env/grid_cue_env.py`.
"""

import math
import random

from env.cue_validity_env import validity_to_probs

ACTION_CAUSAL = 0  # moves toward the decision cell's "up" neighbor
ACTION_CUE = 1  # moves toward the decision cell's "down" neighbor
NEUTRAL_ACTIONS = (2, 3)  # left, right
ACTION_COUNT = 4
_DELTAS = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # up, down, left, right


class CueValidityGridEnv:
    """A grid-navigation task with one designated decision cell.

    - `hidden_state` is sampled i.i.d. Bernoulli(0.5) each episode, never
      observed directly.
    - A single per-episode cue bit is drawn with probability set by
      `cue_validity` via `validity_to_probs` -- identical semantics to
      `CueValidityEnv`.
    - The agent starts at `start` and moves on a `grid_size` x `grid_size`
      grid (4-connected, boundary-clamped) with a per-step penalty. Movement
      before reaching `decision_cell` has no other consequence: the chosen
      action just moves the agent (or is absorbed at a wall), the episode
      does not end, and the cue bit in the observation is 0.
    - Once the agent's position equals `decision_cell`, the episode ends
      immediately regardless of which action is chosen there: reward is
      `step_penalty + 1.0` if the action taken was ACTION_CAUSAL and
      hidden_state == 1, else just `step_penalty`. This mirrors
      `CueValidityEnv.step()`'s target-phase logic exactly, with the grid
      position taking the place of the phase counter. The cue bit is
      visible in the observation only when the agent is at
      `decision_cell` -- the same "cue appears exactly where the decision
      is made" rule `CueValidityEnv` uses (with `cue_phase == target_phase`),
      which that lineage's own MVE found was structurally required for a
      tabular agent to show any dependence at all.
    - `max_steps` truncates episodes that never reach the decision cell
      (reward is just the accumulated step penalty; no bonus, no
      additional penalty).
    """

    def __init__(self, cue_validity, grid_size=5, horizon_slack=12, step_penalty=-0.01, seed=0):
        self.cue_validity = cue_validity
        self.grid_size = grid_size
        self.start = (0, 0)
        self.decision_cell = (grid_size - 1, grid_size - 1)
        manhattan = abs(self.decision_cell[0] - self.start[0]) + abs(self.decision_cell[1] - self.start[1])
        self.max_steps = manhattan + horizon_slack
        self.step_penalty = step_penalty
        self.rng = random.Random(seed)
        self._decorrelated = False
        self._reversed = False
        self.reset()

    @property
    def state_count(self):
        return self.grid_size * self.grid_size * 2

    def decorrelate(self):
        """Same semantics as CueValidityEnv.decorrelate(): the cue stops
        carrying any information about hidden_state from this call
        forward. Nothing else changes."""
        self._decorrelated = True

    def reverse(self):
        """Same semantics as CueValidityEnv.reverse(): the cue's
        correlation with hidden_state flips sign from this call forward,
        same magnitude as during acquisition. Mutually exclusive with
        decorrelate()."""
        self._reversed = True

    @property
    def is_decorrelated(self):
        return self._decorrelated

    @property
    def is_reversed(self):
        return self._reversed

    def reset(self):
        self.pos = self.start
        self.steps = 0
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

    def at_decision_cell(self):
        return self.pos == self.decision_cell

    def observe(self):
        x, y = self.pos
        cue_bit = self.cue_present if self.at_decision_cell() else 0
        return (x * self.grid_size + y) * 2 + cue_bit

    def observation_vector(self):
        """Continuous feature encoding for function-approximation agents,
        in place of the integer observe() state -- same spirit as
        CueValidityEnv.observation_vector()."""
        x, y = self.pos
        norm = max(1, self.grid_size - 1)
        dx, dy = self.decision_cell
        cue_bit = float(self.cue_present) if self.at_decision_cell() else 0.0
        return [
            x / norm,
            y / norm,
            cue_bit,
            1.0 if self.at_decision_cell() else 0.0,
            (dx - x) / norm,
            (dy - y) / norm,
        ]

    def available_actions(self):
        return (ACTION_CAUSAL, ACTION_CUE) + NEUTRAL_ACTIONS

    def step(self, action):
        if self.done:
            raise RuntimeError("step called after episode termination")
        if self.at_decision_cell():
            reward = self.step_penalty
            if action == ACTION_CAUSAL:
                self.causal_action_taken = True
                if self.hidden_state == 1:
                    reward += 1.0
                    self.reward_obtained = True
            self.done = True
            return self.observe(), reward, self.done, {
                "causal_action_taken": self.causal_action_taken,
                "reward_obtained": self.reward_obtained,
                "position": self.pos,
            }

        dx, dy = _DELTAS[action]
        x, y = self.pos
        nx = min(self.grid_size - 1, max(0, x + dx))
        ny = min(self.grid_size - 1, max(0, y + dy))
        self.pos = (nx, ny)
        self.steps += 1
        reward = self.step_penalty
        self.done = self.steps >= self.max_steps
        return self.observe(), reward, self.done, {
            "causal_action_taken": False,
            "reward_obtained": False,
            "position": self.pos,
        }
