import numpy as np
from env.dynamics import DynamicsModel
from env.cues import CueLayer
from env.reward import RewardSystem
from env.gridworld import GridWorld

class RLWrapper:
    """
    Unified environment wrapper for RL agents.
    """

    def __init__(self, config):
        self.grid = GridWorld(size=config["grid_size"])

        self.dynamics = DynamicsModel(
            grid_size=config["grid_size"],
            noise=config["noise"]
        )

        self.cues = CueLayer(
            grid_size=config["grid_size"],
            num_cues=config["num_cues"]
        )

        self.reward = RewardSystem(
            sparse_prob=config["reward_sparse_prob"]
        )

        self.delay = config.get("delay", 0)
        self.delay_buffer = []

    def reset(self):
        obs = self.grid.reset()
        return self._get_obs(obs)

    def step(self, action):
        # apply stochastic dynamics
        self.grid.agent_pos = self.dynamics.transition(
            self.grid.agent_pos,
            action
        )

        # check goal
        reached = (self.grid.agent_pos == self.grid.goal_pos)

        # reward computation
        r = self.reward.compute(reached)

        # delayed reward buffer (uncertainty mechanism)
        self.delay_buffer.append(r)
        if len(self.delay_buffer) > self.delay:
            r = self.delay_buffer.pop(0)
        else:
            r = 0

        obs = self.grid._get_obs()

        return self._get_obs(obs), r, reached, {}

    def _get_obs(self, obs):
        """
        Concatenate:
        - agent position
        - cue map (flattened)
        """

        cue_map = self.cues.get_cue_map().flatten()

        return np.concatenate([
            np.array(obs),
            cue_map
        ])

    def remove_cues(self):
        self.cues.remove_cues()

    def shuffle_cues(self):
        self.cues = CueLayer(self.grid.size, self.cues.num_cues)
