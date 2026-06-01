import numpy as np

class GridWorld:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        return self._get_obs()

    def step(self, action):
        self._move(action)
        reward = self._get_reward()
        obs = self._get_obs()
        done = self.agent_pos == self.goal_pos
        return obs, reward, done, {}

    def _move(self, action):
        x, y = self.agent_pos
        if action == 0: y += 1
        if action == 1: y -= 1
        if action == 2: x -= 1
        if action == 3: x += 1

        self.agent_pos = [
            np.clip(x, 0, self.size-1),
            np.clip(y, 0, self.size-1)
        ]

    def _get_obs(self):
        return np.array(self.agent_pos)

    def _get_reward(self):
        return 1 if self.agent_pos == self.goal_pos else 0
