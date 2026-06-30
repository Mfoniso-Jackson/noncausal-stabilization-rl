import numpy as np


class GridWorld:
    def __init__(self, size=10, max_steps=200):
        self.size = size
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.steps = 0
        return self._get_obs()

    def step(self, action):
        self.steps += 1

        self._move(action)

        reward = self._get_reward()
        obs = self._get_obs()

        done = (
            self.agent_pos == self.goal_pos
            or self.steps >= self.max_steps
        )

        info = {
            "steps": self.steps
        }

        return obs, reward, done, info

    def _move(self, action):
        x, y = self.agent_pos

        if action == 0:
            y += 1
        elif action == 1:
            y -= 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1

        self.agent_pos = [
            int(np.clip(x, 0, self.size - 1)),
            int(np.clip(y, 0, self.size - 1))
        ]

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.float32)

    def _get_reward(self):
        return 1.0 if self.agent_pos == self.goal_pos else 0.0
