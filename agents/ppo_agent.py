import numpy as np

class PPOAgent:
    def __init__(self, obs_dim, action_dim):
        self.policy = {}

    def act(self, obs):
        return np.random.randint(0, 4)

    def update(self, batch):
        pass
