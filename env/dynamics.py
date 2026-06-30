import numpy as np

class DynamicsModel:
    """
    Handles stochastic transitions and environmental noise.
    """

    def __init__(self, grid_size=10, noise=0.1):
        self.grid_size = grid_size
        self.noise = noise

    def transition(self, pos, action):
        """
        Applies noisy movement dynamics.
        """

        x, y = pos

        # stochastic action flip (uncertainty injection)
        if np.random.rand() < self.noise:
            action = np.random.randint(0, 4)

        if action == 0:   # up
            y += 1
        elif action == 1: # down
            y -= 1
        elif action == 2: # left
            x -= 1
        elif action == 3: # right
            x += 1

        # boundary constraints
        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)

        return [x, y]
