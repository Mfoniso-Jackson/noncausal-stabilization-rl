import numpy as np

class RewardSystem:
    def __init__(self, sparse_prob=0.1):
        self.sparse_prob = sparse_prob

    def compute(self, reached_goal):
        if reached_goal and np.random.rand() < self.sparse_prob:
            return 1
        return 0
