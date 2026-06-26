import math
import random
from collections import deque


class DQNAgent:
    """Small dependency-free DQN-style agent with replay and target weights."""

    def __init__(
        self,
        state_size,
        action_size,
        seed=0,
        lr=0.04,
        gamma=0.96,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.992,
        replay_size=5000,
        batch_size=16,
        target_update=25,
        train_frequency=4,
        cue_exploration_bias=0.0,
        cue_regularization=0.0,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.rng = random.Random(seed)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.train_frequency = train_frequency
        self.cue_exploration_bias = cue_exploration_bias
        self.cue_regularization = cue_regularization
        self.train_steps = 0
        self.env_steps = 0
        self.replay = deque(maxlen=replay_size)
        self.weights = [[self.rng.uniform(-0.03, 0.03) for _ in range(state_size + 1)] for _ in range(action_size)]
        self.target_weights = [row[:] for row in self.weights]

    def q_values(self, state, target=False):
        weights = self.target_weights if target else self.weights
        features = state + [1.0]
        values = []
        for row in weights:
            total = 0.0
            for i, x in enumerate(features):
                total += row[i] * x
            values.append(total)
        return values

    def act(self, state, training=True):
        if training and self.rng.random() < self.epsilon:
            if self.cue_exploration_bias > 0.0 and self.rng.random() < self.cue_exploration_bias:
                cue_action = self._cue_directed_action(state)
                if cue_action is not None:
                    return cue_action
            return self.rng.randrange(self.action_size)
        qs = self.q_values(state)
        return max(range(self.action_size), key=lambda a: qs[a])

    def _cue_directed_action(self, state):
        if len(state) < 6:
            return None
        dx = state[4]
        dy = state[5]
        if abs(dx) + abs(dy) < 1e-9:
            return None
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2
        return 1 if dy > 0 else 0

    def remember(self, state, action, reward, next_state, done):
        self.replay.append((state, action, reward, next_state, done))

    def train_step(self):
        self.env_steps += 1
        if self.env_steps % self.train_frequency != 0:
            return 0.0
        if len(self.replay) < self.batch_size:
            return 0.0
        replay_len = len(self.replay)
        batch = [self.replay[self.rng.randrange(replay_len)] for _ in range(self.batch_size)]
        total_loss = 0.0
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * max(self.q_values(next_state, target=True))
            pred = self.q_values(state)[action]
            error = max(-2.0, min(2.0, pred - target))
            total_loss += error * error
            features = state + [1.0]
            for i, x in enumerate(features):
                self.weights[action][i] -= self.lr * error * x / self.batch_size
            if self.cue_regularization:
                for row in self.weights:
                    for i in range(4, self.state_size):
                        row[i] *= 1.0 - self.cue_regularization
        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target_weights = [row[:] for row in self.weights]
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return math.sqrt(total_loss / len(batch))

    def state_dict(self):
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "epsilon": self.epsilon,
            "cue_exploration_bias": self.cue_exploration_bias,
            "cue_regularization": self.cue_regularization,
            "weights": self.weights,
            "target_weights": self.target_weights,
        }

    def load_state_dict(self, data):
        self.epsilon = data["epsilon"]
        self.weights = [list(row) for row in data["weights"]]
        self.target_weights = [list(row) for row in data["target_weights"]]
