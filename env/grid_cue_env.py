import random
from collections import deque


class GridCueEnv:
    ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def __init__(
        self,
        grid_size=10,
        num_cues=0,
        reward_sparse_prob=1.0,
        delay=0,
        noise=0.0,
        max_steps=200,
        seed=0,
        cues_enabled=True,
    ):
        self.grid_size = grid_size
        self.num_cues = num_cues
        self.reward_sparse_prob = reward_sparse_prob
        self.delay = delay
        self.noise = noise
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.seed = seed
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self.cues_enabled = cues_enabled
        self.cues = self._make_cues(num_cues)
        self.pending_rewards = deque([0.0] * delay, maxlen=max(1, delay + 1))
        self.reset()

    @property
    def state_size(self):
        return 6 + self.num_cues

    @property
    def action_size(self):
        return len(self.ACTIONS)

    def clone_for_eval(self, cues_enabled=True):
        env = GridCueEnv(
            grid_size=self.grid_size,
            num_cues=self.num_cues,
            reward_sparse_prob=self.reward_sparse_prob,
            delay=self.delay,
            noise=self.noise,
            max_steps=self.max_steps,
            seed=self.seed,
            cues_enabled=cues_enabled,
        )
        env.cues = list(self.cues)
        return env

    def _make_cues(self, n):
        forbidden = {self.start, self.goal}
        cues = []
        while len(cues) < n:
            pos = (self.rng.randrange(self.grid_size), self.rng.randrange(self.grid_size))
            if pos not in forbidden and pos not in cues:
                cues.append(pos)
        return cues

    def reset(self):
        self.agent_pos = self.start
        self.steps = 0
        self.done = False
        self.pending_rewards = deque([0.0] * self.delay, maxlen=max(1, self.delay + 1))
        return self.observe()

    def observe(self):
        x, y = self.agent_pos
        size = max(1, self.grid_size - 1)
        gx, gy = self.goal
        nearest_dx, nearest_dy, nearest_dist = 0.0, 0.0, 1.0
        cue_bits = [0.0] * self.num_cues
        if self.cues_enabled and self.cues:
            distances = [abs(x - cx) + abs(y - cy) for cx, cy in self.cues]
            nearest = min(range(len(distances)), key=lambda i: distances[i])
            cx, cy = self.cues[nearest]
            nearest_dx = (cx - x) / size
            nearest_dy = (cy - y) / size
            nearest_dist = distances[nearest] / (2 * size)
            if distances[nearest] <= 1:
                cue_bits[nearest] = 1.0
        return [
            x / size,
            y / size,
            (gx - x) / size,
            (gy - y) / size,
            nearest_dx,
            nearest_dy,
            nearest_dist,
        ][:6] + cue_bits

    def step(self, action):
        if self.done:
            raise RuntimeError("step called after episode completed")
        if self.rng.random() < self.noise:
            action = self.rng.randrange(len(self.ACTIONS))
        dx, dy = self.ACTIONS[action]
        x, y = self.agent_pos
        nx = min(self.grid_size - 1, max(0, x + dx))
        ny = min(self.grid_size - 1, max(0, y + dy))
        self.agent_pos = (nx, ny)
        self.steps += 1

        immediate = -0.01
        reached = self.agent_pos == self.goal
        if reached and self.rng.random() <= self.reward_sparse_prob:
            immediate += 1.0

        if self.delay:
            self.pending_rewards.append(immediate)
            reward = self.pending_rewards.popleft()
        else:
            reward = immediate

        self.done = reached or self.steps >= self.max_steps
        if self.done and self.delay:
            reward += sum(self.pending_rewards)
            self.pending_rewards.clear()
        return self.observe(), reward, self.done, {"position": self.agent_pos, "reached_goal": reached}
