"""PPO (Proximal Policy Optimization), Schulman et al. 2017.

Actor-critic network over continuous feature vectors (the same
observation_vector() style used throughout env/cue_validity_env.py and
env/cue_validity_grid_env.py), trained with generalized advantage
estimation (GAE, Schulman et al. 2016) and the clipped surrogate
objective. This replaces the previous stub (random action selection,
no learning) with a real, tested implementation.

Usage: collect a rollout by calling act()/store() across whole episodes,
then call update() once the rollout is complete. update() consumes and
clears the internal buffer. policy_logits_for_features() exposes the raw
policy-head logits for the same exact-probe metric-reading methodology
(metrics/cue_validity.py) used by every other agent in this repository --
passing logits where those functions expect "Q-values" is correct because
cue_dependence_score/proxy_dependence apply their own softmax internally.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorCriticNet(nn.Module):
    def __init__(self, input_dim, action_dim=4, hidden_size=32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_size, action_dim)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor_head(features), self.critic_head(features).squeeze(-1)


class PPOAgent:
    def __init__(
        self,
        feature_dim,
        action_dim=4,
        seed=0,
        lr=3e-4,
        gamma=0.95,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs=4,
        minibatch_size=32,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        hidden_size=32,
    ):
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.device = torch.device("cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.net = ActorCriticNet(feature_dim, action_dim, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self._reset_buffer()

    def _reset_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def act(self, state):
        """Sample an action from the current policy; also records the
        transition's state/action/log-prob/value for the next update()
        call. Callers must still append the resulting reward and done
        flag via record_outcome()."""
        state_t = torch.tensor([state], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, value = self.net(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.states.append(state)
        self.actions.append(int(action.item()))
        self.log_probs.append(float(log_prob.item()))
        self.values.append(float(value.item()))
        return int(action.item())

    def record_outcome(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def buffer_size(self):
        return len(self.states)

    def _compute_gae(self, last_value):
        n = len(self.rewards)
        advantages = [0.0] * n
        gae = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            mask = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + self.gamma * next_value * mask - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = self.values[t]
        returns = [a + v for a, v in zip(advantages, self.values)]
        return advantages, returns

    def update(self, last_value=0.0):
        """Performs the PPO update over the collected rollout, then
        clears the buffer. `last_value` is the critic's estimate of the
        state the rollout was cut off at (0.0 is correct whenever the
        rollout ends exactly on an episode boundary, which every caller
        in this repository guarantees)."""
        n = len(self.states)
        if n == 0:
            return
        advantages, returns = self._compute_gae(last_value)

        states_t = torch.tensor(self.states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        adv_std = advantages_t.std(unbiased=False)
        if adv_std > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / (adv_std + 1e-8)

        indices = list(range(n))
        for _ in range(self.epochs):
            self.rng.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                batch_idx = indices[start : start + self.minibatch_size]
                batch_idx_t = torch.tensor(batch_idx, dtype=torch.long, device=self.device)

                logits, values = self.net(states_t[batch_idx_t])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t[batch_idx_t])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[batch_idx_t])
                batch_adv = advantages_t[batch_idx_t]
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns_t[batch_idx_t])

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self._reset_buffer()

    def policy_logits_for_features(self, features):
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32, device=self.device)
            logits, _ = self.net(x)
            return logits.squeeze(0).tolist()
