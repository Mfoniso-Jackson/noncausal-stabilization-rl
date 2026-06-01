import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)
      
class DQNAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_size=50000,
        batch_size=64,
        target_update_freq=500
    ):

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Q networks
        self.q_net = QNetwork(obs_dim, action_dim)
        self.target_net = QNetwork(obs_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # replay buffer
        self.memory = deque(maxlen=buffer_size)

        self.step_count = 0    
    def act(self, obs):
        """
        Epsilon-greedy policy
        """

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        obs = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(obs)

        return int(torch.argmax(q_values).item())

      def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # current Q values
        q_values = self.q_net(states).gather(1, actions)

        # target Q values
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # loss
        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

     def update(self, transition):
        """
        transition = (state, action, reward)
        NOTE: next_state + done should be handled in training loop
        """

        state, action, reward, next_state, done = transition
        self.store(state, action, reward, next_state, done)
        self.train_step()     
