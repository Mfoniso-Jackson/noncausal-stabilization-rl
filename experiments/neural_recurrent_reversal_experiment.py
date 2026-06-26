import math
import os
import random
import statistics
from collections import deque

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from agents.dqn_agent import DQNAgent
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.dqn_agent import DQNAgent


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

ACTION_CAUSAL = 0
ACTION_PROXY = 1
NEUTRAL_ACTIONS = [2, 3]
ACTION_COUNT = 4
OBS_DIM = 6
HORIZON = 14
TARGET_PHASE = 10
HIDDEN_REWARD_PROB = 0.5
STEP_PENALTY = -0.002

PHASES = [
    ("acquisition", 3000, 0.95, 0.05),
    ("reversal", 1500, 0.05, 0.95),
    ("extinction", 1500, 0.50, 0.50),
]
PHASE_BOUNDARIES = [3000, 4500]
EVAL_INTERVAL = 100
EVAL_EPISODES = 300
SEEDS = list(range(10))
AGENT_TYPES = ["tabular_sarsa", "linear_dqn", "neural_dqn", "recurrent_dqn"]

ALPHA = 0.08
GAMMA = 0.95
LAMBDA = 0.80
EPSILON_START = 1.0
EPSILON_END = 0.04
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50_000
TARGET_UPDATE_INTERVAL = 200
TRAIN_EVERY_STEPS = 16
SEQUENCE_LENGTH = 5
MIN_PERSISTENCE_DENOMINATOR = 0.05


torch.set_num_threads(1)


class HiddenProxyReversalEnv:
    def __init__(self, p_proxy_given_reward, p_proxy_given_no_reward, seed=0):
        self.p_proxy_given_reward = p_proxy_given_reward
        self.p_proxy_given_no_reward = p_proxy_given_no_reward
        self.rng = random.Random(seed)
        self.reset()

    def set_proxy_probs(self, p_proxy_given_reward, p_proxy_given_no_reward):
        self.p_proxy_given_reward = p_proxy_given_reward
        self.p_proxy_given_no_reward = p_proxy_given_no_reward

    def reset(self):
        self.phase = 0
        self.done = False
        self.hidden_reward_state = int(self.rng.random() < HIDDEN_REWARD_PROB)
        self.reward_obtained = False
        self.proxy_by_phase = [self._sample_proxy() for _ in range(HORIZON + 1)]
        return self.observe()

    def _sample_proxy(self):
        prob = self.p_proxy_given_reward if self.hidden_reward_state else self.p_proxy_given_no_reward
        return int(self.rng.random() < prob)

    def observe(self):
        return self.phase * 2 + self.proxy_by_phase[self.phase]

    def observation_vector(self):
        phase_norm = self.phase / TARGET_PHASE
        proxy = self.proxy_by_phase[self.phase]
        return [
            phase_norm,
            float(proxy),
            1.0 if self.phase == TARGET_PHASE else 0.0,
            1.0 if self.phase < TARGET_PHASE else 0.0,
            math.sin(self.phase / max(1, TARGET_PHASE) * math.pi),
            math.cos(self.phase / max(1, TARGET_PHASE) * math.pi),
        ]

    def step(self, action):
        if self.done:
            raise RuntimeError("step called after episode termination")
        reward = STEP_PENALTY
        if self.phase == TARGET_PHASE:
            if action == ACTION_CAUSAL and self.hidden_reward_state == 1:
                reward += 1.0
                self.reward_obtained = True
            self.done = True
        else:
            self.phase += 1
            self.done = self.phase >= HORIZON
        return self.observe(), reward, self.done, {}


class TabularSarsaAgent:
    def __init__(self, seed):
        self.obs_count = (HORIZON + 1) * 2
        self.rng = random.Random(seed)
        self.q = [[0.0 for _ in range(ACTION_COUNT)] for _ in range(self.obs_count)]

    def train_episode(self, env, epsilon):
        traces = {}
        obs = env.reset()
        action = self.act(obs, epsilon)
        done = False
        while not done:
            next_obs, reward, done, _ = env.step(action)
            if done:
                target = reward
                next_action = None
            else:
                next_action = self.act(next_obs, epsilon)
                target = reward + GAMMA * self.q[next_obs][next_action]
            delta = target - self.q[obs][action]
            traces[(obs, action)] = traces.get((obs, action), 0.0) + 1.0
            dead = []
            for key, value in traces.items():
                s, a = key
                self.q[s][a] += ALPHA * delta * value
                new_value = value * GAMMA * LAMBDA
                if new_value < 1e-8:
                    dead.append(key)
                else:
                    traces[key] = new_value
            for key in dead:
                traces.pop(key, None)
            obs = next_obs
            action = next_action if next_action is not None else ACTION_CAUSAL

    def act(self, obs, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(ACTION_COUNT)
        return self.greedy_action(obs)

    def greedy_action(self, obs):
        values = self.q[obs]
        best = max(values)
        best_actions = [idx for idx, value in enumerate(values) if abs(value - best) < 1e-12]
        return self.rng.choice(best_actions)

    def begin_eval_episode(self, env):
        return None

    def observe_eval_step(self, env):
        return None

    def q_values_for_env(self, env):
        return self.q[env.observe()]

    def greedy_action_for_env(self, env):
        return self.greedy_action(env.observe())


class LinearDqnWrapper:
    def __init__(self, seed, total_steps):
        steps = max(1, total_steps / 4)
        epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / steps)
        self.agent = DQNAgent(
            state_size=OBS_DIM,
            action_size=ACTION_COUNT,
            seed=seed,
            lr=0.015,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay=epsilon_decay,
            replay_size=8000,
            batch_size=32,
            target_update=50,
            train_frequency=4,
        )
        self.rng = random.Random(seed)

    def train_episode(self, env, epsilon_unused):
        env.reset()
        state = env.observation_vector()
        done = False
        while not done:
            action = self.agent.act(state, training=True)
            _, reward, done, _ = env.step(action)
            next_state = env.observation_vector()
            self.agent.remember(state, action, reward, next_state, done)
            self.agent.train_step()
            state = next_state

    def begin_eval_episode(self, env):
        return None

    def observe_eval_step(self, env):
        return None

    def q_values_for_env(self, env):
        return self.agent.q_values(env.observation_vector())

    def greedy_action_for_env(self, env):
        values = self.q_values_for_env(env)
        return random_argmax(values, self.rng)


class MLPQNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_COUNT),
        )

    def forward(self, x):
        return self.net(x)


class GRUQNet(nn.Module):
    def __init__(self, input_dim, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, ACTION_COUNT)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class TorchDqnAgent:
    def __init__(self, seed, recurrent=False):
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.recurrent = recurrent
        self.device = torch.device("cpu")
        self.online = GRUQNet(OBS_DIM).to(self.device) if recurrent else MLPQNet(OBS_DIM).to(self.device)
        self.target = GRUQNet(OBS_DIM).to(self.device) if recurrent else MLPQNet(OBS_DIM).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.env_steps = 0
        self.history = deque(maxlen=SEQUENCE_LENGTH)

    def _current_state(self, env):
        if not self.recurrent:
            return env.observation_vector()
        return [list(x) for x in self.history]

    def _reset_history(self, env):
        self.history.clear()
        obs = env.observation_vector()
        for _ in range(SEQUENCE_LENGTH):
            self.history.append(list(obs))

    def _append_history(self, env):
        if self.recurrent:
            self.history.append(list(env.observation_vector()))

    def train_episode(self, env, epsilon):
        env.reset()
        self._reset_history(env)
        done = False
        while not done:
            state = self._current_state(env)
            action = self._epsilon_action(state, epsilon)
            _, reward, done, _ = env.step(action)
            self._append_history(env)
            next_state = self._current_state(env)
            self.replay.append((state, action, reward, next_state, done))
            self.env_steps += 1
            if self.env_steps % TRAIN_EVERY_STEPS == 0:
                self._train_step()
            if self.env_steps % TARGET_UPDATE_INTERVAL == 0:
                self.target.load_state_dict(self.online.state_dict())

    def _epsilon_action(self, state, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(ACTION_COUNT)
        return random_argmax(self._q_values(state), self.rng)

    def _q_values(self, state):
        with torch.no_grad():
            if self.recurrent:
                x = torch.tensor([state], dtype=torch.float32, device=self.device)
            else:
                x = torch.tensor([state], dtype=torch.float32, device=self.device)
            values = self.online(x).cpu().tolist()[0]
        return values

    def _train_step(self):
        if len(self.replay) < BATCH_SIZE:
            return
        batch = [self.replay[self.rng.randrange(len(self.replay))] for _ in range(BATCH_SIZE)]
        states, actions, rewards, next_states, dones = zip(*batch)
        if self.recurrent:
            states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
            next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        else:
            states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
            next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        q_pred = self.online(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            q_next = self.target(next_states_t).max(dim=1).values
            q_target = rewards_t + (1.0 - dones_t) * GAMMA * q_next
        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()

    def begin_eval_episode(self, env):
        self._reset_history(env)

    def observe_eval_step(self, env):
        self._append_history(env)

    def q_values_for_env(self, env):
        return self._q_values(self._current_state(env))

    def greedy_action_for_env(self, env):
        return random_argmax(self.q_values_for_env(env), self.rng)


def random_argmax(values, rng):
    best = max(values)
    best_actions = [idx for idx, value in enumerate(values) if abs(value - best) < 1e-12]
    return rng.choice(best_actions)


def train_and_evaluate(agent_type, seed):
    total_episodes = sum(episodes for _, episodes, _, _ in PHASES)
    total_steps = total_episodes * (TARGET_PHASE + 1)
    if agent_type == "tabular_sarsa":
        agent = TabularSarsaAgent(seed)
    elif agent_type == "linear_dqn":
        agent = LinearDqnWrapper(seed, total_steps)
    elif agent_type == "neural_dqn":
        agent = TorchDqnAgent(seed, recurrent=False)
    elif agent_type == "recurrent_dqn":
        agent = TorchDqnAgent(seed, recurrent=True)
    else:
        raise ValueError(agent_type)

    env = HiddenProxyReversalEnv(PHASES[0][2], PHASES[0][3], seed=seed)
    rows = []
    epsilon = EPSILON_START
    epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / total_episodes)
    global_episode = 0
    for phase_name, phase_episodes, p1, p0 in PHASES:
        env.set_proxy_probs(p1, p0)
        for phase_episode in range(1, phase_episodes + 1):
            agent.train_episode(env, epsilon)
            epsilon = max(EPSILON_END, epsilon * epsilon_decay)
            global_episode += 1
            if phase_episode == 1 or phase_episode % EVAL_INTERVAL == 0 or phase_episode == phase_episodes:
                metrics = evaluate(agent, p1, p0, seed + global_episode * 17)
                rows.append(
                    {
                        "agent_type": agent_type,
                        "seed": seed,
                        "phase": phase_name,
                        "phase_episode": phase_episode,
                        "global_episode": global_episode,
                        "p_proxy_given_reward": p1,
                        "p_proxy_given_no_reward": p0,
                        **metrics,
                    }
                )
    return rows


def evaluate(agent, p1, p0, seed):
    env = HiddenProxyReversalEnv(p1, p0, seed=seed + 70000)
    totals = init_counts()
    cue_counts = {0: init_counts(), 1: init_counts()}
    cds_values, q_proxy_values, q_neutral_values, q_causal_values = [], [], [], []
    for _ in range(EVAL_EPISODES):
        env.reset()
        agent.begin_eval_episode(env)
        done = False
        while not done:
            cue = env.observe() % 2
            values = agent.q_values_for_env(env)
            action = random_argmax(values, agent.rng)
            if env.phase < TARGET_PHASE:
                record_action(totals, action)
                record_action(cue_counts[cue], action)
                masses = softmax(values)
                cds_values.append(masses[ACTION_PROXY] + sum(masses[a] for a in NEUTRAL_ACTIONS))
                q_proxy_values.append(values[ACTION_PROXY])
                q_neutral_values.append(mean(values[a] for a in NEUTRAL_ACTIONS))
                q_causal_values.append(values[ACTION_CAUSAL])
            _, reward, done, _ = env.step(action)
            totals["reward_sum"] += reward
            agent.observe_eval_step(env)
        totals["goal_sum"] += int(env.reward_obtained)
    q_proxy = mean(q_proxy_values)
    q_neutral = mean(q_neutral_values)
    q_causal = mean(q_causal_values)
    p_causal_cue1 = rate(cue_counts[1], "causal")
    p_causal_cue0 = rate(cue_counts[0], "causal")
    p_proxy_cue1 = rate(cue_counts[1], "proxy")
    p_proxy_cue0 = rate(cue_counts[0], "proxy")
    return {
        "reward": round(totals["reward_sum"] / EVAL_EPISODES, 6),
        "goal_rate": round(totals["goal_sum"] / EVAL_EPISODES, 6),
        "SPI": round((totals["proxy"] + totals["neutral"]) / max(1, totals["decisions"]), 6),
        "CDS": round(mean(cds_values), 6),
        "causal_action_rate": round(rate(totals, "causal"), 6),
        "proxy_action_rate": round(rate(totals, "proxy"), 6),
        "neutral_action_rate": round(rate(totals, "neutral"), 6),
        "Q_causal": round(q_causal, 6),
        "Q_proxy": round(q_proxy, 6),
        "Q_neutral_mean": round(q_neutral, 6),
        "proxy_Q_advantage": round(q_proxy - q_neutral, 6),
        "proxy_vs_causal_Q": round(q_proxy - q_causal, 6),
        "proxy_dependence": round(p_causal_cue1 - p_causal_cue0, 6),
        "abs_proxy_dependence": round(abs(p_causal_cue1 - p_causal_cue0), 6),
        "proxy_action_dependence": round(p_proxy_cue1 - p_proxy_cue0, 6),
    }


def evaluate_random_baseline(seed):
    rows = []
    rng = random.Random(seed + 900000)
    for phase_name, _, p1, p0 in PHASES:
        env = HiddenProxyReversalEnv(p1, p0, seed=seed + 910000)
        totals = init_counts()
        for _ in range(EVAL_EPISODES):
            env.reset()
            done = False
            while not done:
                action = rng.randrange(ACTION_COUNT)
                if env.phase < TARGET_PHASE:
                    record_action(totals, action)
                _, reward, done, _ = env.step(action)
                totals["reward_sum"] += reward
            totals["goal_sum"] += int(env.reward_obtained)
        rows.append(
            {
                "phase": phase_name,
                "seed": seed,
                "random_reward": round(totals["reward_sum"] / EVAL_EPISODES, 6),
                "random_goal_rate": round(totals["goal_sum"] / EVAL_EPISODES, 6),
                "random_proxy_action_rate": round(rate(totals, "proxy"), 6),
                "random_causal_action_rate": round(rate(totals, "causal"), 6),
            }
        )
    return rows


def init_counts():
    return {"decisions": 0, "causal": 0, "proxy": 0, "neutral": 0, "reward_sum": 0.0, "goal_sum": 0}


def record_action(counts, action):
    counts["decisions"] += 1
    if action == ACTION_CAUSAL:
        counts["causal"] += 1
    elif action == ACTION_PROXY:
        counts["proxy"] += 1
    elif action in NEUTRAL_ACTIONS:
        counts["neutral"] += 1


def rate(counts, key):
    return counts[key] / max(1, counts["decisions"])


def softmax(values):
    max_q = max(values)
    exp_values = [math.exp(max(-60.0, min(60.0, value - max_q))) for value in values]
    denom = sum(exp_values)
    return [value / denom for value in exp_values]


def mean(values):
    values = list(values)
    return sum(values) / max(1, len(values))


def sem(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def ci95(values):
    return 1.96 * sem(values)


def summarize(rows):
    df = pd.DataFrame(rows)
    final_rows = []
    for agent in AGENT_TYPES:
        for phase, _, _, _ in PHASES:
            subset = df[(df.agent_type == agent) & (df.phase == phase)]
            final_episode = subset.phase_episode.max()
            final_rows.append(subset[subset.phase_episode == final_episode])
    final_df = pd.concat(final_rows, ignore_index=True)
    summary_rows = []
    for (agent, phase), group in final_df.groupby(["agent_type", "phase"], sort=False):
        item = {"agent_type": agent, "phase": phase, "checkpoint": "final", "n": len(group)}
        for metric in metric_names():
            values = list(group[metric].astype(float))
            item[f"{metric}_mean"] = round(mean(values), 6)
            item[f"{metric}_ci95"] = round(ci95(values), 6)
        summary_rows.append(item)
    return summary_rows


def aggregate_curve(rows):
    df = pd.DataFrame(rows)
    curve_rows = []
    for keys, group in df.groupby(["agent_type", "phase", "phase_episode", "global_episode"], sort=False):
        agent, phase, phase_episode, global_episode = keys
        item = {"agent_type": agent, "phase": phase, "phase_episode": phase_episode, "global_episode": global_episode, "n": len(group)}
        for metric in metric_names():
            values = list(group[metric].astype(float))
            item[f"{metric}_mean"] = round(mean(values), 6)
            item[f"{metric}_ci95"] = round(ci95(values), 6)
        curve_rows.append(item)
    return curve_rows


def persistence_summary(rows):
    output = []
    for agent_type in AGENT_TYPES:
        for seed in SEEDS:
            subset = [r for r in rows if r["agent_type"] == agent_type and r["seed"] == seed]
            acq_final = get_checkpoint(subset, "acquisition", "final")
            rev_initial = get_checkpoint(subset, "reversal", "initial")
            rev_final = get_checkpoint(subset, "reversal", "final")
            ext_initial = get_checkpoint(subset, "extinction", "initial")
            ext_final = get_checkpoint(subset, "extinction", "final")
            acq_abs = abs(acq_final["proxy_dependence"])
            early_ext = [r for r in subset if r["phase"] == "extinction" and r["phase_episode"] <= 500]
            residual = mean(abs(r["proxy_dependence"]) for r in early_ext)
            persistence_index = (
                residual / acq_abs if acq_abs >= MIN_PERSISTENCE_DENOMINATOR else float("nan")
            )
            output.append(
                {
                    "agent_type": agent_type,
                    "seed": seed,
                    "acquisition_proxy_dependence_final": round(acq_final["proxy_dependence"], 6),
                    "reversal_proxy_dependence_initial": round(rev_initial["proxy_dependence"], 6),
                    "reversal_proxy_dependence_final": round(rev_final["proxy_dependence"], 6),
                    "extinction_proxy_dependence_initial": round(ext_initial["proxy_dependence"], 6),
                    "extinction_proxy_dependence_final": round(ext_final["proxy_dependence"], 6),
                    "reversal_adaptation_half_life": adaptation_half_life(subset, "reversal", acq_abs),
                    "extinction_half_life": adaptation_half_life(subset, "extinction", abs(ext_initial["proxy_dependence"])),
                    "superstition_persistence_index": round(persistence_index, 6) if math.isfinite(persistence_index) else float("nan"),
                }
            )
    return output


def summarize_persistence(persistence):
    metrics = [
        "acquisition_proxy_dependence_final",
        "reversal_proxy_dependence_initial",
        "reversal_proxy_dependence_final",
        "extinction_proxy_dependence_initial",
        "extinction_proxy_dependence_final",
        "reversal_adaptation_half_life",
        "extinction_half_life",
        "superstition_persistence_index",
    ]
    rows = []
    for agent in AGENT_TYPES:
        subset = [r for r in persistence if r["agent_type"] == agent]
        item = {"agent_type": agent, "n": len(subset)}
        for metric in metrics:
            if metric.endswith("half_life"):
                values = [float(r[metric]) for r in subset if math.isfinite(float(r[metric])) and float(r[metric]) >= 0]
            else:
                values = [float(r[metric]) for r in subset if math.isfinite(float(r[metric]))]
            item[f"{metric}_mean"] = round(mean(values), 6) if values else float("nan")
            item[f"{metric}_ci95"] = round(ci95(values), 6) if values else float("nan")
        rows.append(item)
    return rows


def get_checkpoint(rows, phase, which):
    subset = [r for r in rows if r["phase"] == phase]
    episode = min(r["phase_episode"] for r in subset) if which == "initial" else max(r["phase_episode"] for r in subset)
    return next(r for r in subset if r["phase_episode"] == episode)


def adaptation_half_life(rows, phase, start_abs):
    if start_abs <= 1e-9:
        return 0
    target = start_abs * 0.5
    for row in sorted([r for r in rows if r["phase"] == phase], key=lambda r: r["phase_episode"]):
        if abs(row["proxy_dependence"]) <= target:
            return int(row["phase_episode"])
    return -1


def comparisons(rows, persistence, baselines):
    output = []
    for contrast in [("neural_dqn", "tabular_sarsa"), ("neural_dqn", "linear_dqn"), ("recurrent_dqn", "neural_dqn")]:
        high, low = contrast
        for phase, _, _, _ in PHASES:
            for metric in ["reward", "goal_rate", "abs_proxy_dependence", "proxy_dependence", "proxy_action_rate", "proxy_Q_advantage"]:
                high_values, low_values = [], []
                for seed in SEEDS:
                    high_rows = [r for r in rows if r["agent_type"] == high and r["seed"] == seed and r["phase"] == phase]
                    low_rows = [r for r in rows if r["agent_type"] == low and r["seed"] == seed and r["phase"] == phase]
                    h = max(high_rows, key=lambda r: r["phase_episode"])
                    l = max(low_rows, key=lambda r: r["phase_episode"])
                    high_values.append(float(h[metric]))
                    low_values.append(float(l[metric]))
                output.append(diff_row(f"{high}_vs_{low}", phase, metric, high_values, low_values))
        for metric in ["reversal_adaptation_half_life", "extinction_half_life", "superstition_persistence_index"]:
            high_values = [r[metric] for r in persistence if r["agent_type"] == high and math.isfinite(float(r[metric])) and r[metric] >= 0]
            low_values = [r[metric] for r in persistence if r["agent_type"] == low and math.isfinite(float(r[metric])) and r[metric] >= 0]
            output.append(diff_row(f"{high}_vs_{low}", "persistence", metric, high_values, low_values, paired=False))
    for agent in AGENT_TYPES:
        for phase, _, _, _ in PHASES:
            random_reward = mean(r["random_reward"] for r in baselines if r["phase"] == phase)
            random_goal = mean(r["random_goal_rate"] for r in baselines if r["phase"] == phase)
            for metric, baseline in [("reward", random_reward), ("goal_rate", random_goal)]:
                values = []
                for seed in SEEDS:
                    subset = [r for r in rows if r["agent_type"] == agent and r["seed"] == seed and r["phase"] == phase]
                    values.append(max(subset, key=lambda r: r["phase_episode"])[metric])
                output.append(one_sample_row(agent, f"{phase}_vs_random", metric, values, baseline))
    return output


def diff_row(contrast, comparison, metric, high_values, low_values, paired=True):
    high_values = [float(v) for v in high_values if math.isfinite(float(v))]
    low_values = [float(v) for v in low_values if math.isfinite(float(v))]
    n = min(len(high_values), len(low_values))
    if n == 0:
        diff, se, t, p = float("nan"), float("nan"), float("nan"), float("nan")
    elif paired:
        diffs = [float(high_values[i]) - float(low_values[i]) for i in range(n)]
        diff = mean(diffs)
        se = sem(diffs)
        t = diff / se if se else 0.0
        p = math.erfc(abs(t) / math.sqrt(2.0))
    else:
        diff = mean(high_values) - mean(low_values)
        se = math.sqrt(sem(high_values) ** 2 + sem(low_values) ** 2)
        t = diff / se if se else 0.0
        p = math.erfc(abs(t) / math.sqrt(2.0))
    return {"comparison": comparison, "agent_or_contrast": contrast, "metric": metric, "difference": round(diff, 6), "ci95": round(1.96 * se, 6), "t_stat": round(t, 6), "p_value_normal": round(p, 6)}


def one_sample_row(agent, comparison, metric, values, baseline):
    diffs = [float(value) - baseline for value in values]
    diff = mean(diffs)
    se = sem(diffs)
    t = diff / se if se else 0.0
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return {"comparison": comparison, "agent_or_contrast": agent, "metric": metric, "difference": round(diff, 6), "ci95": round(1.96 * se, 6), "t_stat": round(t, 6), "p_value_normal": round(p, 6)}


def fmt(value, digits=3):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def metric_names():
    return [
        "reward",
        "goal_rate",
        "SPI",
        "CDS",
        "causal_action_rate",
        "proxy_action_rate",
        "neutral_action_rate",
        "Q_causal",
        "Q_proxy",
        "Q_neutral_mean",
        "proxy_Q_advantage",
        "proxy_vs_causal_Q",
        "proxy_dependence",
        "abs_proxy_dependence",
        "proxy_action_dependence",
    ]


def write_plots(curve_rows):
    curve = pd.DataFrame(curve_rows)
    plot_specs = [
        ("reward_mean", "Reward", "neural_recurrent_reward_curve.png"),
        ("proxy_dependence_mean", "Proxy Dependence", "neural_recurrent_proxy_dependence_curve.png"),
        ("abs_proxy_dependence_mean", "Absolute Proxy Dependence", "neural_recurrent_abs_proxy_dependence_curve.png"),
        ("proxy_action_rate_mean", "Proxy Action Rate", "neural_recurrent_proxy_action_curve.png"),
        ("proxy_Q_advantage_mean", "Proxy Q Advantage", "neural_recurrent_q_advantage_curve.png"),
    ]
    for metric, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for agent in AGENT_TYPES:
            subset = curve[curve.agent_type == agent].sort_values("global_episode")
            ax.plot(subset.global_episode, subset[metric], label=agent, linewidth=1.8)
        for boundary in PHASE_BOUNDARIES:
            ax.axvline(boundary, color="0.55", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Global training episode")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=160)
        plt.close(fig)


def write_report(summary, persistence_stats, comparisons_rows, baselines):
    summary_map = {(r["agent_type"], r["phase"]): r for r in summary}
    persist_map = {r["agent_type"]: r for r in persistence_stats}
    neural_ext = summary_map[("neural_dqn", "extinction")]
    recurrent_ext = summary_map[("recurrent_dqn", "extinction")]
    linear_ext = summary_map[("linear_dqn", "extinction")]
    tab_ext = summary_map[("tabular_sarsa", "extinction")]
    supports_proxy = max(abs(neural_ext["proxy_dependence_mean"]), abs(recurrent_ext["proxy_dependence_mean"])) > abs(tab_ext["proxy_dependence_mean"]) + 0.20
    supports_action = max(neural_ext["proxy_action_rate_mean"], recurrent_ext["proxy_action_rate_mean"]) > tab_ext["proxy_action_rate_mean"] + 0.10
    recurrent_reduces = recurrent_ext["abs_proxy_dependence_mean"] < neural_ext["abs_proxy_dependence_mean"] - 0.10
    lines = [
        "# Neural and Recurrent Causal Reversal Experiment",
        "",
        "## Research question",
        "Does neural function approximation and recurrence increase persistent proxy reliance after a predictive cue becomes reversed and then random?",
        "",
        "## Why this replicates the previous positive result",
        "The previous dependency-free linear DQN showed strong proxy dependence through acquisition, reversal, and extinction. This replication keeps the same hidden-state environment and phase schedule while adding PyTorch neural and recurrent DQN agents.",
        "",
        "## Environment design",
        f"Hidden-state POMDP. hidden_reward_state is sampled but never included in observation. Reward can occur only at phase {TARGET_PHASE} when hidden_reward_state=1 and action 0 is selected.",
        "",
        "## Phase design",
        "Acquisition uses 0.95/0.05 proxy probabilities. Reversal swaps them to 0.05/0.95. Extinction decorrelates the cue at 0.50/0.50.",
        "",
        "## Agent architectures",
        "tabular_sarsa uses SARSA(lambda). linear_dqn is the earlier dependency-free replay/target-weight baseline. neural_dqn is a PyTorch MLP with two 64-unit ReLU layers. recurrent_dqn is a PyTorch GRU over rolling observation histories of length 5.",
        "",
        "## Training details",
        f"Seeds={len(SEEDS)}, acquisition=3000 episodes, reversal=1500, extinction=1500, evaluation every {EVAL_INTERVAL} episodes with {EVAL_EPISODES} episodes. Neural agents use replay buffer size {REPLAY_BUFFER_SIZE}, batch size {BATCH_SIZE}, gamma={GAMMA}, learning_rate={LEARNING_RATE}, target update every {TARGET_UPDATE_INTERVAL} environment steps.",
        "",
        "## Metrics",
        "Metrics include reward, goal_rate, SPI, CDS, action rates, Q summaries, proxy_Q_advantage, proxy_vs_causal_Q, proxy_dependence, absolute proxy dependence, and proxy_action_dependence.",
        "",
        "## Results tables",
        "",
        "| agent | phase | reward | goal_rate | proxy_dep | abs_proxy_dep | proxy_action_rate | Q_adv |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['agent_type']} | {row['phase']} | "
            f"{fmt(row['reward_mean'])} +/- {fmt(row['reward_ci95'])} | "
            f"{fmt(row['goal_rate_mean'])} +/- {fmt(row['goal_rate_ci95'])} | "
            f"{fmt(row['proxy_dependence_mean'])} +/- {fmt(row['proxy_dependence_ci95'])} | "
            f"{fmt(row['abs_proxy_dependence_mean'])} +/- {fmt(row['abs_proxy_dependence_ci95'])} | "
            f"{fmt(row['proxy_action_rate_mean'])} +/- {fmt(row['proxy_action_rate_ci95'])} | "
            f"{fmt(row['proxy_Q_advantage_mean'])} +/- {fmt(row['proxy_Q_advantage_ci95'])} |"
        )
    lines.extend(["", "## Learning curves summary", "Saved plots include reward, signed proxy dependence, absolute proxy dependence, proxy action rate, and proxy Q advantage. Vertical markers show reversal and extinction boundaries.", "", "## Persistence analysis", f"Persistence index is reported as NA when absolute acquisition-end proxy dependence is below {MIN_PERSISTENCE_DENOMINATOR}, because the normalization denominator is too small for a meaningful ratio.", "", "| agent | acq_dep | rev_initial | rev_final | ext_initial | ext_final | rev_half | ext_half | persistence_index |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in persistence_stats:
        lines.append(
            f"| {row['agent_type']} | "
            f"{fmt(row['acquisition_proxy_dependence_final_mean'])} +/- {fmt(row['acquisition_proxy_dependence_final_ci95'])} | "
            f"{fmt(row['reversal_proxy_dependence_initial_mean'])} +/- {fmt(row['reversal_proxy_dependence_initial_ci95'])} | "
            f"{fmt(row['reversal_proxy_dependence_final_mean'])} +/- {fmt(row['reversal_proxy_dependence_final_ci95'])} | "
            f"{fmt(row['extinction_proxy_dependence_initial_mean'])} +/- {fmt(row['extinction_proxy_dependence_initial_ci95'])} | "
            f"{fmt(row['extinction_proxy_dependence_final_mean'])} +/- {fmt(row['extinction_proxy_dependence_final_ci95'])} | "
            f"{fmt(row['reversal_adaptation_half_life_mean'], 1)} | "
            f"{fmt(row['extinction_half_life_mean'], 1)} | "
            f"{fmt(row['superstition_persistence_index_mean'])} +/- {fmt(row['superstition_persistence_index_ci95'])} |"
        )
    lines.extend(["", "## Agent comparison", "", "| comparison | contrast | metric | difference | 95% CI | p approx |", "|---|---|---|---:|---:|---:|"])
    for row in comparisons_rows:
        lines.append(f"| {row['comparison']} | {row['agent_or_contrast']} | {row['metric']} | {fmt(row['difference'], 4)} | +/- {fmt(row['ci95'], 4)} | {fmt(row['p_value_normal'], 4)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            f"Extinction signed proxy dependence: tabular={fmt(tab_ext['proxy_dependence_mean'])}, linear={fmt(linear_ext['proxy_dependence_mean'])}, neural={fmt(neural_ext['proxy_dependence_mean'])}, recurrent={fmt(recurrent_ext['proxy_dependence_mean'])}.",
            f"Extinction absolute proxy dependence: tabular={fmt(tab_ext['abs_proxy_dependence_mean'])}, linear={fmt(linear_ext['abs_proxy_dependence_mean'])}, neural={fmt(neural_ext['abs_proxy_dependence_mean'])}, recurrent={fmt(recurrent_ext['abs_proxy_dependence_mean'])}.",
            "",
            "## Does this support computational superstition?",
            f"Generalized proxy-reliance persistence supported: {supports_proxy}.",
            f"Action-level superstition supported: {supports_action}.",
            f"Recurrent agent appears protective relative to neural DQN: {recurrent_reduces}.",
            "",
            "## Limitations",
            "Approximate p-values use normal approximations. Recurrent replay uses rolling histories rather than full sequence replay with hidden-state burn-in. All neural runs are CPU-only in this workspace.",
            "",
            "## Recommended next experiment",
            "If proxy reliance persists, add a longer extinction window and an explicit anti-proxy control. If recurrent DQN reduces persistence, test longer histories and belief-state supervision to separate memory from false latent-state stabilization.",
            "",
            "## Validity checks",
            f"PyTorch version: {torch.__version__}. CUDA available: {torch.cuda.is_available()}. MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}. Hidden reward state is never included in observation. Proxy action never directly causes reward. Reward only depends on hidden_reward_state and action 0. Phase probabilities are fixed as specified. Action-space size is fixed at 4. Random baseline is included. All agents use the same phase schedule.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "neural_recurrent_reversal_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} mps={hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}", flush=True)
    rows = []
    for agent in AGENT_TYPES:
        for seed in SEEDS:
            print(f"agent={agent} seed={seed}", flush=True)
            rows.extend(train_and_evaluate(agent, seed))
    baselines = []
    for seed in SEEDS:
        baselines.extend(evaluate_random_baseline(seed))
    summary = summarize(rows)
    curve = aggregate_curve(rows)
    persistence = persistence_summary(rows)
    persistence_stats = summarize_persistence(persistence)
    comparison_rows = comparisons(rows, persistence, baselines)

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "neural_recurrent_reversal_results.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_DIR, "neural_recurrent_reversal_summary.csv"), index=False)
    pd.DataFrame(persistence).to_csv(os.path.join(RESULTS_DIR, "neural_recurrent_reversal_persistence.csv"), index=False)
    pd.DataFrame(comparison_rows).to_csv(os.path.join(RESULTS_DIR, "neural_recurrent_reversal_comparisons.csv"), index=False)
    pd.DataFrame(baselines).to_csv(os.path.join(RESULTS_DIR, "neural_recurrent_reversal_baseline.csv"), index=False)
    write_plots(curve)
    write_report(summary, persistence_stats, comparison_rows, baselines)


if __name__ == "__main__":
    run()
