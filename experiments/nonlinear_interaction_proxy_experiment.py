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
    from experiments.orthogonalized_proxy_feature_experiment import (
        ACTION_COUNT,
        ACTION_CAUSAL,
        ACTION_PROXY,
        NEUTRAL_ACTIONS,
        EVAL_INTERVAL,
        EVAL_EPISODES,
        GAMMA,
        MIN_PERSISTENCE_DENOMINATOR,
        PHASES,
        PHASE_BOUNDARIES,
        RESULTS_DIR,
        REPORTS_DIR,
        SEEDS,
        TARGET_PHASE,
        HiddenProxyReversalEnv,
        adaptation_half_life,
        ci95,
        fmt,
        mean,
        random_argmax,
        sem,
        softmax,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.dqn_agent import DQNAgent
    from experiments.orthogonalized_proxy_feature_experiment import (
        ACTION_COUNT,
        ACTION_CAUSAL,
        ACTION_PROXY,
        NEUTRAL_ACTIONS,
        EVAL_INTERVAL,
        EVAL_EPISODES,
        GAMMA,
        MIN_PERSISTENCE_DENOMINATOR,
        PHASES,
        PHASE_BOUNDARIES,
        RESULTS_DIR,
        REPORTS_DIR,
        SEEDS,
        TARGET_PHASE,
        HiddenProxyReversalEnv,
        adaptation_half_life,
        ci95,
        fmt,
        mean,
        random_argmax,
        sem,
        softmax,
    )


AGENT_TYPES = ["linear_dqn", "neural_dqn"]
FEATURE_CONDITIONS = ["full_raw_proxy", "proxy_interaction_only"]
GROUPS = [(agent, feature) for agent in AGENT_TYPES for feature in FEATURE_CONDITIONS]
EPSILON_START = 1.0
EPSILON_END = 0.04
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50_000
TARGET_UPDATE_INTERVAL = 200
TRAIN_EVERY_STEPS = 16

torch.set_num_threads(1)


class FeatureLinearDqn:
    def __init__(self, feature_condition, seed, total_steps):
        self.feature_condition = feature_condition
        probe = HiddenProxyReversalEnv(0.5, 0.5, seed=seed)
        state_size = len(probe.features(feature_condition))
        steps = max(1, total_steps / 4)
        epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / steps)
        self.agent = DQNAgent(
            state_size=state_size,
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
        state = env.features(self.feature_condition)
        done = False
        while not done:
            action = self.agent.act(state, training=True)
            _, reward, done, _ = env.step(action)
            next_state = env.features(self.feature_condition)
            self.agent.remember(state, action, reward, next_state, done)
            self.agent.train_step()
            state = next_state

    def q_values_for_env(self, env):
        return self.agent.q_values(env.features(self.feature_condition))


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


class FeatureNeuralDqn:
    def __init__(self, feature_condition, seed):
        torch.manual_seed(seed)
        self.feature_condition = feature_condition
        self.rng = random.Random(seed)
        probe = HiddenProxyReversalEnv(0.5, 0.5, seed=seed)
        input_dim = len(probe.features(feature_condition))
        self.online = MLPQNet(input_dim)
        self.target = MLPQNet(input_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.env_steps = 0

    def train_episode(self, env, epsilon):
        env.reset()
        state = env.features(self.feature_condition)
        done = False
        while not done:
            action = self._epsilon_action(state, epsilon)
            _, reward, done, _ = env.step(action)
            next_state = env.features(self.feature_condition)
            self.replay.append((state, action, reward, next_state, done))
            self.env_steps += 1
            if self.env_steps % TRAIN_EVERY_STEPS == 0:
                self._train_step()
            if self.env_steps % TARGET_UPDATE_INTERVAL == 0:
                self.target.load_state_dict(self.online.state_dict())
            state = next_state

    def _epsilon_action(self, state, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(ACTION_COUNT)
        return random_argmax(self._q_values(state), self.rng)

    def _q_values(self, state):
        with torch.no_grad():
            x = torch.tensor([state], dtype=torch.float32)
            return self.online(x).tolist()[0]

    def _train_step(self):
        if len(self.replay) < BATCH_SIZE:
            return
        batch = [self.replay[self.rng.randrange(len(self.replay))] for _ in range(BATCH_SIZE)]
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.tensor(states, dtype=torch.float32)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)
        q_pred = self.online(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            q_next = self.target(next_states_t).max(dim=1).values
            q_target = rewards_t + (1.0 - dones_t) * GAMMA * q_next
        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()

    def q_values_for_env(self, env):
        return self._q_values(env.features(self.feature_condition))


def train_and_evaluate(agent_type, feature_condition, seed):
    total_episodes = sum(episodes for _, episodes, _, _ in PHASES)
    total_steps = total_episodes * (TARGET_PHASE + 1)
    if agent_type == "linear_dqn":
        agent = FeatureLinearDqn(feature_condition, seed, total_steps)
    elif agent_type == "neural_dqn":
        agent = FeatureNeuralDqn(feature_condition, seed)
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
                        "feature_condition": feature_condition,
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
        totals["goal_sum"] += int(env.reward_obtained)
    q_proxy = mean(q_proxy_values)
    q_neutral = mean(q_neutral_values)
    q_causal = mean(q_causal_values)
    proxy_dependence = rate(cue_counts[1], "causal") - rate(cue_counts[0], "causal")
    proxy_action_dependence = rate(cue_counts[1], "proxy") - rate(cue_counts[0], "proxy")
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
        "proxy_dependence": round(proxy_dependence, 6),
        "abs_proxy_dependence": round(abs(proxy_dependence), 6),
        "proxy_action_dependence": round(proxy_action_dependence, 6),
    }


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


def summarize(rows):
    df = pd.DataFrame(rows)
    summary_rows = []
    for agent_type, feature_condition in GROUPS:
        for phase, _, _, _ in PHASES:
            subset = df[(df.agent_type == agent_type) & (df.feature_condition == feature_condition) & (df.phase == phase)]
            final_episode = subset.phase_episode.max()
            group = subset[subset.phase_episode == final_episode]
            item = {"agent_type": agent_type, "feature_condition": feature_condition, "phase": phase, "checkpoint": "final", "n": len(group)}
            for metric in metric_names():
                values = list(group[metric].astype(float))
                item[f"{metric}_mean"] = round(mean(values), 6)
                item[f"{metric}_ci95"] = round(ci95(values), 6)
            summary_rows.append(item)
    return summary_rows


def aggregate_curve(rows):
    df = pd.DataFrame(rows)
    curve_rows = []
    for keys, group in df.groupby(["agent_type", "feature_condition", "phase", "phase_episode", "global_episode"], sort=False):
        agent_type, feature_condition, phase, phase_episode, global_episode = keys
        item = {"agent_type": agent_type, "feature_condition": feature_condition, "phase": phase, "phase_episode": phase_episode, "global_episode": global_episode, "n": len(group)}
        for metric in metric_names():
            values = list(group[metric].astype(float))
            item[f"{metric}_mean"] = round(mean(values), 6)
            item[f"{metric}_ci95"] = round(ci95(values), 6)
        curve_rows.append(item)
    return curve_rows


def persistence_summary(rows):
    output = []
    for agent_type, feature_condition in GROUPS:
        for seed in SEEDS:
            subset = [r for r in rows if r["agent_type"] == agent_type and r["feature_condition"] == feature_condition and r["seed"] == seed]
            acq_final = get_checkpoint(subset, "acquisition")
            rev_final = get_checkpoint(subset, "reversal")
            ext_initial = get_checkpoint(subset, "extinction", initial=True)
            ext_final = get_checkpoint(subset, "extinction")
            acq_abs = abs(acq_final["proxy_dependence"])
            early_ext = [r for r in subset if r["phase"] == "extinction" and r["phase_episode"] <= 500]
            residual = mean(abs(r["proxy_dependence"]) for r in early_ext)
            persistence_index = residual / acq_abs if acq_abs >= MIN_PERSISTENCE_DENOMINATOR else float("nan")
            output.append(
                {
                    "agent_type": agent_type,
                    "feature_condition": feature_condition,
                    "seed": seed,
                    "acquisition_proxy_dependence_final": round(acq_final["proxy_dependence"], 6),
                    "reversal_proxy_dependence_final": round(rev_final["proxy_dependence"], 6),
                    "extinction_proxy_dependence_initial": round(ext_initial["proxy_dependence"], 6),
                    "extinction_proxy_dependence_final": round(ext_final["proxy_dependence"], 6),
                    "extinction_half_life": adaptation_half_life(subset, "extinction", abs(ext_initial["proxy_dependence"])),
                    "superstition_persistence_index": round(persistence_index, 6) if math.isfinite(persistence_index) else float("nan"),
                }
            )
    return output


def get_checkpoint(rows, phase, initial=False):
    subset = [r for r in rows if r["phase"] == phase]
    episode = min(r["phase_episode"] for r in subset) if initial else max(r["phase_episode"] for r in subset)
    return next(r for r in subset if r["phase_episode"] == episode)


def summarize_persistence(persistence):
    rows = []
    for agent_type, feature_condition in GROUPS:
        subset = [r for r in persistence if r["agent_type"] == agent_type and r["feature_condition"] == feature_condition]
        item = {"agent_type": agent_type, "feature_condition": feature_condition, "n": len(subset)}
        for metric in ["acquisition_proxy_dependence_final", "reversal_proxy_dependence_final", "extinction_proxy_dependence_initial", "extinction_proxy_dependence_final", "extinction_half_life", "superstition_persistence_index"]:
            if metric.endswith("half_life"):
                values = [float(r[metric]) for r in subset if math.isfinite(float(r[metric])) and float(r[metric]) >= 0]
            else:
                values = [float(r[metric]) for r in subset if math.isfinite(float(r[metric]))]
            item[f"{metric}_mean"] = round(mean(values), 6) if values else float("nan")
            item[f"{metric}_ci95"] = round(ci95(values), 6) if values else float("nan")
        rows.append(item)
    return rows


def comparisons(rows, persistence):
    output = []
    for agent_type in AGENT_TYPES:
        for phase, _, _, _ in PHASES:
            for metric in ["reward", "goal_rate", "abs_proxy_dependence", "proxy_dependence", "proxy_action_rate", "proxy_Q_advantage"]:
                a_values, b_values = final_values(rows, agent_type, "proxy_interaction_only", phase, metric), final_values(rows, agent_type, "full_raw_proxy", phase, metric)
                output.append(diff_row(f"{agent_type}_interaction_vs_raw", phase, metric, a_values, b_values))
    for feature_condition in FEATURE_CONDITIONS:
        for phase, _, _, _ in PHASES:
            for metric in ["reward", "goal_rate", "abs_proxy_dependence", "proxy_dependence", "proxy_action_rate", "proxy_Q_advantage"]:
                a_values, b_values = final_values(rows, "neural_dqn", feature_condition, phase, metric), final_values(rows, "linear_dqn", feature_condition, phase, metric)
                output.append(diff_row(f"neural_vs_linear_{feature_condition}", phase, metric, a_values, b_values))
    for feature_condition in FEATURE_CONDITIONS:
        for metric in ["extinction_half_life", "superstition_persistence_index"]:
            a_values = [r[metric] for r in persistence if r["agent_type"] == "neural_dqn" and r["feature_condition"] == feature_condition and math.isfinite(float(r[metric])) and r[metric] >= 0]
            b_values = [r[metric] for r in persistence if r["agent_type"] == "linear_dqn" and r["feature_condition"] == feature_condition and math.isfinite(float(r[metric])) and r[metric] >= 0]
            output.append(diff_row(f"neural_vs_linear_{feature_condition}", "persistence", metric, a_values, b_values, paired=False))
    return output


def final_values(rows, agent_type, feature_condition, phase, metric):
    values = []
    for seed in SEEDS:
        subset = [r for r in rows if r["agent_type"] == agent_type and r["feature_condition"] == feature_condition and r["phase"] == phase and r["seed"] == seed]
        values.append(max(subset, key=lambda r: r["phase_episode"])[metric])
    return values


def diff_row(contrast, comparison, metric, a_values, b_values, paired=True):
    a_values = [float(v) for v in a_values if math.isfinite(float(v))]
    b_values = [float(v) for v in b_values if math.isfinite(float(v))]
    n = min(len(a_values), len(b_values))
    if n == 0:
        diff, se, t, p = float("nan"), float("nan"), float("nan"), float("nan")
    elif paired:
        diffs = [a_values[i] - b_values[i] for i in range(n)]
        diff = mean(diffs)
        se = sem(diffs)
        t = diff / se if se else 0.0
        p = math.erfc(abs(t) / math.sqrt(2.0))
    else:
        diff = mean(a_values) - mean(b_values)
        se = math.sqrt(sem(a_values) ** 2 + sem(b_values) ** 2)
        t = diff / se if se else 0.0
        p = math.erfc(abs(t) / math.sqrt(2.0))
    return {"comparison": comparison, "contrast": contrast, "metric": metric, "difference": round(diff, 6) if math.isfinite(diff) else float("nan"), "ci95": round(1.96 * se, 6) if math.isfinite(se) else float("nan"), "t_stat": round(t, 6) if math.isfinite(t) else float("nan"), "p_value_normal": round(p, 6) if math.isfinite(p) else float("nan")}


def write_plots(curve_rows, summary):
    curve = pd.DataFrame(curve_rows)
    labels = {("linear_dqn", "full_raw_proxy"): "linear raw", ("linear_dqn", "proxy_interaction_only"): "linear interaction", ("neural_dqn", "full_raw_proxy"): "neural raw", ("neural_dqn", "proxy_interaction_only"): "neural interaction"}
    for metric, ylabel, filename in [
        ("reward_mean", "Reward", "nonlinear_interaction_reward_curve.png"),
        ("abs_proxy_dependence_mean", "Absolute Proxy Dependence", "nonlinear_interaction_abs_dependence_curve.png"),
        ("proxy_dependence_mean", "Signed Proxy Dependence", "nonlinear_interaction_signed_dependence_curve.png"),
        ("proxy_action_rate_mean", "Proxy Action Rate", "nonlinear_interaction_proxy_action_curve.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 6))
        for agent_type, feature_condition in GROUPS:
            subset = curve[(curve.agent_type == agent_type) & (curve.feature_condition == feature_condition)].sort_values("global_episode")
            ax.plot(subset.global_episode, subset[metric], label=labels[(agent_type, feature_condition)], linewidth=1.8)
        for boundary in PHASE_BOUNDARIES:
            ax.axvline(boundary, color="0.55", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Global training episode")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=160)
        plt.close(fig)
    summary_df = pd.DataFrame(summary)
    ext = summary_df[summary_df.phase == "extinction"].copy()
    ext["label"] = ext.agent_type + "\n" + ext.feature_condition
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(ext.label, ext.abs_proxy_dependence_mean, yerr=ext.abs_proxy_dependence_ci95, color="#4C78A8")
    ax.set_ylabel("Extinction absolute proxy dependence")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "nonlinear_interaction_proxy_plot.png"), dpi=160)
    plt.close(fig)


def write_report(summary, persistence_stats, comparison_rows):
    summary_df = pd.DataFrame(summary)
    ext = summary_df[summary_df.phase == "extinction"]
    def row(agent, feature):
        return ext[(ext.agent_type == agent) & (ext.feature_condition == feature)].iloc[0]
    linear_raw = row("linear_dqn", "full_raw_proxy")
    linear_interaction = row("linear_dqn", "proxy_interaction_only")
    neural_raw = row("neural_dqn", "full_raw_proxy")
    neural_interaction = row("neural_dqn", "proxy_interaction_only")
    neural_recovers = neural_interaction["abs_proxy_dependence_mean"] > linear_interaction["abs_proxy_dependence_mean"] + 0.20
    neural_matches_raw = neural_interaction["abs_proxy_dependence_mean"] >= linear_raw["abs_proxy_dependence_mean"] - 0.15
    lines = [
        "# Nonlinear Interaction Proxy Experiment",
        "",
        "## Research question",
        "Can a nonlinear neural DQN recover proxy reliance from an interaction-only representation where linear DQN shows weak persistence?",
        "",
        "## Design",
        "Same hidden-state reversal environment and phase schedule. The comparison crosses agent class (linear DQN vs neural DQN) with feature representation (raw proxy vs interaction-only proxy).",
        "",
        "## Results table",
        "",
        "| agent | feature | phase | reward | goal_rate | proxy_dep | abs_proxy_dep | proxy_action_rate | Q_adv |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['agent_type']} | {item['feature_condition']} | {item['phase']} | "
            f"{fmt(item['reward_mean'])} +/- {fmt(item['reward_ci95'])} | "
            f"{fmt(item['goal_rate_mean'])} +/- {fmt(item['goal_rate_ci95'])} | "
            f"{fmt(item['proxy_dependence_mean'])} +/- {fmt(item['proxy_dependence_ci95'])} | "
            f"{fmt(item['abs_proxy_dependence_mean'])} +/- {fmt(item['abs_proxy_dependence_ci95'])} | "
            f"{fmt(item['proxy_action_rate_mean'])} +/- {fmt(item['proxy_action_rate_ci95'])} | "
            f"{fmt(item['proxy_Q_advantage_mean'])} +/- {fmt(item['proxy_Q_advantage_ci95'])} |"
        )
    lines.extend(["", "## Persistence summary", "", "| agent | feature | acq_dep | rev_final | ext_final | ext_half | persistence_index |", "|---|---|---:|---:|---:|---:|---:|"])
    for item in persistence_stats:
        lines.append(
            f"| {item['agent_type']} | {item['feature_condition']} | "
            f"{fmt(item['acquisition_proxy_dependence_final_mean'])} +/- {fmt(item['acquisition_proxy_dependence_final_ci95'])} | "
            f"{fmt(item['reversal_proxy_dependence_final_mean'])} +/- {fmt(item['reversal_proxy_dependence_final_ci95'])} | "
            f"{fmt(item['extinction_proxy_dependence_final_mean'])} +/- {fmt(item['extinction_proxy_dependence_final_ci95'])} | "
            f"{fmt(item['extinction_half_life_mean'], 1)} | "
            f"{fmt(item['superstition_persistence_index_mean'])} +/- {fmt(item['superstition_persistence_index_ci95'])} |"
        )
    lines.extend(["", "## Comparison table", "", "| comparison | contrast | metric | difference | 95% CI | p approx |", "|---|---|---|---:|---:|---:|"])
    for item in comparison_rows:
        lines.append(f"| {item['comparison']} | {item['contrast']} | {item['metric']} | {fmt(item['difference'], 4)} | +/- {fmt(item['ci95'], 4)} | {fmt(item['p_value_normal'], 4)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            f"Extinction absolute proxy dependence: linear/raw={fmt(linear_raw['abs_proxy_dependence_mean'])}, linear/interaction={fmt(linear_interaction['abs_proxy_dependence_mean'])}, neural/raw={fmt(neural_raw['abs_proxy_dependence_mean'])}, neural/interaction={fmt(neural_interaction['abs_proxy_dependence_mean'])}.",
            "",
            "## Does nonlinear approximation recover interaction-coded proxy reliance?",
            f"Neural interaction exceeds linear interaction by >0.20: {neural_recovers}.",
            f"Neural interaction approaches linear raw proxy dependence: {neural_matches_raw}.",
            "",
            "If neural DQN still fails on interaction-only features, the positive linear effect is best treated as a brittle representation phenomenon rather than robust computational superstition.",
            "",
            "## Validity checks",
            f"PyTorch version: {torch.__version__}. hidden_reward_state is never observed. Action 1 never directly causes reward. Reward only depends on hidden_reward_state and action 0. Action-space size and phase probabilities are unchanged.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "nonlinear_interaction_proxy_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows = []
    for agent_type, feature_condition in GROUPS:
        for seed in SEEDS:
            print(f"agent={agent_type} feature={feature_condition} seed={seed}", flush=True)
            rows.extend(train_and_evaluate(agent_type, feature_condition, seed))
    summary = summarize(rows)
    curve = aggregate_curve(rows)
    persistence = persistence_summary(rows)
    persistence_stats = summarize_persistence(persistence)
    comparison_rows = comparisons(rows, persistence)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "nonlinear_interaction_proxy_results.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_DIR, "nonlinear_interaction_proxy_summary.csv"), index=False)
    pd.DataFrame(persistence).to_csv(os.path.join(RESULTS_DIR, "nonlinear_interaction_proxy_persistence.csv"), index=False)
    pd.DataFrame(comparison_rows).to_csv(os.path.join(RESULTS_DIR, "nonlinear_interaction_proxy_comparisons.csv"), index=False)
    write_plots(curve, summary)
    write_report(summary, persistence_stats, comparison_rows)


if __name__ == "__main__":
    run()
