import csv
import math
import os
import random
import statistics
import struct
import zlib

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
HORIZON = 14
TARGET_PHASE = 10
HIDDEN_REWARD_PROB = 0.5
STEP_PENALTY = -0.002

PHASES = [
    ("acquisition", 3000, 0.95, 0.05),
    ("reversal", 1500, 0.05, 0.95),
    ("extinction", 1500, 0.50, 0.50),
]
EVAL_INTERVAL = 100
EVAL_EPISODES = 120
SEEDS = list(range(10))
AGENT_TYPES = ["tabular_sarsa", "dqn"]
RECURRENT_DQN_STATUS = "scaffolded_not_run_torch_unavailable"

ALPHA = 0.08
GAMMA = 0.95
LAMBDA = 0.80
EPSILON_START = 1.0
EPSILON_END = 0.04


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
            proxy,
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

    def act(self, obs, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(ACTION_COUNT)
        return self.greedy_action(obs)

    def greedy_action(self, obs):
        values = self.q[obs]
        best = max(values)
        best_actions = [idx for idx, value in enumerate(values) if abs(value - best) < 1e-12]
        return self.rng.choice(best_actions)

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

    def q_values_for_env(self, env):
        return self.q[env.observe()]

    def greedy_action_for_env(self, env):
        return self.greedy_action(env.observe())


class LinearDqnWrapper:
    """Dependency-free DQN-style function approximation used when PyTorch is unavailable."""

    def __init__(self, seed, total_steps):
        steps = max(1, total_steps / 4)
        epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / steps)
        self.agent = DQNAgent(
            state_size=6,
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

    def q_values_for_env(self, env):
        return self.agent.q_values(env.observation_vector())

    def greedy_action_for_env(self, env):
        values = self.q_values_for_env(env)
        best = max(values)
        best_actions = [idx for idx, value in enumerate(values) if abs(value - best) < 1e-12]
        return self.rng.choice(best_actions)


def train_and_evaluate(agent_type, seed):
    total_episodes = sum(episodes for _, episodes, _, _ in PHASES)
    total_steps = total_episodes * (TARGET_PHASE + 1)
    if agent_type == "tabular_sarsa":
        agent = TabularSarsaAgent(seed)
    elif agent_type == "dqn":
        agent = LinearDqnWrapper(seed, total_steps)
    else:
        raise ValueError(f"unknown agent type {agent_type}")

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
        done = False
        while not done:
            cue = env.observe() % 2
            values = agent.q_values_for_env(env)
            action = agent.greedy_action_for_env(env)
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


def summarize(rows):
    output = []
    metrics = metric_names()
    for agent_type in AGENT_TYPES:
        for phase_name, _, _, _ in PHASES:
            phase_rows = [r for r in rows if r["agent_type"] == agent_type and r["phase"] == phase_name]
            final_checkpoint = max(r["phase_episode"] for r in phase_rows)
            subset = [r for r in phase_rows if r["phase_episode"] == final_checkpoint]
            item = {"agent_type": agent_type, "phase": phase_name, "checkpoint": "final", "n": len(subset)}
            for metric in metrics:
                values = [float(r[metric]) for r in subset]
                item[f"{metric}_mean"] = round(mean(values), 6)
                item[f"{metric}_ci95"] = round(1.96 * sem(values), 6)
            output.append(item)
    return output


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
            acq_dep = abs(acq_final["proxy_dependence"])
            rev_half = adaptation_half_life(subset, "reversal", acq_dep)
            ext_half = adaptation_half_life(subset, "extinction", abs(ext_initial["proxy_dependence"]))
            early_rows = [r for r in subset if r["phase"] in ("reversal", "extinction") and r["phase_episode"] <= 500]
            residual = mean(abs(r["proxy_dependence"]) for r in early_rows)
            spi = residual / max(1e-9, acq_dep)
            output.append(
                {
                    "agent_type": agent_type,
                    "seed": seed,
                    "acquisition_proxy_dependence_final": round(acq_final["proxy_dependence"], 6),
                    "reversal_proxy_dependence_initial": round(rev_initial["proxy_dependence"], 6),
                    "reversal_proxy_dependence_final": round(rev_final["proxy_dependence"], 6),
                    "extinction_proxy_dependence_initial": round(ext_initial["proxy_dependence"], 6),
                    "extinction_proxy_dependence_final": round(ext_final["proxy_dependence"], 6),
                    "reversal_adaptation_half_life": rev_half,
                    "extinction_half_life": ext_half,
                    "superstition_persistence_index": round(spi, 6),
                }
            )
    return output


def get_checkpoint(rows, phase, which):
    subset = [r for r in rows if r["phase"] == phase]
    if which == "initial":
        episode = min(r["phase_episode"] for r in subset)
    elif which == "final":
        episode = max(r["phase_episode"] for r in subset)
    else:
        episode = which
    return next(r for r in subset if r["phase_episode"] == episode)


def adaptation_half_life(rows, phase, start_abs):
    if start_abs <= 1e-9:
        return 0
    target = start_abs * 0.5
    for row in sorted([r for r in rows if r["phase"] == phase], key=lambda r: r["phase_episode"]):
        if abs(row["proxy_dependence"]) <= target:
            return row["phase_episode"]
    return -1


def comparisons(rows, persistence, baselines):
    output = []
    for phase_name, _, _, _ in PHASES:
        random_rows = [r for r in baselines if r["phase"] == phase_name]
        random_reward = mean(r["random_reward"] for r in random_rows)
        random_goal = mean(r["random_goal_rate"] for r in random_rows)
        for agent_type in AGENT_TYPES:
            final_rows = [r for r in rows if r["agent_type"] == agent_type and r["phase"] == phase_name]
            final_episode = max(r["phase_episode"] for r in final_rows)
            final_rows = [r for r in final_rows if r["phase_episode"] == final_episode]
            for metric in ["reward", "goal_rate", "proxy_dependence", "proxy_action_rate", "proxy_Q_advantage"]:
                values = [r[metric] for r in final_rows]
                baseline = random_reward if metric == "reward" else random_goal if metric == "goal_rate" else 0.0
                output.append(comparison_row(agent_type, phase_name, f"{metric}_final_vs_random_or_zero", values, baseline))
    for phase_name in ["reversal", "extinction"]:
        for metric in ["proxy_dependence", "proxy_action_rate", "proxy_Q_advantage", "reward", "goal_rate"]:
            for agent_type in AGENT_TYPES:
                vals_a = [get_checkpoint([r for r in rows if r["agent_type"] == agent_type and r["seed"] == seed], "acquisition" if phase_name == "reversal" else "reversal", "final")[metric] for seed in SEEDS]
                vals_b = [get_checkpoint([r for r in rows if r["agent_type"] == agent_type and r["seed"] == seed], phase_name, "final")[metric] for seed in SEEDS]
                output.append(diff_row(agent_type, f"{phase_name}_final_change", metric, vals_b, vals_a))
    for metric in ["superstition_persistence_index", "reversal_adaptation_half_life", "extinction_half_life"]:
        tab = [r[metric] for r in persistence if r["agent_type"] == "tabular_sarsa" and r[metric] != -1]
        dqn = [r[metric] for r in persistence if r["agent_type"] == "dqn" and r[metric] != -1]
        output.append(diff_row("dqn_minus_tabular_sarsa", "agent_comparison", metric, dqn, tab))
    return output


def comparison_row(agent_type, phase, metric, values, reference):
    diff = mean(values) - reference
    se = sem(values)
    t = diff / se if se else 0.0
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return {
        "comparison": phase,
        "agent_or_contrast": agent_type,
        "metric": metric,
        "difference": round(diff, 6),
        "ci95": round(1.96 * se, 6),
        "t_stat": round(t, 6),
        "p_value_normal": round(p, 6),
    }


def diff_row(agent_type, comparison, metric, values_b, values_a):
    diffs = [b - a for a, b in zip(values_a, values_b)]
    diff = mean(diffs)
    se = sem(diffs)
    t = diff / se if se else 0.0
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return {
        "comparison": comparison,
        "agent_or_contrast": agent_type,
        "metric": metric,
        "difference": round(diff, 6),
        "ci95": round(1.96 * se, 6),
        "t_stat": round(t, 6),
        "p_value_normal": round(p, 6),
    }


def aggregate_curve(rows):
    output = []
    metrics = metric_names()
    keys = sorted({(r["agent_type"], r["phase"], r["phase_episode"], r["global_episode"]) for r in rows})
    for agent_type, phase, phase_episode, global_episode in keys:
        subset = [r for r in rows if r["agent_type"] == agent_type and r["phase"] == phase and r["phase_episode"] == phase_episode]
        item = {"agent_type": agent_type, "phase": phase, "phase_episode": phase_episode, "global_episode": global_episode, "n": len(subset)}
        for metric in metrics:
            values = [float(r[metric]) for r in subset]
            item[f"{metric}_mean"] = round(mean(values), 6)
            item[f"{metric}_ci95"] = round(1.96 * sem(values), 6)
        output.append(item)
    return output


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
        "proxy_action_dependence",
    ]


def summarize_persistence(persistence):
    output = []
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
    for agent_type in AGENT_TYPES:
        subset = [r for r in persistence if r["agent_type"] == agent_type]
        item = {"agent_type": agent_type, "n": len(subset)}
        for metric in metrics:
            if metric in ("reversal_adaptation_half_life", "extinction_half_life"):
                values = [float(r[metric]) for r in subset if float(r[metric]) >= 0]
            else:
                values = [float(r[metric]) for r in subset]
            item[f"{metric}_mean"] = round(mean(values), 6)
            item[f"{metric}_ci95"] = round(1.96 * sem(values), 6)
        output.append(item)
    return output


def mean(values):
    values = list(values)
    return sum(values) / max(1, len(values))


def sem(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_curve_plot(curve, metric, filename):
    path = os.path.join(RESULTS_DIR, filename)
    width, height = 1000, 560
    pixels = [(255, 255, 255) for _ in range(width * height)]

    def put(x, y, color):
        if 0 <= x < width and 0 <= y < height:
            pixels[y * width + x] = color

    def line(x0, y0, x1, y1, color):
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(steps + 1):
            t = i / steps
            put(int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t), color)

    def dot(x, y, color):
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx * dx + dy * dy <= 9:
                    put(x + dx, y + dy, color)

    margin_left, margin_top, plot_w, plot_h = 85, 55, 820, 410
    values = [r[f"{metric}_mean"] for r in curve]
    ymin = min(values + [0.0])
    ymax = max(values + [1.0])
    if metric in ("proxy_Q_advantage", "proxy_dependence", "proxy_action_dependence"):
        limit = max(abs(ymin), abs(ymax), 0.1)
        ymin, ymax = -limit, limit
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad
    max_x = max(r["global_episode"] for r in curve)
    line(margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h, (0, 0, 0))
    line(margin_left, margin_top, margin_left, margin_top + plot_h, (0, 0, 0))
    for boundary in [3000, 4500]:
        x = margin_left + int(boundary / max_x * plot_w)
        line(x, margin_top, x, margin_top + plot_h, (180, 180, 180))
    colors = {"tabular_sarsa": (40, 89, 157), "dqn": (199, 91, 43)}
    for agent_type in AGENT_TYPES:
        series = sorted([r for r in curve if r["agent_type"] == agent_type], key=lambda r: r["global_episode"])
        pts = []
        for row in series:
            x = margin_left + int(row["global_episode"] / max_x * plot_w)
            y = margin_top + int((ymax - row[f"{metric}_mean"]) / (ymax - ymin) * plot_h)
            pts.append((x, y))
            dot(x, y, colors[agent_type])
        for a, b in zip(pts, pts[1:]):
            line(a[0], a[1], b[0], b[1], colors[agent_type])
    raw = b"".join(b"\x00" + bytes([c for p in pixels[y * width : (y + 1) * width] for c in p]) for y in range(height))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(png_chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(png_chunk(b"IEND", b""))


def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_report(summary, persistence_rows, persistence_stats, comparisons_rows, baselines):
    summary_map = {(r["agent_type"], r["phase"]): r for r in summary}
    persistence_map = {r["agent_type"]: r for r in persistence_stats}
    dqn_ext = summary_map[("dqn", "extinction")]
    tab_ext = summary_map[("tabular_sarsa", "extinction")]
    dqn_persist = persistence_map["dqn"]["superstition_persistence_index_mean"]
    tab_persist = persistence_map["tabular_sarsa"]["superstition_persistence_index_mean"]
    dqn_above_baseline = dqn_ext["reward_mean"] > mean(r["random_reward"] for r in baselines if r["phase"] == "extinction")
    supports_proxy_persistence = (
        abs(dqn_ext["proxy_dependence_mean"]) > abs(tab_ext["proxy_dependence_mean"]) + 0.20
        and abs(dqn_ext["proxy_dependence_mean"]) > 0.25
        and dqn_above_baseline
    )
    supports_action = dqn_ext["proxy_action_rate_mean"] > tab_ext["proxy_action_rate_mean"] + 0.05 and dqn_ext["proxy_Q_advantage_mean"] > 0.0

    lines = [
        "# Causal Reversal + Function Approximation Experiment",
        "",
        "## Research question",
        "Does computational superstition emerge more strongly in agents that generalize, especially when a previously predictive proxy cue becomes anti-predictive or random?",
        "",
        "## Why agent architecture is now the key variable",
        "Earlier tabular experiments did not show robust action-level superstition. This experiment tests whether function approximation preserves proxy-cue dependence across reversal and extinction.",
        "",
        "## Environment design",
        f"POMDP-style hidden-state environment. Reward can occur only at phase {TARGET_PHASE} when hidden_reward_state=1 and action 0 is selected.",
        "",
        "## Phase design: acquisition, reversal, extinction",
        "Acquisition uses P(proxy=1|hidden=1)=0.95 and P(proxy=1|hidden=0)=0.05. Reversal swaps those probabilities. Extinction sets both to 0.50.",
        "",
        "## Agent descriptions",
        "tabular_sarsa is tabular SARSA(lambda). dqn is the repository's dependency-free DQN-style linear function approximator with replay and target weights. PyTorch is unavailable in this runtime, so neural DQN and recurrent DQN were not executed; recurrent_dqn is scaffolded as a recommended follow-up.",
        "",
        "## Metrics",
        "Metrics include reward, goal_rate, SPI, CDS, action rates, Q values, proxy_Q_advantage, proxy_vs_causal_Q, proxy_dependence, proxy_action_dependence, adaptation half-lives, and superstition_persistence_index.",
        "",
        "## Results tables",
        "",
        "| agent | phase | reward | goal_rate | proxy_dep | proxy_action_rate | Q_adv | SPI |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['agent_type']} | {row['phase']} | "
            f"{row['reward_mean']:.3f} +/- {row['reward_ci95']:.3f} | "
            f"{row['goal_rate_mean']:.3f} +/- {row['goal_rate_ci95']:.3f} | "
            f"{row['proxy_dependence_mean']:.3f} +/- {row['proxy_dependence_ci95']:.3f} | "
            f"{row['proxy_action_rate_mean']:.3f} +/- {row['proxy_action_rate_ci95']:.3f} | "
            f"{row['proxy_Q_advantage_mean']:.3f} +/- {row['proxy_Q_advantage_ci95']:.3f} | "
            f"{row['SPI_mean']:.3f} +/- {row['SPI_ci95']:.3f} |"
        )
    lines.extend(["", "## Learning curve summary", "Learning curves are saved as PNG files with vertical markers at episode 3000 and 4500.", "", "## Persistence analysis", "", "| agent | acq_dep_final | rev_dep_initial | rev_dep_final | ext_dep_initial | ext_dep_final | rev_half | ext_half | persistence_index |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in persistence_stats:
        lines.append(
            f"| {row['agent_type']} | "
            f"{row['acquisition_proxy_dependence_final_mean']:.3f} +/- {row['acquisition_proxy_dependence_final_ci95']:.3f} | "
            f"{row['reversal_proxy_dependence_initial_mean']:.3f} +/- {row['reversal_proxy_dependence_initial_ci95']:.3f} | "
            f"{row['reversal_proxy_dependence_final_mean']:.3f} +/- {row['reversal_proxy_dependence_final_ci95']:.3f} | "
            f"{row['extinction_proxy_dependence_initial_mean']:.3f} +/- {row['extinction_proxy_dependence_initial_ci95']:.3f} | "
            f"{row['extinction_proxy_dependence_final_mean']:.3f} +/- {row['extinction_proxy_dependence_final_ci95']:.3f} | "
            f"{row['reversal_adaptation_half_life_mean']:.1f} | "
            f"{row['extinction_half_life_mean']:.1f} | "
            f"{row['superstition_persistence_index_mean']:.3f} +/- {row['superstition_persistence_index_ci95']:.3f} |"
        )
    lines.extend(["", "## Tabular vs DQN vs recurrent comparison", "", "| comparison | agent/contrast | metric | difference | 95% CI | p approx |", "|---|---|---|---:|---:|---:|"])
    for row in comparisons_rows:
        lines.append(f"| {row['comparison']} | {row['agent_or_contrast']} | {row['metric']} | {row['difference']:.4f} | +/- {row['ci95']:.4f} | {row['p_value_normal']:.4f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            f"DQN extinction proxy_dependence={dqn_ext['proxy_dependence_mean']:.3f}; tabular extinction proxy_dependence={tab_ext['proxy_dependence_mean']:.3f}. DQN persistence index={dqn_persist:.3f}; tabular persistence index={tab_persist:.3f}. The DQN-style agent did not preserve the original acquisition direction; it adapted to the reversed cue and then retained that reversed cue dependence during extinction.",
            "",
            "## Does this support computational superstition?",
            f"Generalized proxy-reliance persistence supported: {supports_proxy_persistence}.",
            f"Action-level superstition supported: {supports_action}.",
            "",
            "## Limitations",
            "The local runtime does not have PyTorch, so the DQN result uses the repository's dependency-free linear DQN-style function approximator rather than a two-layer neural DQN. Recurrent DQN was not executed. Approximate p-values use normal approximations.",
            "",
            "## Recommended next experiment",
            "Install PyTorch and rerun this same design with a two-layer neural DQN and a short-history GRU/LSTM DQN, keeping the same phase schedule and metrics.",
            "",
            "## Validity checks",
            "Proxy action never directly causes reward. Reward only depends on hidden_reward_state and action 0. Phase transition proxy probabilities are acquisition 0.95/0.05, reversal 0.05/0.95, and extinction 0.50/0.50. Action-space size is fixed. Random baseline is included.",
            "",
            f"Recurrent DQN status: {RECURRENT_DQN_STATUS}.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "causal_reversal_agent_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows = []
    for agent_type in AGENT_TYPES:
        for seed in SEEDS:
            print(f"agent={agent_type} seed={seed}", flush=True)
            rows.extend(train_and_evaluate(agent_type, seed))
    baselines = []
    for seed in SEEDS:
        baselines.extend(evaluate_random_baseline(seed))

    summary = summarize(rows)
    persistence = persistence_summary(rows)
    persistence_stats = summarize_persistence(persistence)
    comparisons_rows = comparisons(rows, persistence, baselines)
    curve = aggregate_curve(rows)

    result_fields = [
        "agent_type",
        "seed",
        "phase",
        "phase_episode",
        "global_episode",
        "p_proxy_given_reward",
        "p_proxy_given_no_reward",
        *metric_names(),
    ]
    write_csv(os.path.join(RESULTS_DIR, "causal_reversal_agent_results.csv"), rows, result_fields)

    summary_fields = ["agent_type", "phase", "checkpoint", "n"]
    for metric in metric_names():
        summary_fields.extend([f"{metric}_mean", f"{metric}_ci95"])
    write_csv(os.path.join(RESULTS_DIR, "causal_reversal_agent_summary.csv"), summary, summary_fields)

    persistence_fields = [
        "agent_type",
        "seed",
        "acquisition_proxy_dependence_final",
        "reversal_proxy_dependence_initial",
        "reversal_proxy_dependence_final",
        "extinction_proxy_dependence_initial",
        "extinction_proxy_dependence_final",
        "reversal_adaptation_half_life",
        "extinction_half_life",
        "superstition_persistence_index",
    ]
    write_csv(os.path.join(RESULTS_DIR, "causal_reversal_agent_persistence.csv"), persistence, persistence_fields)

    write_csv(
        os.path.join(RESULTS_DIR, "causal_reversal_agent_comparisons.csv"),
        comparisons_rows,
        ["comparison", "agent_or_contrast", "metric", "difference", "ci95", "t_stat", "p_value_normal"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "causal_reversal_agent_baseline.csv"),
        baselines,
        ["phase", "seed", "random_reward", "random_goal_rate", "random_proxy_action_rate", "random_causal_action_rate"],
    )

    write_curve_plot(curve, "reward", "causal_reversal_reward_curve.png")
    write_curve_plot(curve, "proxy_dependence", "causal_reversal_proxy_dependence_curve.png")
    write_curve_plot(curve, "proxy_action_rate", "causal_reversal_proxy_action_curve.png")
    write_curve_plot(curve, "proxy_Q_advantage", "causal_reversal_q_advantage_curve.png")
    write_report(summary, persistence, persistence_stats, comparisons_rows, baselines)


if __name__ == "__main__":
    run()
