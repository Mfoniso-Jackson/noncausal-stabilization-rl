import math
import os
import random
import statistics

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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
PHASE_BOUNDARIES = [3000, 4500]
EVAL_INTERVAL = 100
EVAL_EPISODES = 300
SEEDS = list(range(10))
FEATURE_CONDITIONS = [
    "phase_only",
    "full_raw_proxy",
    "orthogonal_rotation_full",
    "centered_proxy_residual",
    "proxy_interaction_only",
]

GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.04
MIN_PERSISTENCE_DENOMINATOR = 0.05

HADAMARD_6 = [
    [1, 1, 1, 1, 1, 1],
    [1, -1, 1, -1, 1, -1],
    [1, 1, -1, -1, -1, -1],
    [1, -1, -1, 1, -1, 1],
    [1, 1, 1, 1, -1, -1],
    [1, -1, 1, -1, -1, 1],
]


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

    def base_features(self):
        phase_norm = self.phase / TARGET_PHASE
        proxy = float(self.proxy_by_phase[self.phase])
        target_flag = 1.0 if self.phase == TARGET_PHASE else 0.0
        pre_target_flag = 1.0 if self.phase < TARGET_PHASE else 0.0
        sin_phase = math.sin(self.phase / max(1, TARGET_PHASE) * math.pi)
        cos_phase = math.cos(self.phase / max(1, TARGET_PHASE) * math.pi)
        return [phase_norm, proxy, target_flag, pre_target_flag, sin_phase, cos_phase]

    def features(self, condition):
        base = self.base_features()
        phase_norm, proxy, target_flag, pre_target_flag, sin_phase, cos_phase = base
        proxy_signed = 2.0 * proxy - 1.0
        if condition == "phase_only":
            return [phase_norm, target_flag, pre_target_flag, sin_phase, cos_phase]
        if condition == "full_raw_proxy":
            return base
        if condition == "orthogonal_rotation_full":
            # Invertible linear mixing of the exact same full feature vector.
            return [sum(w * x for w, x in zip(row, base)) / math.sqrt(6.0) for row in HADAMARD_6]
        if condition == "centered_proxy_residual":
            # Proxy is present as a zero-centered coordinate, orthogonal to the constant offset.
            return [phase_norm, proxy - 0.5, target_flag, pre_target_flag, sin_phase, cos_phase]
        if condition == "proxy_interaction_only":
            # Proxy information is present, but only through phase interactions; no raw proxy coordinate.
            return [
                phase_norm,
                target_flag,
                pre_target_flag,
                sin_phase,
                cos_phase,
                proxy_signed * phase_norm,
                proxy_signed * sin_phase,
                proxy_signed * cos_phase,
            ]
        raise ValueError(condition)

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


class FeatureLinearDqn:
    def __init__(self, condition, seed, total_steps):
        self.condition = condition
        probe = HiddenProxyReversalEnv(0.5, 0.5, seed=seed)
        state_size = len(probe.features(condition))
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

    def train_episode(self, env):
        env.reset()
        state = env.features(self.condition)
        done = False
        while not done:
            action = self.agent.act(state, training=True)
            _, reward, done, _ = env.step(action)
            next_state = env.features(self.condition)
            self.agent.remember(state, action, reward, next_state, done)
            self.agent.train_step()
            state = next_state

    def q_values_for_env(self, env):
        return self.agent.q_values(env.features(self.condition))


def train_and_evaluate(condition, seed):
    total_episodes = sum(episodes for _, episodes, _, _ in PHASES)
    total_steps = total_episodes * (TARGET_PHASE + 1)
    agent = FeatureLinearDqn(condition, seed, total_steps)
    env = HiddenProxyReversalEnv(PHASES[0][2], PHASES[0][3], seed=seed)
    rows = []
    global_episode = 0
    for phase_name, phase_episodes, p1, p0 in PHASES:
        env.set_proxy_probs(p1, p0)
        for phase_episode in range(1, phase_episodes + 1):
            agent.train_episode(env)
            global_episode += 1
            if phase_episode == 1 or phase_episode % EVAL_INTERVAL == 0 or phase_episode == phase_episodes:
                metrics = evaluate(agent, p1, p0, seed + global_episode * 17)
                rows.append(
                    {
                        "feature_condition": condition,
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
        rows.append({"phase": phase_name, "seed": seed, "random_reward": round(totals["reward_sum"] / EVAL_EPISODES, 6), "random_goal_rate": round(totals["goal_sum"] / EVAL_EPISODES, 6)})
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


def random_argmax(values, rng):
    best = max(values)
    best_actions = [idx for idx, value in enumerate(values) if abs(value - best) < 1e-12]
    return rng.choice(best_actions)


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
    for condition in FEATURE_CONDITIONS:
        for phase, _, _, _ in PHASES:
            subset = df[(df.feature_condition == condition) & (df.phase == phase)]
            final_episode = subset.phase_episode.max()
            group = subset[subset.phase_episode == final_episode]
            item = {"feature_condition": condition, "phase": phase, "checkpoint": "final", "n": len(group)}
            for metric in metric_names():
                values = list(group[metric].astype(float))
                item[f"{metric}_mean"] = round(mean(values), 6)
                item[f"{metric}_ci95"] = round(ci95(values), 6)
            summary_rows.append(item)
    return summary_rows


def aggregate_curve(rows):
    df = pd.DataFrame(rows)
    curve_rows = []
    for keys, group in df.groupby(["feature_condition", "phase", "phase_episode", "global_episode"], sort=False):
        condition, phase, phase_episode, global_episode = keys
        item = {"feature_condition": condition, "phase": phase, "phase_episode": phase_episode, "global_episode": global_episode, "n": len(group)}
        for metric in metric_names():
            values = list(group[metric].astype(float))
            item[f"{metric}_mean"] = round(mean(values), 6)
            item[f"{metric}_ci95"] = round(ci95(values), 6)
        curve_rows.append(item)
    return curve_rows


def persistence_summary(rows):
    output = []
    for condition in FEATURE_CONDITIONS:
        for seed in SEEDS:
            subset = [r for r in rows if r["feature_condition"] == condition and r["seed"] == seed]
            acq_final = get_checkpoint(subset, "acquisition", "final")
            rev_final = get_checkpoint(subset, "reversal", "final")
            ext_initial = get_checkpoint(subset, "extinction", "initial")
            ext_final = get_checkpoint(subset, "extinction", "final")
            acq_abs = abs(acq_final["proxy_dependence"])
            early_ext = [r for r in subset if r["phase"] == "extinction" and r["phase_episode"] <= 500]
            residual = mean(abs(r["proxy_dependence"]) for r in early_ext)
            persistence_index = residual / acq_abs if acq_abs >= MIN_PERSISTENCE_DENOMINATOR else float("nan")
            output.append(
                {
                    "feature_condition": condition,
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


def summarize_persistence(persistence):
    rows = []
    for condition in FEATURE_CONDITIONS:
        subset = [r for r in persistence if r["feature_condition"] == condition]
        item = {"feature_condition": condition, "n": len(subset)}
        for metric in ["acquisition_proxy_dependence_final", "reversal_proxy_dependence_final", "extinction_proxy_dependence_initial", "extinction_proxy_dependence_final", "extinction_half_life", "superstition_persistence_index"]:
            if metric.endswith("half_life"):
                values = [float(r[metric]) for r in subset if math.isfinite(float(r[metric])) and float(r[metric]) >= 0]
            else:
                values = [float(r[metric]) for r in subset if math.isfinite(float(r[metric]))]
            item[f"{metric}_mean"] = round(mean(values), 6) if values else float("nan")
            item[f"{metric}_ci95"] = round(ci95(values), 6) if values else float("nan")
        rows.append(item)
    return rows


def comparisons(rows, persistence, baselines):
    output = []
    reference = "full_raw_proxy"
    for condition in FEATURE_CONDITIONS:
        if condition == reference:
            continue
        for phase, _, _, _ in PHASES:
            for metric in ["reward", "goal_rate", "abs_proxy_dependence", "proxy_dependence", "proxy_action_rate", "proxy_Q_advantage"]:
                a_values, b_values = [], []
                for seed in SEEDS:
                    a_rows = [r for r in rows if r["feature_condition"] == condition and r["seed"] == seed and r["phase"] == phase]
                    b_rows = [r for r in rows if r["feature_condition"] == reference and r["seed"] == seed and r["phase"] == phase]
                    a_values.append(max(a_rows, key=lambda r: r["phase_episode"])[metric])
                    b_values.append(max(b_rows, key=lambda r: r["phase_episode"])[metric])
                output.append(diff_row(f"{condition}_vs_{reference}", phase, metric, a_values, b_values))
        for metric in ["extinction_half_life", "superstition_persistence_index"]:
            a_values = [r[metric] for r in persistence if r["feature_condition"] == condition and math.isfinite(float(r[metric])) and r[metric] >= 0]
            b_values = [r[metric] for r in persistence if r["feature_condition"] == reference and math.isfinite(float(r[metric])) and r[metric] >= 0]
            output.append(diff_row(f"{condition}_vs_{reference}", "persistence", metric, a_values, b_values, paired=False))
    for condition in FEATURE_CONDITIONS:
        for phase, _, _, _ in PHASES:
            random_reward = mean(r["random_reward"] for r in baselines if r["phase"] == phase)
            random_goal = mean(r["random_goal_rate"] for r in baselines if r["phase"] == phase)
            for metric, baseline in [("reward", random_reward), ("goal_rate", random_goal)]:
                values = []
                for seed in SEEDS:
                    subset = [r for r in rows if r["feature_condition"] == condition and r["seed"] == seed and r["phase"] == phase]
                    values.append(max(subset, key=lambda r: r["phase_episode"])[metric])
                output.append(one_sample_row(condition, f"{phase}_vs_random", metric, values, baseline))
    return output


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


def one_sample_row(condition, comparison, metric, values, baseline):
    diffs = [float(value) - baseline for value in values]
    diff = mean(diffs)
    se = sem(diffs)
    t = diff / se if se else 0.0
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return {"comparison": comparison, "contrast": condition, "metric": metric, "difference": round(diff, 6), "ci95": round(1.96 * se, 6), "t_stat": round(t, 6), "p_value_normal": round(p, 6)}


def fmt(value, digits=3):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_plots(curve_rows, summary):
    curve = pd.DataFrame(curve_rows)
    for metric, ylabel, filename in [
        ("reward_mean", "Reward", "orthogonalized_proxy_reward_curve.png"),
        ("abs_proxy_dependence_mean", "Absolute Proxy Dependence", "orthogonalized_proxy_abs_dependence_curve.png"),
        ("proxy_dependence_mean", "Signed Proxy Dependence", "orthogonalized_proxy_signed_dependence_curve.png"),
        ("proxy_action_rate_mean", "Proxy Action Rate", "orthogonalized_proxy_action_curve.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 6))
        for condition in FEATURE_CONDITIONS:
            subset = curve[curve.feature_condition == condition].sort_values("global_episode")
            ax.plot(subset.global_episode, subset[metric], label=condition, linewidth=1.7)
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
    ext = summary_df[summary_df.phase == "extinction"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(ext.feature_condition, ext.abs_proxy_dependence_mean, yerr=ext.abs_proxy_dependence_ci95, color="#4C78A8")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Extinction absolute proxy dependence")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "orthogonalized_proxy_feature_plot.png"), dpi=160)
    plt.close(fig)


def write_report(summary, persistence_stats, comparisons_rows):
    summary_df = pd.DataFrame(summary)
    ext = summary_df[summary_df.phase == "extinction"].copy()
    raw = ext[ext.feature_condition == "full_raw_proxy"].iloc[0]
    rotated = ext[ext.feature_condition == "orthogonal_rotation_full"].iloc[0]
    interaction = ext[ext.feature_condition == "proxy_interaction_only"].iloc[0]
    phase_only = ext[ext.feature_condition == "phase_only"].iloc[0]
    coordinate_artifact = abs(rotated["abs_proxy_dependence_mean"] - raw["abs_proxy_dependence_mean"]) > 0.20
    linear_access_artifact = interaction["abs_proxy_dependence_mean"] < raw["abs_proxy_dependence_mean"] - 0.20
    lines = [
        "# Orthogonalized Proxy Feature Control",
        "",
        "## Research question",
        "Does the linear DQN proxy-persistence effect require an easy raw proxy coordinate, or does it survive feature rotations and interaction-coded proxy information?",
        "",
        "## Design",
        "Same hidden-state reversal environment, same linear DQN, same phase schedule, same seeds, same metrics. Only the feature representation changes.",
        "",
        "## Feature conditions",
        "- phase_only: no proxy information.",
        "- full_raw_proxy: original full feature vector with raw proxy coordinate.",
        "- orthogonal_rotation_full: invertible linear rotation of the same full feature vector.",
        "- centered_proxy_residual: proxy centered at zero while keeping phase features.",
        "- proxy_interaction_only: no raw proxy coordinate; proxy appears only through phase interaction terms.",
        "",
        "## Results table",
        "",
        "| feature_condition | phase | reward | goal_rate | proxy_dep | abs_proxy_dep | proxy_action_rate | Q_adv |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['feature_condition']} | {row['phase']} | "
            f"{fmt(row['reward_mean'])} +/- {fmt(row['reward_ci95'])} | "
            f"{fmt(row['goal_rate_mean'])} +/- {fmt(row['goal_rate_ci95'])} | "
            f"{fmt(row['proxy_dependence_mean'])} +/- {fmt(row['proxy_dependence_ci95'])} | "
            f"{fmt(row['abs_proxy_dependence_mean'])} +/- {fmt(row['abs_proxy_dependence_ci95'])} | "
            f"{fmt(row['proxy_action_rate_mean'])} +/- {fmt(row['proxy_action_rate_ci95'])} | "
            f"{fmt(row['proxy_Q_advantage_mean'])} +/- {fmt(row['proxy_Q_advantage_ci95'])} |"
        )
    lines.extend(["", "## Persistence summary", "", "| feature_condition | acq_dep | rev_final | ext_final | ext_half | persistence_index |", "|---|---:|---:|---:|---:|---:|"])
    for row in persistence_stats:
        lines.append(
            f"| {row['feature_condition']} | "
            f"{fmt(row['acquisition_proxy_dependence_final_mean'])} +/- {fmt(row['acquisition_proxy_dependence_final_ci95'])} | "
            f"{fmt(row['reversal_proxy_dependence_final_mean'])} +/- {fmt(row['reversal_proxy_dependence_final_ci95'])} | "
            f"{fmt(row['extinction_proxy_dependence_final_mean'])} +/- {fmt(row['extinction_proxy_dependence_final_ci95'])} | "
            f"{fmt(row['extinction_half_life_mean'], 1)} | "
            f"{fmt(row['superstition_persistence_index_mean'])} +/- {fmt(row['superstition_persistence_index_ci95'])} |"
        )
    lines.extend(["", "## Comparison summary", "", "| comparison | contrast | metric | difference | 95% CI | p approx |", "|---|---|---|---:|---:|---:|"])
    for row in comparisons_rows:
        lines.append(f"| {row['comparison']} | {row['contrast']} | {row['metric']} | {fmt(row['difference'], 4)} | +/- {fmt(row['ci95'], 4)} | {fmt(row['p_value_normal'], 4)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            f"Extinction absolute proxy dependence: phase_only={fmt(phase_only['abs_proxy_dependence_mean'])}, raw={fmt(raw['abs_proxy_dependence_mean'])}, rotated={fmt(rotated['abs_proxy_dependence_mean'])}, interaction_only={fmt(interaction['abs_proxy_dependence_mean'])}.",
            "",
            "## Does this support a coordinate artifact?",
            f"Raw coordinate-specific artifact supported: {coordinate_artifact}.",
            f"Linear accessibility artifact supported: {linear_access_artifact}.",
            "",
            "If the orthogonal rotation preserves the effect, the result is not tied to one named input coordinate. If interaction-only coding weakens the effect, the result depends on how linearly accessible the proxy rule is.",
            "",
            "## Validity checks",
            "hidden_reward_state is never observed. Action 1 never directly causes reward. Reward only depends on hidden_reward_state and action 0. Phase probabilities and action-space size are fixed across all conditions. Random baseline is included.",
            "",
            "## Recommended next experiment",
            "Use an explicitly nonlinear function approximator on the interaction-only representation to test whether neural models can recover proxy reliance when linear DQN cannot.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "orthogonalized_proxy_feature_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows = []
    for condition in FEATURE_CONDITIONS:
        for seed in SEEDS:
            print(f"feature={condition} seed={seed}", flush=True)
            rows.extend(train_and_evaluate(condition, seed))
    baselines = []
    for seed in SEEDS:
        baselines.extend(evaluate_random_baseline(seed))
    summary = summarize(rows)
    curve = aggregate_curve(rows)
    persistence = persistence_summary(rows)
    persistence_stats = summarize_persistence(persistence)
    comparison_rows = comparisons(rows, persistence, baselines)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "orthogonalized_proxy_feature_results.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_DIR, "orthogonalized_proxy_feature_summary.csv"), index=False)
    pd.DataFrame(persistence).to_csv(os.path.join(RESULTS_DIR, "orthogonalized_proxy_feature_persistence.csv"), index=False)
    pd.DataFrame(comparison_rows).to_csv(os.path.join(RESULTS_DIR, "orthogonalized_proxy_feature_comparisons.csv"), index=False)
    pd.DataFrame(baselines).to_csv(os.path.join(RESULTS_DIR, "orthogonalized_proxy_feature_baseline.csv"), index=False)
    write_plots(curve, summary)
    write_report(summary, persistence_stats, comparison_rows)


if __name__ == "__main__":
    run()
