import csv
import math
import os
import random
import statistics
import struct
import zlib


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

CONDITIONS = [
    ("no_spurious_correlation", 0.10),
    ("weak_spurious_correlation", 0.60),
    ("medium_spurious_correlation", 0.75),
    ("strong_spurious_correlation", 0.90),
]
SEEDS = list(range(30))
EPISODES = 2500
EVAL_EPISODES = 300
HORIZON = 16
TARGET_PHASE = 10
ACTION_CAUSAL = 0
ACTION_SPURIOUS = 1
NEUTRAL_ACTIONS = [2, 3]
ACTION_COUNT = 4
ALPHA = 0.08
GAMMA = 0.95
LAMBDA = 0.80
EPSILON_START = 1.0
EPSILON_END = 0.04
STEP_PENALTY = -0.002
BASE_SPURIOUS_PROB = 0.10
EVAL_SPURIOUS_PROB = 0.25


class SpuriousCorrelationEnv:
    def __init__(self, correlation_strength, training=True, seed=0):
        self.correlation_strength = correlation_strength
        self.training = training
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.phase = 0
        self.done = False
        self.goal_completed = False
        self.spurious_phases = self._sample_spurious_phases()
        return self.observe()

    def _sample_spurious_phases(self):
        phases = set()
        for phase in range(TARGET_PHASE):
            if self.training:
                prob = self.correlation_strength if phase == TARGET_PHASE - 1 else BASE_SPURIOUS_PROB
            else:
                prob = EVAL_SPURIOUS_PROB
            if self.rng.random() < prob:
                phases.add(phase)
        return phases

    def observe(self):
        cue = 1 if self.phase in self.spurious_phases and self.phase < TARGET_PHASE else 0
        return self.phase * 2 + cue

    def available_actions(self):
        actions = [ACTION_CAUSAL] + NEUTRAL_ACTIONS
        if self.phase in self.spurious_phases and self.phase < TARGET_PHASE:
            actions.append(ACTION_SPURIOUS)
        return sorted(actions)

    def step(self, action):
        if self.done:
            raise RuntimeError("step called after episode termination")
        reward = STEP_PENALTY
        if self.phase == TARGET_PHASE and action == ACTION_CAUSAL:
            reward += 1.0
            self.goal_completed = True
            self.done = True
        else:
            self.phase += 1
            self.done = self.phase >= HORIZON
        return self.observe(), reward, self.done, {"goal_completed": self.goal_completed}


class SarsaLambdaAgent:
    def __init__(self, state_count, seed):
        self.rng = random.Random(seed)
        self.q = [[0.0 for _ in range(ACTION_COUNT)] for _ in range(state_count)]

    def act(self, state, available, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.choice(available)
        return self.greedy_action(state, available)

    def greedy_action(self, state, available):
        best = max(self.q[state][action] for action in available)
        best_actions = [action for action in available if abs(self.q[state][action] - best) < 1e-12]
        return self.rng.choice(best_actions)

    def train_episode(self, env, epsilon):
        traces = [[0.0 for _ in range(ACTION_COUNT)] for _ in range(len(self.q))]
        state = env.reset()
        action = self.act(state, env.available_actions(), epsilon)
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            if done:
                target = reward
                next_action = None
            else:
                next_action = self.act(next_state, env.available_actions(), epsilon)
                target = reward + GAMMA * self.q[next_state][next_action]
            delta = target - self.q[state][action]
            traces[state][action] += 1.0
            for s in range(len(self.q)):
                for a in range(ACTION_COUNT):
                    self.q[s][a] += ALPHA * delta * traces[s][a]
                    traces[s][a] *= GAMMA * LAMBDA
            state = next_state
            action = next_action if next_action is not None else ACTION_CAUSAL


def run_seed(condition, strength, seed):
    train_env = SpuriousCorrelationEnv(strength, training=True, seed=seed)
    state_count = HORIZON * 2 + 2
    agent = SarsaLambdaAgent(state_count, seed)
    epsilon = EPSILON_START
    epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / EPISODES)
    for _ in range(EPISODES):
        agent.train_episode(train_env, epsilon)
        epsilon = max(EPSILON_END, epsilon * epsilon_decay)
    metrics = evaluate(agent, strength, seed)
    return {"condition": condition, "correlation_strength": strength, "seed": seed, **metrics}


def evaluate(agent, strength, seed):
    env = SpuriousCorrelationEnv(strength, training=False, seed=seed + 90000)
    total_reward = 0.0
    goals = 0
    decisions = 0
    spurious_choices = 0
    neutral_choices = 0
    causal_choices = 0
    cds_values = []
    q_spurious_values = []
    q_neutral_values = []
    q_causal_values = []
    for _ in range(EVAL_EPISODES):
        state = env.reset()
        done = False
        while not done:
            available = env.available_actions()
            if ACTION_SPURIOUS in available and env.phase < TARGET_PHASE:
                masses = softmax_mass(agent.q[state], available)
                decisions += 1
                cds_values.append(masses[ACTION_SPURIOUS] + sum(masses[a] for a in NEUTRAL_ACTIONS))
                q_spurious_values.append(agent.q[state][ACTION_SPURIOUS])
                q_neutral_values.append(mean(agent.q[state][a] for a in NEUTRAL_ACTIONS))
                q_causal_values.append(agent.q[state][ACTION_CAUSAL])
            action = agent.greedy_action(state, available)
            if ACTION_SPURIOUS in available and env.phase < TARGET_PHASE:
                if action == ACTION_SPURIOUS:
                    spurious_choices += 1
                elif action in NEUTRAL_ACTIONS:
                    neutral_choices += 1
                elif action == ACTION_CAUSAL:
                    causal_choices += 1
            state, reward, done, _ = env.step(action)
            total_reward += reward
        goals += int(env.goal_completed)
    q_spurious = mean(q_spurious_values) if q_spurious_values else 0.0
    q_neutral = mean(q_neutral_values) if q_neutral_values else 0.0
    q_causal = mean(q_causal_values) if q_causal_values else 0.0
    spurious_rate = spurious_choices / decisions if decisions else 0.0
    neutral_rate = neutral_choices / decisions if decisions else 0.0
    causal_rate = causal_choices / decisions if decisions else 0.0
    return {
        "reward": round(total_reward / EVAL_EPISODES, 6),
        "goal_rate": round(goals / EVAL_EPISODES, 6),
        "CDS": round(mean(cds_values) if cds_values else 0.0, 6),
        "SPI": round((spurious_choices + neutral_choices) / decisions if decisions else 0.0, 6),
        "spurious_action_rate": round(spurious_rate, 6),
        "neutral_action_rate": round(neutral_rate, 6),
        "causal_action_rate": round(causal_rate, 6),
        "Q_spurious": round(q_spurious, 6),
        "Q_neutral_mean": round(q_neutral, 6),
        "Q_causal": round(q_causal, 6),
        "spurious_Q_advantage": round(q_spurious - q_neutral, 6),
        "spurious_policy_advantage": round(spurious_rate - neutral_rate, 6),
    }


def softmax_mass(values, available):
    max_q = max(values[action] for action in available)
    exp_values = {action: math.exp(values[action] - max_q) for action in available}
    denom = sum(exp_values.values())
    return {action: exp_values[action] / denom for action in available}


def summarize(rows):
    metrics = [
        "reward",
        "goal_rate",
        "CDS",
        "SPI",
        "spurious_action_rate",
        "neutral_action_rate",
        "causal_action_rate",
        "Q_spurious",
        "Q_neutral_mean",
        "Q_causal",
        "spurious_Q_advantage",
        "spurious_policy_advantage",
    ]
    summary = []
    for condition, strength in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        item = {"condition": condition, "correlation_strength": strength, "n": len(subset)}
        for metric in metrics:
            values = [float(row[metric]) for row in subset]
            item[f"{metric}_mean"] = round(mean(values), 6)
            item[f"{metric}_ci95"] = round(1.96 * sem(values), 6)
        summary.append(item)
    return summary


def trend_analysis(rows):
    trends = []
    for metric in ["spurious_action_rate", "spurious_Q_advantage", "SPI", "reward", "goal_rate"]:
        x = [float(row["correlation_strength"]) for row in rows]
        y = [float(row[metric]) for row in rows]
        fit = linear_regression(x, y)
        trends.append(
            {
                "metric": metric,
                "slope": round(fit["slope"], 6),
                "slope_ci95": round(1.96 * fit["slope_se"], 6),
                "standardized_effect": round(fit["standardized_effect"], 6),
                "t_stat": round(fit["t_stat"], 6),
                "p_value_normal": round(fit["p_value"], 6),
            }
        )
    return trends


def pairwise(rows):
    output = []
    none = [row for row in rows if row["condition"] == "no_spurious_correlation"]
    strong = [row for row in rows if row["condition"] == "strong_spurious_correlation"]
    for metric in ["spurious_action_rate", "spurious_Q_advantage", "SPI", "reward", "goal_rate"]:
        none_values = [float(row[metric]) for row in none]
        strong_values = [float(row[metric]) for row in strong]
        diff = mean(strong_values) - mean(none_values)
        se = math.sqrt(sem(none_values) ** 2 + sem(strong_values) ** 2)
        t_stat = diff / se if se > 0 else 0.0
        p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
        output.append(
            {
                "metric": metric,
                "strong_minus_none": round(diff, 6),
                "ci95": round(1.96 * se, 6),
                "t_stat": round(t_stat, 6),
                "p_value_normal": round(p_value, 6),
            }
        )
    return output


def linear_regression(x, y):
    x_mean = mean(x)
    y_mean = mean(y)
    sxx = sum((value - x_mean) ** 2 for value in x)
    sxy = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x, y))
    slope = sxy / sxx if sxx else 0.0
    intercept = y_mean - slope * x_mean
    residuals = [yv - (intercept + slope * xv) for xv, yv in zip(x, y)]
    df = max(1, len(y) - 2)
    residual_var = sum(r * r for r in residuals) / df
    slope_se = math.sqrt(residual_var / sxx) if sxx else 0.0
    t_stat = slope / slope_se if slope_se else 0.0
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
    x_sd = statistics.stdev(x) if len(x) > 1 else 0.0
    y_sd = statistics.stdev(y) if len(y) > 1 else 0.0
    standardized = slope * x_sd / y_sd if y_sd else 0.0
    return {"slope": slope, "slope_se": slope_se, "t_stat": t_stat, "p_value": p_value, "standardized_effect": standardized}


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


def write_plot(summary):
    path = os.path.join(RESULTS_DIR, "spurious_correlation_plot.png")
    width, height = 980, 620
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
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                if dx * dx + dy * dy <= 16:
                    put(x + dx, y + dy, color)

    panels = [
        ("spurious_action_rate_mean", 70, 70),
        ("spurious_Q_advantage_mean", 540, 70),
        ("SPI_mean", 70, 360),
        ("goal_rate_mean", 540, 360),
    ]
    xs = [row["correlation_strength"] for row in summary]
    for key, ox, oy in panels:
        panel_w, panel_h = 340, 190
        values = [row[key] for row in summary]
        ymin = min(values + [0.0])
        ymax = max(values + [1.0])
        if ymax <= ymin:
            ymax = ymin + 1.0
        pad = (ymax - ymin) * 0.1
        ymin -= pad
        ymax += pad
        line(ox, oy + panel_h, ox + panel_w, oy + panel_h, (0, 0, 0))
        line(ox, oy, ox, oy + panel_h, (0, 0, 0))
        points = []
        for x, value in zip(xs, values):
            px = ox + int((x - min(xs)) / (max(xs) - min(xs)) * panel_w)
            py = oy + int((ymax - value) / (ymax - ymin) * panel_h)
            points.append((px, py))
            dot(px, py, (36, 92, 156))
        for a, b in zip(points, points[1:]):
            line(a[0], a[1], b[0], b[1], (36, 92, 156))
    raw = b"".join(b"\x00" + bytes([channel for pixel in pixels[y * width : (y + 1) * width] for channel in pixel]) for y in range(height))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(png_chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(png_chunk(b"IEND", b""))


def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_report(summary, trends, pairs):
    trend_map = {row["metric"]: row for row in trends}
    pair_map = {row["metric"]: row for row in pairs}
    strong = next(row for row in summary if row["condition"] == "strong_spurious_correlation")
    supports = (
        strong["goal_rate_mean"] >= 0.8
        and trend_map["spurious_action_rate"]["slope"] > 0
        and trend_map["spurious_action_rate"]["p_value_normal"] < 0.05
        and trend_map["spurious_Q_advantage"]["slope"] > 0
        and trend_map["spurious_Q_advantage"]["p_value_normal"] < 0.05
    )
    lines = [
        "# Spurious Correlation Experiment",
        "",
        "## Research question",
        "Does an RL agent learn and preserve a non-causal action when that action is statistically associated with reward during training?",
        "",
        "## Why this follows from previous experiments",
        "The distractor-count experiment was likely confounded by action-space geometry, and temporal proximity alone did not produce a near-greater-than-far gradient. This experiment adds an explicit training correlation and then removes it during evaluation.",
        "",
        "## Environment design",
        f"Fixed-horizon tabular environment. Action 0 is the only true causal reward action at phase {TARGET_PHASE}. Action 1 is spurious and never rewards or changes reward availability. Actions 2 and 3 are neutral distractors. The action-space size is fixed across all conditions.",
        "",
        "## Training/evaluation distribution shift",
        f"During training, action 1 is available at phase {TARGET_PHASE - 1} with condition-specific probability. During evaluation, action 1 availability is decorrelated and sampled independently at each pre-reward phase with probability {EVAL_SPURIOUS_PROB}.",
        "",
        "## Agent design",
        f"Tabular SARSA(lambda), alpha={ALPHA}, gamma={GAMMA}, lambda={LAMBDA}, epsilon {EPSILON_START}->{EPSILON_END}, {EPISODES} training episodes, {EVAL_EPISODES} evaluation episodes, 30 seeds.",
        "",
        "## Metrics",
        "CDS is normalized Q-mass assigned to non-causal actions. SPI is greedy non-causal action selection when the spurious action is available. Spurious Q advantage is Q(spurious)-mean Q(neutral). Spurious policy advantage is spurious action rate minus neutral action rate.",
        "",
        "## Results table",
        "",
        "| condition | reward | goal_rate | SPI | spurious_rate | Q_advantage | policy_advantage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['condition']} | {row['reward_mean']:.3f} +/- {row['reward_ci95']:.3f} | "
            f"{row['goal_rate_mean']:.3f} +/- {row['goal_rate_ci95']:.3f} | "
            f"{row['SPI_mean']:.3f} +/- {row['SPI_ci95']:.3f} | "
            f"{row['spurious_action_rate_mean']:.3f} +/- {row['spurious_action_rate_ci95']:.3f} | "
            f"{row['spurious_Q_advantage_mean']:.3f} +/- {row['spurious_Q_advantage_ci95']:.3f} | "
            f"{row['spurious_policy_advantage_mean']:.3f} +/- {row['spurious_policy_advantage_ci95']:.3f} |"
        )
    lines.extend(["", "## Trend table", "", "| metric | slope | 95% CI | std effect | p approx |", "|---|---:|---:|---:|---:|"])
    for row in trends:
        lines.append(
            f"| {row['metric']} | {row['slope']:.4f} | +/- {row['slope_ci95']:.4f} | "
            f"{row['standardized_effect']:.3f} | {row['p_value_normal']:.4f} |"
        )
    lines.extend(["", "## Strong-vs-none comparison", "", "| metric | strong - none | 95% CI | p approx |", "|---|---:|---:|---:|"])
    for row in pairs:
        lines.append(f"| {row['metric']} | {row['strong_minus_none']:.4f} | +/- {row['ci95']:.4f} | {row['p_value_normal']:.4f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            f"Strong condition goal_rate={strong['goal_rate_mean']:.3f}. Spurious action slope={trend_map['spurious_action_rate']['slope']:.4f}. Spurious Q-advantage slope={trend_map['spurious_Q_advantage']['slope']:.4f}.",
            "",
            "## Does this support computational superstition?",
            f"Supported: {supports}.",
            "",
            "## Limitations",
            "This design creates a distribution shift between training and evaluation, so persistent spurious action use could reflect learned superstition or imperfect adaptation to cue availability. However, the spurious action never directly causes reward and action-space size is fixed, so a positive trend would not be a simple action-count artifact.",
            "",
            "## Recommended next experiment",
            "Add a reversal phase where the spurious action remains available but is explicitly anti-correlated with reward, then measure extinction speed of spurious action selection.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "spurious_correlation_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows = []
    for condition, strength in CONDITIONS:
        for seed in SEEDS:
            print(f"condition={condition} seed={seed}", flush=True)
            rows.append(run_seed(condition, strength, seed))
    summary = summarize(rows)
    trends = trend_analysis(rows)
    pairs = pairwise(rows)
    result_fields = [
        "condition",
        "correlation_strength",
        "seed",
        "reward",
        "goal_rate",
        "CDS",
        "SPI",
        "spurious_action_rate",
        "neutral_action_rate",
        "causal_action_rate",
        "Q_spurious",
        "Q_neutral_mean",
        "Q_causal",
        "spurious_Q_advantage",
        "spurious_policy_advantage",
    ]
    write_csv(os.path.join(RESULTS_DIR, "spurious_correlation_results.csv"), rows, result_fields)
    summary_fields = ["condition", "correlation_strength", "n"]
    for metric in result_fields[3:]:
        summary_fields.extend([f"{metric}_mean", f"{metric}_ci95"])
    write_csv(os.path.join(RESULTS_DIR, "spurious_correlation_summary.csv"), summary, summary_fields)
    write_csv(
        os.path.join(RESULTS_DIR, "spurious_correlation_trends.csv"),
        trends,
        ["metric", "slope", "slope_ci95", "standardized_effect", "t_stat", "p_value_normal"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "spurious_correlation_pairwise.csv"),
        pairs,
        ["metric", "strong_minus_none", "ci95", "t_stat", "p_value_normal"],
    )
    write_plot(summary)
    write_report(summary, trends, pairs)


if __name__ == "__main__":
    run()
