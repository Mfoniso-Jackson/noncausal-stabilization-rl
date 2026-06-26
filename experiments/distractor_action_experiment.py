import csv
import math
import os
import random
import shutil
import statistics
import struct
import subprocess
import zlib


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

DISTRACTOR_COUNTS = [0, 2, 5, 10]
SEEDS = list(range(30))
EPISODES = 2500
EVAL_EPISODES = 100
HORIZON = 20
TARGET_PHASE = 10
ALPHA = 0.12
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
STEP_PENALTY = -0.002


class DistractorActionEnv:
    def __init__(self, distractor_count, horizon=HORIZON, target_phase=TARGET_PHASE):
        self.distractor_count = distractor_count
        self.horizon = horizon
        self.target_phase = target_phase
        self.action_count = 1 + distractor_count
        self.reset()

    def reset(self):
        self.phase = 0
        self.done = False
        self.goal_completed = False
        return self.phase

    def step(self, action):
        if self.done:
            raise RuntimeError("step called after episode termination")
        reward = STEP_PENALTY
        if self.phase == self.target_phase and action == 0:
            reward += 1.0
            self.goal_completed = True
            self.done = True
        else:
            self.phase += 1
            self.done = self.phase >= self.horizon
        return self.phase, reward, self.done, {"goal_completed": self.goal_completed}


class QLearningAgent:
    def __init__(self, state_count, action_count, seed):
        self.state_count = state_count
        self.action_count = action_count
        self.rng = random.Random(seed)
        self.q = [[0.0 for _ in range(action_count)] for _ in range(state_count + 1)]

    def act(self, state, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.action_count)
        return self.greedy_action(state)

    def greedy_action(self, state):
        values = self.q[state]
        best = max(values)
        best_actions = [i for i, value in enumerate(values) if abs(value - best) < 1e-12]
        return self.rng.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += GAMMA * max(self.q[next_state])
        self.q[state][action] += ALPHA * (target - self.q[state][action])


def train_and_evaluate(distractor_count, seed):
    env = DistractorActionEnv(distractor_count)
    agent = QLearningAgent(env.horizon, env.action_count, seed)
    epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / EPISODES)
    epsilon = EPSILON_START
    for _ in range(EPISODES):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        epsilon = max(EPSILON_END, epsilon * epsilon_decay)
    reward, goal_rate, spi = evaluate_policy(agent, distractor_count)
    cds = credit_diffusion_score(agent, distractor_count)
    return {
        "condition": f"distractors_{distractor_count}",
        "distractor_count": distractor_count,
        "seed": seed,
        "reward": round(reward, 6),
        "goal_rate": round(goal_rate, 6),
        "CDS": round(cds, 6),
        "SPI": round(spi, 6),
    }


def evaluate_policy(agent, distractor_count):
    env = DistractorActionEnv(distractor_count)
    total_reward = 0.0
    goals = 0
    distractor_decisions = 0
    relevant_decisions = 0
    for _ in range(EVAL_EPISODES):
        state = env.reset()
        done = False
        while not done:
            action = agent.greedy_action(state)
            if state < TARGET_PHASE and distractor_count > 0:
                relevant_decisions += 1
                if action > 0:
                    distractor_decisions += 1
            state, reward, done, info = env.step(action)
            total_reward += reward
        goals += int(env.goal_completed)
    spi = distractor_decisions / relevant_decisions if relevant_decisions else 0.0
    return total_reward / EVAL_EPISODES, goals / EVAL_EPISODES, spi


def credit_diffusion_score(agent, distractor_count):
    if distractor_count == 0:
        return 0.0
    masses = []
    for state in range(TARGET_PHASE):
        values = agent.q[state]
        max_q = max(values)
        exp_values = [math.exp(value - max_q) for value in values]
        denom = sum(exp_values)
        masses.append(sum(exp_values[1:]) / denom if denom else 0.0)
    return sum(masses) / len(masses)


def summarize(rows):
    summary = []
    for count in DISTRACTOR_COUNTS:
        subset = [row for row in rows if row["distractor_count"] == count]
        item = {"condition": f"distractors_{count}", "distractor_count": count, "n": len(subset)}
        for metric in ["reward", "goal_rate", "CDS", "SPI"]:
            values = [float(row[metric]) for row in subset]
            m = mean(values)
            ci = 1.96 * sem(values)
            item[f"{metric}_mean"] = round(m, 6)
            item[f"{metric}_ci95"] = round(ci, 6)
        summary.append(item)
    return summary


def trend_analysis(rows):
    trends = []
    for metric in ["reward", "goal_rate", "CDS", "SPI"]:
        x = [float(row["distractor_count"]) for row in rows]
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
                "supports_positive_trend": metric in ["CDS", "SPI"] and fit["slope"] > 0 and fit["p_value"] < 0.05,
            }
        )
    return trends


def linear_regression(x, y):
    n = len(x)
    x_mean = mean(x)
    y_mean = mean(y)
    sxx = sum((value - x_mean) ** 2 for value in x)
    sxy = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x, y))
    slope = sxy / sxx if sxx else 0.0
    intercept = y_mean - slope * x_mean
    residuals = [yv - (intercept + slope * xv) for xv, yv in zip(x, y)]
    df = max(1, n - 2)
    residual_var = sum(r * r for r in residuals) / df
    slope_se = math.sqrt(residual_var / sxx) if sxx else 0.0
    t_stat = slope / slope_se if slope_se else 0.0
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
    x_sd = statistics.stdev(x) if len(x) > 1 else 0.0
    y_sd = statistics.stdev(y) if len(y) > 1 else 0.0
    standardized = slope * x_sd / y_sd if y_sd else 0.0
    return {
        "slope": slope,
        "slope_se": slope_se,
        "standardized_effect": standardized,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def mean(values):
    return sum(values) / max(1, len(values))


def sem(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_plot(summary):
    svg_path = os.path.join(RESULTS_DIR, "distractor_action_plot.svg")
    png_path = os.path.join(RESULTS_DIR, "distractor_action_plot.png")
    write_svg_plot(svg_path, summary)
    if shutil.which("sips"):
        subprocess.run(["sips", "-s", "format", "png", svg_path, "--out", png_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(png_path):
        write_fallback_png(png_path, summary)


def write_svg_plot(path, summary):
    width, height = 920, 680
    panel_w, panel_h = 390, 250
    origins = [(70, 70), (520, 70), (70, 380), (520, 380)]
    metrics = [("reward", "Reward"), ("goal_rate", "Goal rate"), ("CDS", "CDS"), ("SPI", "SPI")]
    xs = [row["distractor_count"] for row in summary]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="460" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">Distractor Action Experiment</text>',
    ]
    for (metric, title), (ox, oy) in zip(metrics, origins):
        values = [row[f"{metric}_mean"] for row in summary]
        cis = [row[f"{metric}_ci95"] for row in summary]
        ymin = min([v - c for v, c in zip(values, cis)] + [0.0])
        ymax = max([v + c for v, c in zip(values, cis)] + [1.0])
        pad = (ymax - ymin) * 0.1
        ymin -= pad
        ymax += pad

        def sx(x):
            return ox + (x / max(xs)) * panel_w if max(xs) else ox

        def sy(y):
            return oy + (ymax - y) / (ymax - ymin) * panel_h

        parts.extend(
            [
                f'<text x="{ox + panel_w/2}" y="{oy-18}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700">{title}</text>',
                f'<line x1="{ox}" y1="{oy+panel_h}" x2="{ox+panel_w}" y2="{oy+panel_h}" stroke="#222"/>',
                f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+panel_h}" stroke="#222"/>',
                f'<text x="{ox+panel_w/2}" y="{oy+panel_h+38}" text-anchor="middle" font-family="Arial" font-size="12">Distractor count</text>',
            ]
        )
        coords = " ".join(f"{sx(x):.2f},{sy(v):.2f}" for x, v in zip(xs, values))
        parts.append(f'<polyline points="{coords}" fill="none" stroke="#245c9c" stroke-width="2.5"/>')
        for row, value, ci in zip(summary, values, cis):
            x = sx(row["distractor_count"])
            y = sy(value)
            parts.append(f'<line x1="{x:.2f}" y1="{sy(value-ci):.2f}" x2="{x:.2f}" y2="{sy(value+ci):.2f}" stroke="#333"/>')
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#245c9c"/>')
            parts.append(f'<text x="{x:.2f}" y="{oy+panel_h+18}" text-anchor="middle" font-family="Arial" font-size="11">{row["distractor_count"]}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_fallback_png(path, summary):
    width, height = 900, 620
    pixels = [(255, 255, 255) for _ in range(width * height)]

    def put(x, y, color):
        if 0 <= x < width and 0 <= y < height:
            pixels[y * width + x] = color

    def line(x0, y0, x1, y1, color):
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(steps + 1):
            t = i / steps
            put(int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t), color)

    metrics = ["reward", "goal_rate", "CDS", "SPI"]
    origins = [(60, 60), (500, 60), (60, 340), (500, 340)]
    xs = [row["distractor_count"] for row in summary]
    for metric, (ox, oy) in zip(metrics, origins):
        values = [row[f"{metric}_mean"] for row in summary]
        ymax = max(values + [1.0])
        panel_w, panel_h = 320, 190
        line(ox, oy + panel_h, ox + panel_w, oy + panel_h, (0, 0, 0))
        line(ox, oy, ox, oy + panel_h, (0, 0, 0))
        points = []
        for x, value in zip(xs, values):
            px = int(ox + (x / max(xs)) * panel_w) if max(xs) else ox
            py = int(oy + (1.0 - value / ymax) * panel_h)
            points.append((px, py))
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    put(px + dx, py + dy, (36, 92, 156))
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


def write_report(summary, trends):
    supports_cds = next(row for row in trends if row["metric"] == "CDS")["supports_positive_trend"]
    supports_spi = next(row for row in trends if row["metric"] == "SPI")["supports_positive_trend"]
    min_goal = min(row["goal_rate_mean"] for row in summary)
    supports = supports_cds and supports_spi and min_goal >= 0.8
    lines = [
        "# Distractor Action Experiment",
        "",
        "## Research question",
        "Does superstitious behavior increase as the number of non-causal but available actions increases?",
        "",
        "## Hypothesis",
        "Increasing distractor actions should increase credit diffusion onto non-causal actions and increase superstitious policy choices, while preserving task learnability.",
        "",
        "## Environment description",
        f"Tabular fixed-horizon environment with horizon {HORIZON}. The true causal action is action 0 and produces reward only at phase {TARGET_PHASE}. Distractor actions are available at every phase but never causally produce reward. Pre-goal distractor choices do not prevent later goal completion, allowing credit diffusion without making the task impossible.",
        "",
        "## Agent description",
        f"Tabular Q-learning with alpha={ALPHA}, gamma={GAMMA}, epsilon annealed from {EPSILON_START} to {EPSILON_END}, trained for {EPISODES} episodes per seed.",
        "",
        "## Metrics",
        "Reward is mean evaluation return over 100 greedy episodes. Goal rate is the fraction of evaluation episodes completing the causal reward action. CDS is the average softmax-normalized Q-value mass assigned to distractor actions across pre-goal phases. SPI is the fraction of greedy pre-goal decisions selecting a distractor action.",
        "",
        "## Results table",
        "",
        "| condition | reward | goal_rate | CDS | SPI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['condition']} | {row['reward_mean']:.3f} +/- {row['reward_ci95']:.3f} | "
            f"{row['goal_rate_mean']:.3f} +/- {row['goal_rate_ci95']:.3f} | "
            f"{row['CDS_mean']:.3f} +/- {row['CDS_ci95']:.3f} | "
            f"{row['SPI_mean']:.3f} +/- {row['SPI_ci95']:.3f} |"
        )
    lines.extend(["", "## Trend analysis table", "", "| metric | slope | 95% CI | standardized effect | p approx |", "|---|---:|---:|---:|---:|"])
    for row in trends:
        lines.append(
            f"| {row['metric']} | {row['slope']:.4f} | +/- {row['slope_ci95']:.4f} | "
            f"{row['standardized_effect']:.3f} | {row['p_value_normal']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            f"CDS positive trend supported: {supports_cds}. SPI positive trend supported: {supports_spi}. Minimum goal rate across distractor conditions was {min_goal:.3f}.",
            "",
            "## Credit Diffusion Hypothesis",
            f"Supported: {supports}.",
            "",
            "## Next recommended experiment",
            "Add an intervention phase that removes distractor actions after training and measures whether policies continue to allocate action probability or value to now-unavailable distractor analogues. This would separate harmless exploratory rituals from persistent computational superstition.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "distractor_action_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows = []
    for count in DISTRACTOR_COUNTS:
        for seed in SEEDS:
            print(f"condition=distractors_{count} seed={seed}", flush=True)
            rows.append(train_and_evaluate(count, seed))
    summary = summarize(rows)
    trends = trend_analysis(rows)
    write_csv(
        os.path.join(RESULTS_DIR, "distractor_action_results.csv"),
        rows,
        ["condition", "distractor_count", "seed", "reward", "goal_rate", "CDS", "SPI"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "distractor_action_summary.csv"),
        summary,
        [
            "condition",
            "distractor_count",
            "n",
            "reward_mean",
            "reward_ci95",
            "goal_rate_mean",
            "goal_rate_ci95",
            "CDS_mean",
            "CDS_ci95",
            "SPI_mean",
            "SPI_ci95",
        ],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "distractor_action_trends.csv"),
        trends,
        ["metric", "slope", "slope_ci95", "standardized_effect", "t_stat", "p_value_normal", "supports_positive_trend"],
    )
    make_plot(summary)
    write_report(summary, trends)


if __name__ == "__main__":
    run()
