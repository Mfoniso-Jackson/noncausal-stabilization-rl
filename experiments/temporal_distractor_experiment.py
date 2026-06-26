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

CONDITIONS = ["no_distractors", "balanced_distractors", "temporally_biased_distractors"]
DISTRACTOR_TYPES = ["near", "mid", "far", "neutral"]
TYPE_TO_ACTION = {"near": 1, "mid": 2, "far": 3, "neutral": 4}
ACTION_TO_TYPE = {value: key for key, value in TYPE_TO_ACTION.items()}
SEEDS = list(range(30))
EPISODES = 2500
EVAL_EPISODES = 100
HORIZON = 20
TARGET_PHASE = 10
ALPHA = 0.08
GAMMA = 0.95
LAMBDA = 0.80
EPSILON_START = 1.0
EPSILON_END = 0.04
STEP_PENALTY = -0.002


class TemporalDistractorEnv:
    def __init__(self, condition, seed=0):
        self.condition = condition
        self.rng = random.Random(seed)
        self.action_count = 1 if condition == "no_distractors" else 5
        self.reset()

    def reset(self):
        self.phase = 0
        self.done = False
        self.goal_completed = False
        self.neutral_phases = set(self.rng.sample(range(1, TARGET_PHASE), 3))
        return self.phase

    def available_actions(self, phase=None):
        phase = self.phase if phase is None else phase
        actions = [0]
        if self.condition == "no_distractors" or phase >= TARGET_PHASE:
            return actions
        if self.condition == "balanced_distractors":
            return [0, 1, 2, 3, 4]
        if phase in [7, 8, 9]:
            actions.append(TYPE_TO_ACTION["near"])
        if phase in [4, 5, 6]:
            actions.append(TYPE_TO_ACTION["mid"])
        if phase in [1, 2, 3]:
            actions.append(TYPE_TO_ACTION["far"])
        if phase in self.neutral_phases:
            actions.append(TYPE_TO_ACTION["neutral"])
        return sorted(set(actions))

    def step(self, action):
        if self.done:
            raise RuntimeError("step called after episode termination")
        reward = STEP_PENALTY
        if self.phase == TARGET_PHASE and action == 0:
            reward += 1.0
            self.goal_completed = True
            self.done = True
        else:
            self.phase += 1
            self.done = self.phase >= HORIZON
        return self.phase, reward, self.done, {"goal_completed": self.goal_completed}


class SarsaLambdaAgent:
    def __init__(self, state_count, action_count, seed):
        self.state_count = state_count
        self.action_count = action_count
        self.rng = random.Random(seed)
        self.q = [[0.0 for _ in range(action_count)] for _ in range(state_count + 1)]

    def act(self, state, available, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.choice(available)
        return self.greedy_action(state, available)

    def greedy_action(self, state, available):
        best = max(self.q[state][action] for action in available)
        best_actions = [action for action in available if abs(self.q[state][action] - best) < 1e-12]
        return self.rng.choice(best_actions)

    def train_episode(self, env, epsilon):
        eligibility = [[0.0 for _ in range(self.action_count)] for _ in range(self.state_count + 1)]
        state = env.reset()
        action = self.act(state, env.available_actions(state), epsilon)
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            if done:
                target = reward
                next_action = None
            else:
                next_action = self.act(next_state, env.available_actions(next_state), epsilon)
                target = reward + GAMMA * self.q[next_state][next_action]
            delta = target - self.q[state][action]
            eligibility[state][action] += 1.0
            for s in range(self.state_count + 1):
                for a in range(self.action_count):
                    self.q[s][a] += ALPHA * delta * eligibility[s][a]
                    eligibility[s][a] *= GAMMA * LAMBDA
            state = next_state
            action = next_action if next_action is not None else 0


def train_and_evaluate(condition, seed):
    env = TemporalDistractorEnv(condition, seed)
    agent = SarsaLambdaAgent(HORIZON, env.action_count, seed)
    epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / EPISODES)
    epsilon = EPSILON_START
    for _ in range(EPISODES):
        agent.train_episode(env, epsilon)
        epsilon = max(EPSILON_END, epsilon * epsilon_decay)
    return evaluate(condition, seed, agent)


def evaluate(condition, seed, agent):
    eval_env = TemporalDistractorEnv(condition, seed + 100000)
    total_reward = 0.0
    goals = 0
    type_available = {kind: 0 for kind in DISTRACTOR_TYPES}
    type_selected = {kind: 0 for kind in DISTRACTOR_TYPES}
    q_mass = {kind: [] for kind in DISTRACTOR_TYPES}
    cds_values = []
    for _ in range(EVAL_EPISODES):
        state = eval_env.reset()
        done = False
        while not done:
            available = eval_env.available_actions(state)
            if condition != "no_distractors" and state < TARGET_PHASE:
                masses = normalized_action_mass(agent.q[state], available)
                distractor_mass = 0.0
                for action in available:
                    if action in ACTION_TO_TYPE:
                        kind = ACTION_TO_TYPE[action]
                        type_available[kind] += 1
                        q_mass[kind].append(masses[action])
                        distractor_mass += masses[action]
                cds_values.append(distractor_mass)
            action = agent.greedy_action(state, available)
            if action in ACTION_TO_TYPE and state < TARGET_PHASE:
                type_selected[ACTION_TO_TYPE[action]] += 1
            state, reward, done, _ = eval_env.step(action)
            total_reward += reward
        goals += int(eval_env.goal_completed)
    spi_by_type = {
        kind: type_selected[kind] / type_available[kind] if type_available[kind] else 0.0
        for kind in DISTRACTOR_TYPES
    }
    q_mass_by_type = {kind: mean(q_mass[kind]) if q_mass[kind] else 0.0 for kind in DISTRACTOR_TYPES}
    return {
        "condition": condition,
        "seed": seed,
        "reward": round(total_reward / EVAL_EPISODES, 6),
        "goal_rate": round(goals / EVAL_EPISODES, 6),
        "CDS": round(mean(cds_values) if cds_values else 0.0, 6),
        "SPI": round(sum(type_selected.values()) / sum(type_available.values()) if sum(type_available.values()) else 0.0, 6),
        "SPI_near": round(spi_by_type["near"], 6),
        "SPI_mid": round(spi_by_type["mid"], 6),
        "SPI_far": round(spi_by_type["far"], 6),
        "SPI_neutral": round(spi_by_type["neutral"], 6),
        "Q_mass_near": round(q_mass_by_type["near"], 6),
        "Q_mass_mid": round(q_mass_by_type["mid"], 6),
        "Q_mass_far": round(q_mass_by_type["far"], 6),
        "Q_mass_neutral": round(q_mass_by_type["neutral"], 6),
    }


def normalized_action_mass(values, available):
    max_q = max(values[action] for action in available)
    exp_values = {action: math.exp(values[action] - max_q) for action in available}
    denom = sum(exp_values.values())
    return {action: exp_values[action] / denom for action in available}


def summarize(rows):
    summary = []
    metrics = [
        "reward",
        "goal_rate",
        "CDS",
        "SPI",
        "SPI_near",
        "SPI_mid",
        "SPI_far",
        "SPI_neutral",
        "Q_mass_near",
        "Q_mass_mid",
        "Q_mass_far",
        "Q_mass_neutral",
    ]
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        item = {"condition": condition, "n": len(subset)}
        for metric in metrics:
            values = [float(row[metric]) for row in subset]
            item[f"{metric}_mean"] = round(mean(values), 6)
            item[f"{metric}_ci95"] = round(1.96 * sem(values), 6)
        summary.append(item)
    return summary


def comparisons(rows):
    output = []
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        for metric, near_key, far_key in [
            ("SPI", "SPI_near", "SPI_far"),
            ("Q_mass", "Q_mass_near", "Q_mass_far"),
        ]:
            diffs = [float(row[near_key]) - float(row[far_key]) for row in subset]
            m = mean(diffs)
            se = sem(diffs)
            t_stat = m / se if se > 0 else 0.0
            p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
            output.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "near_minus_far": round(m, 6),
                    "ci95": round(1.96 * se, 6),
                    "t_stat": round(t_stat, 6),
                    "p_value_normal": round(p_value, 6),
                    "near_greater_far": m > 0 and p_value < 0.05,
                }
            )
    return output


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
    png_path = os.path.join(RESULTS_DIR, "temporal_distractor_plot.png")
    write_png_plot(png_path, summary)


def write_png_plot(path, summary):
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
        ("SPI", ["SPI_near_mean", "SPI_mid_mean", "SPI_far_mean", "SPI_neutral_mean"], 70, 70),
        ("Q mass", ["Q_mass_near_mean", "Q_mass_mid_mean", "Q_mass_far_mean", "Q_mass_neutral_mean"], 540, 70),
        ("Reward", ["reward_mean"], 70, 360),
        ("Goal rate", ["goal_rate_mean"], 540, 360),
    ]
    colors = [(36, 92, 156), (214, 107, 42), (47, 125, 85), (139, 63, 146)]
    for _, keys, ox, oy in panels:
        panel_w, panel_h = 340, 190
        line(ox, oy + panel_h, ox + panel_w, oy + panel_h, (0, 0, 0))
        line(ox, oy, ox, oy + panel_h, (0, 0, 0))
        for key_idx, key in enumerate(keys):
            values = [row[key] for row in summary]
            ymax = max(values + [1.0])
            points = []
            for i, value in enumerate(values):
                x = ox + int(i * panel_w / max(1, len(summary) - 1))
                y = oy + int((1.0 - value / ymax) * panel_h)
                points.append((x, y))
                dot(x, y, colors[key_idx % len(colors)])
            for a, b in zip(points, points[1:]):
                line(a[0], a[1], b[0], b[1], colors[key_idx % len(colors)])
    raw = b"".join(b"\x00" + bytes([channel for pixel in pixels[y * width : (y + 1) * width] for channel in pixel]) for y in range(height))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(png_chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(png_chunk(b"IEND", b""))


def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_report(summary, comp):
    biased = next(row for row in summary if row["condition"] == "temporally_biased_distractors")
    biased_spi = next(row for row in comp if row["condition"] == "temporally_biased_distractors" and row["metric"] == "SPI")
    biased_q = next(row for row in comp if row["condition"] == "temporally_biased_distractors" and row["metric"] == "Q_mass")
    high_performance = biased["goal_rate_mean"] >= 0.8
    temporal_supported = biased_spi["near_greater_far"] and biased_q["near_greater_far"] and high_performance
    lines = [
        "# Temporal Distractor Experiment",
        "",
        "## Research question",
        "Was the previous distractor-count effect genuine temporal credit misassignment, or a mechanical artifact of enlarging the action space?",
        "",
        "## Why this experiment is needed",
        "The previous CDS values closely followed the proportion of distractor actions in the action set. This follow-up fixes the distractor types and asks whether temporal proximity to reward predicts superstitious selection and Q-value mass.",
        "",
        "## Environment design",
        f"Tabular fixed-horizon environment with target reward at phase {TARGET_PHASE}. Action 0 is the true causal action at the reward phase. Distractor actions never causally affect reward. In the temporally biased condition, near/mid/far distractors are available close to, several steps before, or far before reward respectively; neutral availability is random and unrelated to reward timing.",
        "",
        "## Agent design",
        f"Tabular SARSA(lambda) with alpha={ALPHA}, gamma={GAMMA}, lambda={LAMBDA}, epsilon annealed from {EPSILON_START} to {EPSILON_END}, trained for {EPISODES} episodes.",
        "",
        "## Metrics",
        "Reward and goal_rate measure task learnability. CDS is total normalized distractor Q-mass when distractors are available. SPI is greedy distractor selection rate. Type-specific SPI and Q-mass compare near, mid, far, and neutral distractors.",
        "",
        "## Results tables",
        "",
        "| condition | reward | goal_rate | CDS | SPI | SPI_near | SPI_mid | SPI_far | Q_near | Q_mid | Q_far |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['condition']} | {row['reward_mean']:.3f} +/- {row['reward_ci95']:.3f} | "
            f"{row['goal_rate_mean']:.3f} +/- {row['goal_rate_ci95']:.3f} | "
            f"{row['CDS_mean']:.3f} +/- {row['CDS_ci95']:.3f} | "
            f"{row['SPI_mean']:.3f} +/- {row['SPI_ci95']:.3f} | "
            f"{row['SPI_near_mean']:.3f} | {row['SPI_mid_mean']:.3f} | {row['SPI_far_mean']:.3f} | "
            f"{row['Q_mass_near_mean']:.3f} | {row['Q_mass_mid_mean']:.3f} | {row['Q_mass_far_mean']:.3f} |"
        )
    lines.extend(["", "## Near/mid/far comparison", "", "| condition | metric | near_minus_far | 95% CI | p approx |", "|---|---|---:|---:|---:|"])
    for row in comp:
        lines.append(
            f"| {row['condition']} | {row['metric']} | {row['near_minus_far']:.3f} | +/- {row['ci95']:.3f} | {row['p_value_normal']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            f"In the temporally biased condition, SPI near-minus-far = {biased_spi['near_minus_far']:.3f}; Q-mass near-minus-far = {biased_q['near_minus_far']:.3f}; goal_rate = {biased['goal_rate_mean']:.3f}.",
            "",
            "## Genuine or mechanical?",
            f"Temporal proximity predicts superstitious action selection and Q-mass: {temporal_supported}.",
            "If true, this argues against a pure action-space-size artifact and supports temporal credit-assignment error as a contributor. If false, the earlier distractor-count effect is more likely mechanical or policy-tie geometry.",
            "",
            "## Recommended next experiment",
            "Hold temporal availability fixed and manipulate reward delay after near versus far distractor exposure. This would test whether the proximity gradient strengthens when credit assignment is explicitly made harder.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "temporal_distractor_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            print(f"condition={condition} seed={seed}", flush=True)
            rows.append(train_and_evaluate(condition, seed))
    summary = summarize(rows)
    comp = comparisons(rows)
    write_csv(
        os.path.join(RESULTS_DIR, "temporal_distractor_results.csv"),
        rows,
        [
            "condition",
            "seed",
            "reward",
            "goal_rate",
            "CDS",
            "SPI",
            "SPI_near",
            "SPI_mid",
            "SPI_far",
            "SPI_neutral",
            "Q_mass_near",
            "Q_mass_mid",
            "Q_mass_far",
            "Q_mass_neutral",
        ],
    )
    summary_fields = ["condition", "n"]
    for metric in [
        "reward",
        "goal_rate",
        "CDS",
        "SPI",
        "SPI_near",
        "SPI_mid",
        "SPI_far",
        "SPI_neutral",
        "Q_mass_near",
        "Q_mass_mid",
        "Q_mass_far",
        "Q_mass_neutral",
    ]:
        summary_fields.extend([f"{metric}_mean", f"{metric}_ci95"])
    write_csv(os.path.join(RESULTS_DIR, "temporal_distractor_summary.csv"), summary, summary_fields)
    write_csv(
        os.path.join(RESULTS_DIR, "temporal_distractor_trends.csv"),
        comp,
        ["condition", "metric", "near_minus_far", "ci95", "t_stat", "p_value_normal", "near_greater_far"],
    )
    make_plot(summary)
    write_report(summary, comp)


if __name__ == "__main__":
    run()
