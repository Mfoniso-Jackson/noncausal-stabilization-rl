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
    {
        "condition": "clear_causality",
        "ambiguity_level": 0.0,
        "reward_prob": 0.95,
        "cue_given_reward": 0.50,
    },
    {
        "condition": "moderate_ambiguity",
        "ambiguity_level": 1.0,
        "reward_prob": 0.70,
        "cue_given_reward": 0.75,
    },
    {
        "condition": "high_ambiguity",
        "ambiguity_level": 2.0,
        "reward_prob": 0.55,
        "cue_given_reward": 0.90,
    },
    {
        "condition": "extreme_ambiguity",
        "ambiguity_level": 3.0,
        "reward_prob": 0.45,
        "cue_given_reward": 0.95,
    },
]

SEEDS = list(range(30))
EPISODES = 3000
EVAL_EPISODES = 500
HORIZON = 14
TARGET_PHASE = 10
PROBE_PHASE = TARGET_PHASE - 1
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
FALSE_CUE_PROB = 0.10
DECORRELATED_CUE_PROB = 0.50


class AmbiguousCausalityEnv:
    def __init__(self, reward_prob, cue_given_reward, mode, seed=0):
        self.reward_prob = reward_prob
        self.cue_given_reward = cue_given_reward
        self.mode = mode
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.phase = 0
        self.done = False
        self.reward_latent = self.rng.random() < self.reward_prob
        self.cue_present = self._sample_cue()
        self.causal_target_selected = False
        self.reward_obtained = False
        return self.observe()

    def _sample_cue(self):
        if self.mode == "training_distribution":
            prob = self.cue_given_reward if self.reward_latent else FALSE_CUE_PROB
            return self.rng.random() < prob
        if self.mode == "decorrelated":
            return self.rng.random() < DECORRELATED_CUE_PROB
        raise ValueError(f"unknown mode: {self.mode}")

    def observe(self):
        cue_bit = 1 if self.phase == PROBE_PHASE and self.cue_present else 0
        return self.phase * 2 + cue_bit

    def available_actions(self):
        return [0, 1, 2, 3]

    def step(self, action):
        if self.done:
            raise RuntimeError("step called after episode termination")
        reward = STEP_PENALTY
        if self.phase == TARGET_PHASE:
            if action == ACTION_CAUSAL:
                self.causal_target_selected = True
                if self.reward_latent:
                    reward += 1.0
                    self.reward_obtained = True
            self.done = True
        else:
            self.phase += 1
            self.done = self.phase >= HORIZON
        return self.observe(), reward, self.done, {
            "causal_target_selected": self.causal_target_selected,
            "reward_obtained": self.reward_obtained,
        }


class SarsaLambdaAgent:
    def __init__(self, state_count, seed):
        self.rng = random.Random(seed)
        self.q = [[0.0 for _ in range(ACTION_COUNT)] for _ in range(state_count)]

    def act(self, state, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(ACTION_COUNT)
        return self.greedy_action(state)

    def greedy_action(self, state):
        values = self.q[state]
        best = max(values)
        best_actions = [action for action, value in enumerate(values) if abs(value - best) < 1e-12]
        return self.rng.choice(best_actions)

    def train_episode(self, env, epsilon):
        traces = [[0.0 for _ in range(ACTION_COUNT)] for _ in range(len(self.q))]
        state = env.reset()
        action = self.act(state, epsilon)
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            if done:
                target = reward
                next_action = None
            else:
                next_action = self.act(next_state, epsilon)
                target = reward + GAMMA * self.q[next_state][next_action]
            delta = target - self.q[state][action]
            traces[state][action] += 1.0
            for s in range(len(self.q)):
                for a in range(ACTION_COUNT):
                    self.q[s][a] += ALPHA * delta * traces[s][a]
                    traces[s][a] *= GAMMA * LAMBDA
            state = next_state
            action = next_action if next_action is not None else ACTION_CAUSAL


def run_seed(config, seed):
    train_env = AmbiguousCausalityEnv(
        config["reward_prob"],
        config["cue_given_reward"],
        mode="training_distribution",
        seed=seed,
    )
    agent = SarsaLambdaAgent(HORIZON * 2 + 2, seed)
    epsilon = EPSILON_START
    epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / EPISODES)
    for _ in range(EPISODES):
        agent.train_episode(train_env, epsilon)
        epsilon = max(EPSILON_END, epsilon * epsilon_decay)
    rows = []
    for mode in ["training_distribution", "decorrelated"]:
        metrics = evaluate(agent, config, seed, mode)
        rows.append(
            {
                "condition": config["condition"],
                "ambiguity_level": config["ambiguity_level"],
                "reward_prob": config["reward_prob"],
                "cue_given_reward": config["cue_given_reward"],
                "eval_mode": mode,
                "seed": seed,
                **metrics,
            }
        )
    random_metrics = evaluate_random_policy(config, seed)
    return rows, random_metrics


def evaluate(agent, config, seed, mode):
    env = AmbiguousCausalityEnv(config["reward_prob"], config["cue_given_reward"], mode=mode, seed=seed + 50000)
    total_reward = 0.0
    reward_obtained = 0
    target_causal = 0
    probe_decisions = 0
    spurious_choices = 0
    neutral_choices = 0
    causal_choices = 0
    cds_values = []
    q_spurious = []
    q_neutral = []
    q_causal = []
    for _ in range(EVAL_EPISODES):
        state = env.reset()
        done = False
        while not done:
            if env.phase == PROBE_PHASE and env.cue_present:
                probe_decisions += 1
                values = agent.q[state]
                masses = softmax_mass(values)
                cds_values.append(masses[ACTION_SPURIOUS] + sum(masses[a] for a in NEUTRAL_ACTIONS))
                q_spurious.append(values[ACTION_SPURIOUS])
                q_neutral.append(mean(values[a] for a in NEUTRAL_ACTIONS))
                q_causal.append(values[ACTION_CAUSAL])
            action = agent.greedy_action(state)
            if env.phase == PROBE_PHASE and env.cue_present:
                if action == ACTION_SPURIOUS:
                    spurious_choices += 1
                elif action in NEUTRAL_ACTIONS:
                    neutral_choices += 1
                elif action == ACTION_CAUSAL:
                    causal_choices += 1
            state, reward, done, info = env.step(action)
            total_reward += reward
        reward_obtained += int(env.reward_obtained)
        target_causal += int(env.causal_target_selected)
    q_spur = mean(q_spurious) if q_spurious else 0.0
    q_neut = mean(q_neutral) if q_neutral else 0.0
    q_caus = mean(q_causal) if q_causal else 0.0
    return {
        "reward": round(total_reward / EVAL_EPISODES, 6),
        "goal_rate": round(reward_obtained / EVAL_EPISODES, 6),
        "causal_target_rate": round(target_causal / EVAL_EPISODES, 6),
        "SPI": round((spurious_choices + neutral_choices) / probe_decisions if probe_decisions else 0.0, 6),
        "CDS": round(mean(cds_values) if cds_values else 0.0, 6),
        "causal_action_rate": round(causal_choices / probe_decisions if probe_decisions else 0.0, 6),
        "spurious_action_rate": round(spurious_choices / probe_decisions if probe_decisions else 0.0, 6),
        "neutral_action_rate": round(neutral_choices / probe_decisions if probe_decisions else 0.0, 6),
        "Q_causal": round(q_caus, 6),
        "Q_spurious": round(q_spur, 6),
        "Q_neutral_mean": round(q_neut, 6),
        "spurious_Q_advantage": round(q_spur - q_neut, 6),
        "causal_Q_advantage": round(q_caus - q_neut, 6),
        "spurious_vs_causal_Q": round(q_spur - q_caus, 6),
    }


def evaluate_random_policy(config, seed):
    env = AmbiguousCausalityEnv(config["reward_prob"], config["cue_given_reward"], mode="decorrelated", seed=seed + 80000)
    rng = random.Random(seed + 90000)
    total_reward = 0.0
    rewards = 0
    target_causal = 0
    probe_decisions = 0
    spurious = 0
    neutral = 0
    causal = 0
    for _ in range(EVAL_EPISODES):
        env.reset()
        done = False
        while not done:
            action = rng.randrange(ACTION_COUNT)
            if env.phase == PROBE_PHASE and env.cue_present:
                probe_decisions += 1
                if action == ACTION_SPURIOUS:
                    spurious += 1
                elif action in NEUTRAL_ACTIONS:
                    neutral += 1
                elif action == ACTION_CAUSAL:
                    causal += 1
            _, reward, done, _ = env.step(action)
            total_reward += reward
        rewards += int(env.reward_obtained)
        target_causal += int(env.causal_target_selected)
    return {
        "condition": config["condition"],
        "ambiguity_level": config["ambiguity_level"],
        "seed": seed,
        "random_reward": round(total_reward / EVAL_EPISODES, 6),
        "random_goal_rate": round(rewards / EVAL_EPISODES, 6),
        "random_causal_target_rate": round(target_causal / EVAL_EPISODES, 6),
        "random_spurious_action_rate": round(spurious / probe_decisions if probe_decisions else 0.0, 6),
        "random_neutral_action_rate": round(neutral / probe_decisions if probe_decisions else 0.0, 6),
        "random_causal_action_rate": round(causal / probe_decisions if probe_decisions else 0.0, 6),
    }


def softmax_mass(values):
    max_q = max(values)
    exp_values = [math.exp(value - max_q) for value in values]
    denom = sum(exp_values)
    return [value / denom for value in exp_values]


def summarize(rows):
    metrics = [
        "reward",
        "goal_rate",
        "causal_target_rate",
        "SPI",
        "CDS",
        "causal_action_rate",
        "spurious_action_rate",
        "neutral_action_rate",
        "Q_causal",
        "Q_spurious",
        "Q_neutral_mean",
        "spurious_Q_advantage",
        "causal_Q_advantage",
        "spurious_vs_causal_Q",
    ]
    output = []
    for config in CONDITIONS:
        for mode in ["training_distribution", "decorrelated"]:
            subset = [row for row in rows if row["condition"] == config["condition"] and row["eval_mode"] == mode]
            item = {
                "condition": config["condition"],
                "ambiguity_level": config["ambiguity_level"],
                "eval_mode": mode,
                "n": len(subset),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in subset]
                item[f"{metric}_mean"] = round(mean(values), 6)
                item[f"{metric}_ci95"] = round(1.96 * sem(values), 6)
            output.append(item)
    return output


def baseline_summary(baselines, learned_rows):
    output = []
    for config in CONDITIONS:
        b = [row for row in baselines if row["condition"] == config["condition"]]
        learned = [
            row
            for row in learned_rows
            if row["condition"] == config["condition"] and row["eval_mode"] == "decorrelated"
        ]
        for metric, learned_key, random_key in [
            ("reward", "reward", "random_reward"),
            ("goal_rate", "goal_rate", "random_goal_rate"),
            ("causal_target_rate", "causal_target_rate", "random_causal_target_rate"),
            ("spurious_action_rate", "spurious_action_rate", "random_spurious_action_rate"),
        ]:
            learned_values = [float(row[learned_key]) for row in learned]
            random_values = [float(row[random_key]) for row in b]
            diff = mean(learned_values) - mean(random_values)
            ci = 1.96 * math.sqrt(sem(learned_values) ** 2 + sem(random_values) ** 2)
            output.append(
                {
                    "condition": config["condition"],
                    "metric": metric,
                    "learned_mean": round(mean(learned_values), 6),
                    "random_mean": round(mean(random_values), 6),
                    "learned_minus_random": round(diff, 6),
                    "ci95": round(ci, 6),
                }
            )
    return output


def trend_analysis(rows):
    trends = []
    for mode in ["training_distribution", "decorrelated"]:
        mode_rows = [row for row in rows if row["eval_mode"] == mode]
        for metric in [
            "spurious_action_rate",
            "spurious_Q_advantage",
            "spurious_vs_causal_Q",
            "SPI",
            "reward",
            "goal_rate",
        ]:
            x = [float(row["ambiguity_level"]) for row in mode_rows]
            y = [float(row[metric]) for row in mode_rows]
            fit = linear_regression(x, y)
            trends.append(
                {
                    "eval_mode": mode,
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
    for mode in ["training_distribution", "decorrelated"]:
        clear = [row for row in rows if row["condition"] == "clear_causality" and row["eval_mode"] == mode]
        extreme = [row for row in rows if row["condition"] == "extreme_ambiguity" and row["eval_mode"] == mode]
        for metric in [
            "spurious_action_rate",
            "spurious_Q_advantage",
            "spurious_vs_causal_Q",
            "SPI",
            "reward",
            "goal_rate",
        ]:
            clear_values = [float(row[metric]) for row in clear]
            extreme_values = [float(row[metric]) for row in extreme]
            diff = mean(extreme_values) - mean(clear_values)
            ci = 1.96 * math.sqrt(sem(clear_values) ** 2 + sem(extreme_values) ** 2)
            se = ci / 1.96 if ci > 0 else 0.0
            t_stat = diff / se if se > 0 else 0.0
            p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
            output.append(
                {
                    "eval_mode": mode,
                    "metric": metric,
                    "extreme_minus_clear": round(diff, 6),
                    "ci95": round(ci, 6),
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
    path = os.path.join(RESULTS_DIR, "ambiguous_causality_plot.png")
    width, height = 1000, 650
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
        ("spurious_Q_advantage_mean", 560, 70),
        ("SPI_mean", 70, 380),
        ("goal_rate_mean", 560, 380),
    ]
    xs = [row["ambiguity_level"] for row in summary if row["eval_mode"] == "decorrelated"]
    for key, ox, oy in panels:
        panel_w, panel_h = 350, 200
        rows = [row for row in summary if row["eval_mode"] == "decorrelated"]
        values = [row[key] for row in rows]
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


def write_report(summary, trends, pairs, baseline):
    decor = [row for row in summary if row["eval_mode"] == "decorrelated"]
    trend_map = {(row["eval_mode"], row["metric"]): row for row in trends}
    extreme = next(row for row in decor if row["condition"] == "extreme_ambiguity")
    clear = next(row for row in decor if row["condition"] == "clear_causality")
    baseline_extreme = [
        row for row in baseline if row["condition"] == "extreme_ambiguity" and row["metric"] == "reward"
    ][0]
    supports = (
        extreme["reward_mean"] > baseline_extreme["random_mean"]
        and extreme["goal_rate_mean"] > 0.25
        and trend_map[("decorrelated", "spurious_action_rate")]["slope"] > 0
        and trend_map[("decorrelated", "spurious_Q_advantage")]["slope"] > 0
        and trend_map[("decorrelated", "spurious_vs_causal_Q")]["slope"] > 0
    )
    lines = [
        "# Ambiguous Causality Experiment",
        "",
        "## Research question",
        "Does superstition emerge when true causal control is unreliable and a non-causal cue/action is more statistically predictive of reward during training?",
        "",
        "## Why this experiment follows from prior negative results",
        "Distractor-count effects looked mechanical, temporal proximity alone failed, and spurious correlation alone did not produce reliable spurious preference. This experiment adds causal uncertainty: the true action is probabilistic while the spurious cue predicts the latent reward outcome during training.",
        "",
        "## Environment design",
        f"Fixed-horizon tabular environment with reward phase {TARGET_PHASE}. Action 0 is the only causal reward action. Action 1 is spurious. Actions 2 and 3 are neutral. The action space is fixed across conditions.",
        "",
        "## Causal structure",
        "Reward is sampled from a latent stochastic outcome and is delivered only if the agent selects the causal action at the reward phase. The spurious cue/action never changes reward probability, transitions, or action availability.",
        "",
        "## Training/evaluation distribution shift",
        "Training cue presence is correlated with the latent reward outcome. Evaluation has both training-distribution and decorrelated modes; decorrelated evaluation samples the cue independently of reward outcome.",
        "",
        "## Agent design",
        f"Tabular SARSA(lambda), alpha={ALPHA}, gamma={GAMMA}, lambda={LAMBDA}, epsilon {EPSILON_START}->{EPSILON_END}, {EPISODES} episodes, {EVAL_EPISODES} evaluation episodes, 30 seeds.",
        "",
        "## Metrics",
        "SPI is non-causal greedy action selection at the cue probe state. CDS is normalized Q-mass assigned to non-causal actions. Spurious_Q_advantage compares Q(spurious) to neutral actions. Spurious_vs_causal_Q compares Q(spurious) to Q(causal).",
        "",
        "## Results table (decorrelated evaluation)",
        "",
        "| condition | reward | goal_rate | SPI | spurious_rate | Q_advantage | spurious_vs_causal_Q |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decor:
        lines.append(
            f"| {row['condition']} | {row['reward_mean']:.3f} +/- {row['reward_ci95']:.3f} | "
            f"{row['goal_rate_mean']:.3f} +/- {row['goal_rate_ci95']:.3f} | "
            f"{row['SPI_mean']:.3f} +/- {row['SPI_ci95']:.3f} | "
            f"{row['spurious_action_rate_mean']:.3f} +/- {row['spurious_action_rate_ci95']:.3f} | "
            f"{row['spurious_Q_advantage_mean']:.3f} +/- {row['spurious_Q_advantage_ci95']:.3f} | "
            f"{row['spurious_vs_causal_Q_mean']:.3f} +/- {row['spurious_vs_causal_Q_ci95']:.3f} |"
        )
    lines.extend(["", "## Trend analysis table", "", "| eval mode | metric | slope | 95% CI | p approx |", "|---|---|---:|---:|---:|"])
    for row in trends:
        lines.append(
            f"| {row['eval_mode']} | {row['metric']} | {row['slope']:.4f} | +/- {row['slope_ci95']:.4f} | {row['p_value_normal']:.4f} |"
        )
    lines.extend(["", "## Extreme-vs-clear comparison", "", "| eval mode | metric | extreme - clear | 95% CI | p approx |", "|---|---|---:|---:|---:|"])
    for row in pairs:
        lines.append(
            f"| {row['eval_mode']} | {row['metric']} | {row['extreme_minus_clear']:.4f} | +/- {row['ci95']:.4f} | {row['p_value_normal']:.4f} |"
        )
    lines.extend(["", "## Random baseline comparison", "", "| condition | metric | learned | random | learned - random |", "|---|---|---:|---:|---:|"])
    for row in baseline:
        lines.append(
            f"| {row['condition']} | {row['metric']} | {row['learned_mean']:.3f} | {row['random_mean']:.3f} | {row['learned_minus_random']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            f"Decorrelated extreme ambiguity reward={extreme['reward_mean']:.3f}; clear reward={clear['reward_mean']:.3f}. Spurious action trend under decorrelation={trend_map[('decorrelated', 'spurious_action_rate')]['slope']:.4f}. Spurious Q-advantage trend under decorrelation={trend_map[('decorrelated', 'spurious_Q_advantage')]['slope']:.4f}. Spurious-vs-causal Q trend under decorrelation={trend_map[('decorrelated', 'spurious_vs_causal_Q')]['slope']:.4f}.",
            "",
            "## Does this support computational superstition?",
            f"Supported: {supports}.",
            "",
            "## Limitations",
            "The cue is predictive of latent reward outcome during training, so persistence under decorrelation is the critical test. If SPI rises without spurious Q advantage, the effect may reflect policy noise or action-space effects. If reward collapses, the agent is failing rather than superstitious.",
            "",
            "## Recommended next experiment",
            "Increase partial observability or add memory constraints so the agent must infer causal structure from history, then repeat the ambiguous-causality manipulation.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "ambiguous_causality_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows = []
    baselines = []
    for config in CONDITIONS:
        for seed in SEEDS:
            print(f"condition={config['condition']} seed={seed}", flush=True)
            seed_rows, random_metrics = run_seed(config, seed)
            rows.extend(seed_rows)
            baselines.append(random_metrics)
    summary = summarize(rows)
    trends = trend_analysis(rows)
    pairs = pairwise(rows)
    baseline = baseline_summary(baselines, rows)
    result_fields = [
        "condition",
        "ambiguity_level",
        "reward_prob",
        "cue_given_reward",
        "eval_mode",
        "seed",
        "reward",
        "goal_rate",
        "causal_target_rate",
        "SPI",
        "CDS",
        "causal_action_rate",
        "spurious_action_rate",
        "neutral_action_rate",
        "Q_causal",
        "Q_spurious",
        "Q_neutral_mean",
        "spurious_Q_advantage",
        "causal_Q_advantage",
        "spurious_vs_causal_Q",
    ]
    write_csv(os.path.join(RESULTS_DIR, "ambiguous_causality_results.csv"), rows, result_fields)
    summary_fields = ["condition", "ambiguity_level", "eval_mode", "n"]
    for metric in result_fields[6:]:
        summary_fields.extend([f"{metric}_mean", f"{metric}_ci95"])
    write_csv(os.path.join(RESULTS_DIR, "ambiguous_causality_summary.csv"), summary, summary_fields)
    write_csv(
        os.path.join(RESULTS_DIR, "ambiguous_causality_trends.csv"),
        trends,
        ["eval_mode", "metric", "slope", "slope_ci95", "standardized_effect", "t_stat", "p_value_normal"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "ambiguous_causality_pairwise.csv"),
        pairs,
        ["eval_mode", "metric", "extreme_minus_clear", "ci95", "t_stat", "p_value_normal"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "ambiguous_causality_baseline.csv"),
        baseline,
        ["condition", "metric", "learned_mean", "random_mean", "learned_minus_random", "ci95"],
    )
    write_plot(summary)
    write_report(summary, trends, pairs, baseline)


if __name__ == "__main__":
    run()
