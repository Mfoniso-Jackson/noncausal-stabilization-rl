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
    ("no_proxy", 0.50, 0.50),
    ("weak_proxy", 0.65, 0.35),
    ("medium_proxy", 0.80, 0.20),
    ("strong_proxy", 0.95, 0.05),
]
AGENT_TYPES = ["memoryless_agent", "memory_agent"]
EVAL_MODES = ["training_distribution_eval", "decorrelated_eval"]
SEEDS = list(range(30))
EPISODES = 4000
EVAL_EPISODES = 500
HORIZON = 14
TARGET_PHASE = 10
ACTION_CAUSAL = 0
ACTION_PROXY = 1
NEUTRAL_ACTIONS = [2, 3]
ACTION_COUNT = 4
ALPHA = 0.08
GAMMA = 0.95
LAMBDA = 0.80
EPSILON_START = 1.0
EPSILON_END = 0.04
STEP_PENALTY = -0.002
HIDDEN_REWARD_PROB = 0.5


class HiddenStateEnv:
    def __init__(self, p_proxy_given_reward, p_proxy_given_no_reward, mode, seed=0):
        self.p_proxy_given_reward = p_proxy_given_reward
        self.p_proxy_given_no_reward = p_proxy_given_no_reward
        self.mode = mode
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.phase = 0
        self.done = False
        self.hidden_reward_state = int(self.rng.random() < HIDDEN_REWARD_PROB)
        self.reward_obtained = False
        self.causal_selected_at_target = False
        self.proxy_by_phase = [self._sample_proxy() for _ in range(HORIZON + 1)]
        return self.observe()

    def _sample_proxy(self):
        if self.mode == "decorrelated_eval":
            return int(self.rng.random() < 0.5)
        prob = self.p_proxy_given_reward if self.hidden_reward_state else self.p_proxy_given_no_reward
        return int(self.rng.random() < prob)

    def observe(self):
        return self.phase * 2 + self.proxy_by_phase[self.phase]

    def step(self, action):
        if self.done:
            raise RuntimeError("step called after episode termination")
        reward = STEP_PENALTY
        if self.phase == TARGET_PHASE:
            if action == ACTION_CAUSAL:
                self.causal_selected_at_target = True
                if self.hidden_reward_state == 1:
                    reward += 1.0
                    self.reward_obtained = True
            self.done = True
        else:
            self.phase += 1
            self.done = self.phase >= HORIZON
        return self.observe(), reward, self.done, {}


class SarsaLambdaAgent:
    def __init__(self, agent_type, seed):
        self.agent_type = agent_type
        self.obs_count = (HORIZON + 1) * 2
        self.state_count = self.obs_count if agent_type == "memoryless_agent" else self.obs_count * self.obs_count
        self.rng = random.Random(seed)
        self.q = [[0.0 for _ in range(ACTION_COUNT)] for _ in range(self.state_count)]

    def encode(self, prev_obs, obs):
        if self.agent_type == "memoryless_agent":
            return obs
        return prev_obs * self.obs_count + obs

    def act(self, state, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(ACTION_COUNT)
        return self.greedy_action(state)

    def greedy_action(self, state):
        values = self.q[state]
        best = max(values)
        best_actions = [idx for idx, value in enumerate(values) if abs(value - best) < 1e-12]
        return self.rng.choice(best_actions)

    def train_episode(self, env, epsilon):
        traces = {}
        obs = env.reset()
        prev_obs = obs
        state = self.encode(prev_obs, obs)
        action = self.act(state, epsilon)
        done = False
        while not done:
            next_obs, reward, done, _ = env.step(action)
            next_state = self.encode(obs, next_obs)
            if done:
                target = reward
                next_action = None
            else:
                next_action = self.act(next_state, epsilon)
                target = reward + GAMMA * self.q[next_state][next_action]
            delta = target - self.q[state][action]
            traces[(state, action)] = traces.get((state, action), 0.0) + 1.0
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
            prev_obs, obs = obs, next_obs
            state = next_state
            action = next_action if next_action is not None else ACTION_CAUSAL


def run_seed(condition, p1, p0, agent_type, seed):
    train_env = HiddenStateEnv(p1, p0, "training_distribution_eval", seed)
    agent = SarsaLambdaAgent(agent_type, seed)
    epsilon = EPSILON_START
    epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / EPISODES)
    for _ in range(EPISODES):
        agent.train_episode(train_env, epsilon)
        epsilon = max(EPSILON_END, epsilon * epsilon_decay)
    rows = []
    for mode in EVAL_MODES:
        metrics = evaluate(agent, p1, p0, mode, seed)
        rows.append(
            {
                "condition": condition,
                "proxy_strength": round(p1 - p0, 6),
                "p_proxy_given_reward": p1,
                "p_proxy_given_no_reward": p0,
                "eval_mode": mode,
                "agent_type": agent_type,
                "seed": seed,
                **metrics,
            }
        )
    baseline = evaluate_random_policy(condition, p1, p0, agent_type, seed)
    return rows, baseline


def evaluate(agent, p1, p0, mode, seed):
    env = HiddenStateEnv(p1, p0, mode, seed + 60000)
    totals = init_counts()
    q_proxy, q_neutral, q_causal, cds_values = [], [], [], []
    cue_counts = {0: init_counts(), 1: init_counts()}
    for _ in range(EVAL_EPISODES):
        obs = env.reset()
        prev_obs = obs
        done = False
        while not done:
            state = agent.encode(prev_obs, obs)
            cue = obs % 2
            action = agent.greedy_action(state)
            if env.phase < TARGET_PHASE:
                record_action(totals, action)
                record_action(cue_counts[cue], action)
                values = agent.q[state]
                masses = softmax(values)
                cds_values.append(masses[ACTION_PROXY] + sum(masses[a] for a in NEUTRAL_ACTIONS))
                q_proxy.append(values[ACTION_PROXY])
                q_neutral.append(mean(values[a] for a in NEUTRAL_ACTIONS))
                q_causal.append(values[ACTION_CAUSAL])
            next_obs, reward, done, _ = env.step(action)
            totals["reward_sum"] += reward
            prev_obs, obs = obs, next_obs
        totals["goal_sum"] += int(env.reward_obtained)
    proxy_q = mean(q_proxy)
    neutral_q = mean(q_neutral)
    causal_q = mean(q_causal)
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
        "Q_causal": round(causal_q, 6),
        "Q_proxy": round(proxy_q, 6),
        "Q_neutral_mean": round(neutral_q, 6),
        "proxy_Q_advantage": round(proxy_q - neutral_q, 6),
        "proxy_vs_causal_Q": round(proxy_q - causal_q, 6),
        "proxy_dependence": round(p_causal_cue1 - p_causal_cue0, 6),
        "proxy_action_dependence": round(p_proxy_cue1 - p_proxy_cue0, 6),
    }


def evaluate_random_policy(condition, p1, p0, agent_type, seed):
    rng = random.Random(seed + 90000)
    env = HiddenStateEnv(p1, p0, "decorrelated_eval", seed + 70000)
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
    return {
        "condition": condition,
        "proxy_strength": round(p1 - p0, 6),
        "agent_type": agent_type,
        "seed": seed,
        "random_reward": round(totals["reward_sum"] / EVAL_EPISODES, 6),
        "random_goal_rate": round(totals["goal_sum"] / EVAL_EPISODES, 6),
        "random_proxy_action_rate": round(rate(totals, "proxy"), 6),
        "random_causal_action_rate": round(rate(totals, "causal"), 6),
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


def softmax(values):
    max_q = max(values)
    exp_values = [math.exp(v - max_q) for v in values]
    denom = sum(exp_values)
    return [v / denom for v in exp_values]


def summarize(rows):
    metrics = [
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
    output = []
    for condition, p1, p0 in CONDITIONS:
        for agent_type in AGENT_TYPES:
            for mode in EVAL_MODES:
                subset = [r for r in rows if r["condition"] == condition and r["agent_type"] == agent_type and r["eval_mode"] == mode]
                item = {
                    "condition": condition,
                    "proxy_strength": round(p1 - p0, 6),
                    "agent_type": agent_type,
                    "eval_mode": mode,
                    "n": len(subset),
                }
                for metric in metrics:
                    values = [float(r[metric]) for r in subset]
                    item[f"{metric}_mean"] = round(mean(values), 6)
                    item[f"{metric}_ci95"] = round(1.96 * sem(values), 6)
                output.append(item)
    return output


def trend_analysis(rows):
    output = []
    for agent_type in AGENT_TYPES:
        for mode in EVAL_MODES:
            subset = [r for r in rows if r["agent_type"] == agent_type and r["eval_mode"] == mode]
            for metric in ["proxy_action_rate", "proxy_Q_advantage", "proxy_action_dependence", "SPI", "reward", "goal_rate", "proxy_dependence"]:
                fit = linear_regression(
                    [float(r["proxy_strength"]) for r in subset],
                    [float(r[metric]) for r in subset],
                )
                output.append(
                    {
                        "agent_type": agent_type,
                        "eval_mode": mode,
                        "metric": metric,
                        "slope": round(fit["slope"], 6),
                        "slope_ci95": round(1.96 * fit["slope_se"], 6),
                        "standardized_effect": round(fit["standardized_effect"], 6),
                        "t_stat": round(fit["t_stat"], 6),
                        "p_value_normal": round(fit["p_value"], 6),
                    }
                )
    return output


def pairwise(rows):
    output = []
    for agent_type in AGENT_TYPES:
        for mode in EVAL_MODES:
            low = [r for r in rows if r["condition"] == "no_proxy" and r["agent_type"] == agent_type and r["eval_mode"] == mode]
            high = [r for r in rows if r["condition"] == "strong_proxy" and r["agent_type"] == agent_type and r["eval_mode"] == mode]
            for metric in ["proxy_action_rate", "proxy_Q_advantage", "proxy_action_dependence", "SPI", "reward", "goal_rate", "proxy_dependence"]:
                a = [float(r[metric]) for r in low]
                b = [float(r[metric]) for r in high]
                diff = mean(b) - mean(a)
                ci = 1.96 * math.sqrt(sem(a) ** 2 + sem(b) ** 2)
                se = ci / 1.96 if ci else 0.0
                t = diff / se if se else 0.0
                p = math.erfc(abs(t) / math.sqrt(2.0))
                output.append(
                    {
                        "agent_type": agent_type,
                        "eval_mode": mode,
                        "metric": metric,
                        "strong_minus_none": round(diff, 6),
                        "ci95": round(ci, 6),
                        "t_stat": round(t, 6),
                        "p_value_normal": round(p, 6),
                    }
                )
    return output


def baseline_summary(baselines, rows):
    output = []
    for condition, p1, p0 in CONDITIONS:
        for agent_type in AGENT_TYPES:
            b = [r for r in baselines if r["condition"] == condition and r["agent_type"] == agent_type]
            learned = [r for r in rows if r["condition"] == condition and r["agent_type"] == agent_type and r["eval_mode"] == "decorrelated_eval"]
            for metric, learned_key, random_key in [
                ("reward", "reward", "random_reward"),
                ("goal_rate", "goal_rate", "random_goal_rate"),
                ("proxy_action_rate", "proxy_action_rate", "random_proxy_action_rate"),
                ("causal_action_rate", "causal_action_rate", "random_causal_action_rate"),
            ]:
                lv = [float(r[learned_key]) for r in learned]
                rv = [float(r[random_key]) for r in b]
                diff = mean(lv) - mean(rv)
                output.append(
                    {
                        "condition": condition,
                        "agent_type": agent_type,
                        "metric": metric,
                        "learned_mean": round(mean(lv), 6),
                        "random_mean": round(mean(rv), 6),
                        "learned_minus_random": round(diff, 6),
                    }
                )
    return output


def linear_regression(x, y):
    xm, ym = mean(x), mean(y)
    sxx = sum((v - xm) ** 2 for v in x)
    sxy = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    slope = sxy / sxx if sxx else 0.0
    intercept = ym - slope * xm
    residuals = [b - (intercept + slope * a) for a, b in zip(x, y)]
    df = max(1, len(y) - 2)
    residual_var = sum(r * r for r in residuals) / df
    se = math.sqrt(residual_var / sxx) if sxx else 0.0
    t = slope / se if se else 0.0
    p = math.erfc(abs(t) / math.sqrt(2.0))
    xsd = statistics.stdev(x) if len(x) > 1 else 0.0
    ysd = statistics.stdev(y) if len(y) > 1 else 0.0
    return {"slope": slope, "slope_se": se, "t_stat": t, "p_value": p, "standardized_effect": slope * xsd / ysd if ysd else 0.0}


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
    path = os.path.join(RESULTS_DIR, "hidden_state_superstition_plot.png")
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
        ("proxy_action_rate_mean", 70, 70),
        ("proxy_Q_advantage_mean", 560, 70),
        ("proxy_action_dependence_mean", 70, 380),
        ("goal_rate_mean", 560, 380),
    ]
    colors = {"memoryless_agent": (36, 92, 156), "memory_agent": (214, 107, 42)}
    strengths = [0.0, 0.30, 0.60, 0.90]
    rows = [r for r in summary if r["eval_mode"] == "decorrelated_eval"]
    for metric, ox, oy in panels:
        panel_w, panel_h = 350, 200
        values = [r[metric] for r in rows]
        ymin = min(values + [0.0])
        ymax = max(values + [1.0])
        if ymax <= ymin:
            ymax = ymin + 1.0
        pad = (ymax - ymin) * 0.1
        ymin -= pad
        ymax += pad
        line(ox, oy + panel_h, ox + panel_w, oy + panel_h, (0, 0, 0))
        line(ox, oy, ox, oy + panel_h, (0, 0, 0))
        for agent_type in AGENT_TYPES:
            series = [r for r in rows if r["agent_type"] == agent_type]
            pts = []
            for r in series:
                x = r["proxy_strength"]
                y = r[metric]
                px = ox + int((x - min(strengths)) / (max(strengths) - min(strengths)) * panel_w)
                py = oy + int((ymax - y) / (ymax - ymin) * panel_h)
                pts.append((px, py))
                dot(px, py, colors[agent_type])
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


def write_report(summary, trends, pairs, baseline):
    decor = [r for r in summary if r["eval_mode"] == "decorrelated_eval"]
    trend_map = {(r["agent_type"], r["eval_mode"], r["metric"]): r for r in trends}
    mem_strong = next(r for r in decor if r["agent_type"] == "memory_agent" and r["condition"] == "strong_proxy")
    mem_base = next(r for r in baseline if r["agent_type"] == "memory_agent" and r["condition"] == "strong_proxy" and r["metric"] == "reward")
    action_superstition = (
        trend_map[("memory_agent", "decorrelated_eval", "proxy_action_rate")]["slope"] > 0
        and trend_map[("memory_agent", "decorrelated_eval", "proxy_Q_advantage")]["slope"] > 0
        and trend_map[("memory_agent", "decorrelated_eval", "proxy_action_dependence")]["slope"] > 0
        and mem_base["learned_minus_random"] > 0
    )
    proxy_reliance = (
        trend_map[("memory_agent", "decorrelated_eval", "proxy_dependence")]["slope"] > 0
        and not action_superstition
    )
    lines = [
        "# Hidden-State Superstition Experiment",
        "",
        "## Research question",
        "Does superstition emerge when the true reward-generating state is hidden and the agent must rely on a non-causal proxy cue correlated with reward during training?",
        "",
        "## Why partial observability is the next theoretical step",
        "Prior experiments ruled out simple distractor count, temporal proximity, spurious correlation alone, and causal ambiguity alone. Hidden causal structure is a stronger test because the agent cannot directly observe the reward-generating state.",
        "",
        "## Environment design",
        f"POMDP-style tabular environment. Hidden reward state is sampled each episode. Reward can occur at phase {TARGET_PHASE} only when hidden_reward_state=1 and action 0 is selected.",
        "",
        "## Hidden causal structure",
        "The agent observes phase and proxy cue, but not hidden_reward_state. Action 1 is a proxy/spurious action and never directly causes reward. Neutral actions never cause reward.",
        "",
        "## Proxy-cue manipulation",
        "Proxy strength varies from no_proxy to strong_proxy by changing P(proxy=1|hidden=1) and P(proxy=1|hidden=0). Decorrelated evaluation sets both probabilities to 0.50.",
        "",
        "## Agent design",
        f"Tabular SARSA(lambda), alpha={ALPHA}, gamma={GAMMA}, lambda={LAMBDA}, epsilon {EPSILON_START}->{EPSILON_END}, {EPISODES} training episodes, {EVAL_EPISODES} evaluation episodes, 30 seeds. Both memoryless and previous-observation memory agents were run.",
        "",
        "## Metrics",
        "Metrics include reward, goal_rate, SPI, CDS, action rates, Q advantages, proxy_dependence for causal action selection, and proxy_action_dependence for proxy action selection.",
        "",
        "## Results tables",
        "",
        "| condition | agent | mode | reward | goal_rate | proxy_rate | Q_adv | proxy_dep | proxy_action_dep |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        if r["eval_mode"] == "decorrelated_eval":
            lines.append(
                f"| {r['condition']} | {r['agent_type']} | decorrelated | "
                f"{r['reward_mean']:.3f} +/- {r['reward_ci95']:.3f} | "
                f"{r['goal_rate_mean']:.3f} +/- {r['goal_rate_ci95']:.3f} | "
                f"{r['proxy_action_rate_mean']:.3f} +/- {r['proxy_action_rate_ci95']:.3f} | "
                f"{r['proxy_Q_advantage_mean']:.3f} +/- {r['proxy_Q_advantage_ci95']:.3f} | "
                f"{r['proxy_dependence_mean']:.3f} +/- {r['proxy_dependence_ci95']:.3f} | "
                f"{r['proxy_action_dependence_mean']:.3f} +/- {r['proxy_action_dependence_ci95']:.3f} |"
            )
    lines.extend(["", "## Trend analysis", "", "| agent | mode | metric | slope | 95% CI | p approx |", "|---|---|---|---:|---:|---:|"])
    for r in trends:
        lines.append(f"| {r['agent_type']} | {r['eval_mode']} | {r['metric']} | {r['slope']:.4f} | +/- {r['slope_ci95']:.4f} | {r['p_value_normal']:.4f} |")
    lines.extend(["", "## Strong-vs-none comparison", "", "| agent | mode | metric | strong - none | 95% CI | p approx |", "|---|---|---|---:|---:|---:|"])
    for r in pairs:
        lines.append(f"| {r['agent_type']} | {r['eval_mode']} | {r['metric']} | {r['strong_minus_none']:.4f} | +/- {r['ci95']:.4f} | {r['p_value_normal']:.4f} |")
    lines.extend(["", "## Random baseline comparison", "", "| condition | agent | metric | learned | random | learned-random |", "|---|---|---|---:|---:|---:|"])
    for r in baseline:
        lines.append(f"| {r['condition']} | {r['agent_type']} | {r['metric']} | {r['learned_mean']:.3f} | {r['random_mean']:.3f} | {r['learned_minus_random']:.3f} |")
    lines.extend(
        [
            "",
            "## Training vs decorrelated evaluation",
            "Both evaluation modes are reported separately. Decorrelated evaluation is the primary persistence test because the proxy cue no longer predicts hidden_reward_state.",
            "",
            "## Memoryless vs memory-agent comparison",
            "Both agents use the same learning algorithm. The memory agent observes previous+current observation, allowing short-history dependence.",
            "",
            "## Interpretation",
            f"Memory-agent strong_proxy decorrelated reward={mem_strong['reward_mean']:.3f}; goal_rate={mem_strong['goal_rate_mean']:.3f}; proxy_action_rate={mem_strong['proxy_action_rate_mean']:.3f}; proxy_Q_advantage={mem_strong['proxy_Q_advantage_mean']:.3f}.",
            "",
            "## Does this support computational superstition?",
            f"Action-level superstition supported: {action_superstition}.",
            f"Proxy reliance without proxy-action superstition: {proxy_reliance}.",
            "",
            "## Limitations",
            "The proxy cue is observational, and tabular agents may learn to use the causal action under cue states rather than selecting the proxy action itself. If reward drops to random, the task is too hard; if proxy dependence rises without proxy action rate, the result is proxy reliance rather than full action superstition.",
            "",
            "## Recommended next experiment",
            "Use a recurrent or belief-state agent and add an extinction/reversal phase where the proxy cue becomes anti-predictive, then measure persistence and extinction speed.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "hidden_state_superstition_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows, baselines = [], []
    for condition, p1, p0 in CONDITIONS:
        for agent_type in AGENT_TYPES:
            for seed in SEEDS:
                print(f"condition={condition} agent={agent_type} seed={seed}", flush=True)
                new_rows, baseline = run_seed(condition, p1, p0, agent_type, seed)
                rows.extend(new_rows)
                baselines.append(baseline)
    summary = summarize(rows)
    trends = trend_analysis(rows)
    pairs = pairwise(rows)
    baseline = baseline_summary(baselines, rows)
    result_fields = [
        "condition",
        "proxy_strength",
        "p_proxy_given_reward",
        "p_proxy_given_no_reward",
        "eval_mode",
        "agent_type",
        "seed",
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
    write_csv(os.path.join(RESULTS_DIR, "hidden_state_superstition_results.csv"), rows, result_fields)
    summary_fields = ["condition", "proxy_strength", "agent_type", "eval_mode", "n"]
    for metric in result_fields[7:]:
        summary_fields.extend([f"{metric}_mean", f"{metric}_ci95"])
    write_csv(os.path.join(RESULTS_DIR, "hidden_state_superstition_summary.csv"), summary, summary_fields)
    write_csv(
        os.path.join(RESULTS_DIR, "hidden_state_superstition_trends.csv"),
        trends,
        ["agent_type", "eval_mode", "metric", "slope", "slope_ci95", "standardized_effect", "t_stat", "p_value_normal"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "hidden_state_superstition_pairwise.csv"),
        pairs,
        ["agent_type", "eval_mode", "metric", "strong_minus_none", "ci95", "t_stat", "p_value_normal"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "hidden_state_superstition_baseline.csv"),
        baseline,
        ["condition", "agent_type", "metric", "learned_mean", "random_mean", "learned_minus_random"],
    )
    write_plot(summary)
    write_report(summary, trends, pairs, baseline)


if __name__ == "__main__":
    run()
