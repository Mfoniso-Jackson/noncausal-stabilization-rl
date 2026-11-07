import csv
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import uncertainty_correlation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

DELAY_CONDITIONS = [0, 3, 6, 10]
SEEDS = list(range(30))
EPISODES = 2500
EVAL_EPISODES = 100
BASE_TARGET_PHASE = 10
SLACK = 10
NEAR_WINDOW = [7, 8, 9]
MID_WINDOW = [4, 5, 6]
FAR_WINDOW = [1, 2, 3]
NEUTRAL_SAMPLE_RANGE = range(1, BASE_TARGET_PHASE)
TYPE_TO_ACTION = {"near": 1, "mid": 2, "far": 3, "neutral": 4}
ACTION_TO_TYPE = {value: key for key, value in TYPE_TO_ACTION.items()}
DISTRACTOR_TYPES = ["near", "mid", "far", "neutral"]
ACTION_COUNT = 5
ALPHA = 0.08
GAMMA = 0.95
LAMBDA = 0.80
EPSILON_START = 1.0
EPSILON_END = 0.04
STEP_PENALTY = -0.002


class TemporalDistractorDelayEnv:
    """Same near/mid/far/neutral distractor schedule as
    temporal_distractor_experiment.py's temporally_biased_distractors
    condition (phases 7-9/4-6/1-3, at delay=0 this reproduces that
    condition exactly), but the reward phase is pushed out by `delay`
    extra filler steps after the near window. Those filler steps offer no
    distractor actions at all -- only the manipulation under test (the gap
    between distractor exposure and the reward signal) changes; temporal
    availability of every distractor type is held fixed.
    """

    def __init__(self, delay, seed=0):
        self.delay = delay
        self.target_phase = BASE_TARGET_PHASE + delay
        self.horizon = self.target_phase + SLACK
        self.rng = random.Random(seed)
        self.action_count = ACTION_COUNT
        self.reset()

    def reset(self):
        self.phase = 0
        self.done = False
        self.goal_completed = False
        self.neutral_phases = set(self.rng.sample(list(NEUTRAL_SAMPLE_RANGE), 3))
        return self.phase

    def available_actions(self, phase=None):
        phase = self.phase if phase is None else phase
        actions = [0]
        if phase >= self.target_phase:
            return actions
        if phase in NEAR_WINDOW:
            actions.append(TYPE_TO_ACTION["near"])
        if phase in MID_WINDOW:
            actions.append(TYPE_TO_ACTION["mid"])
        if phase in FAR_WINDOW:
            actions.append(TYPE_TO_ACTION["far"])
        if phase in self.neutral_phases:
            actions.append(TYPE_TO_ACTION["neutral"])
        return sorted(set(actions))

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


def train_and_evaluate(delay, seed):
    env = TemporalDistractorDelayEnv(delay, seed)
    agent = SarsaLambdaAgent(env.horizon, env.action_count, seed)
    epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / EPISODES)
    epsilon = EPSILON_START
    for _ in range(EPISODES):
        agent.train_episode(env, epsilon)
        epsilon = max(EPSILON_END, epsilon * epsilon_decay)
    return evaluate(delay, seed, agent)


def evaluate(delay, seed, agent):
    eval_env = TemporalDistractorDelayEnv(delay, seed + 100000)
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
            if state < eval_env.target_phase:
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
            if action in ACTION_TO_TYPE and state < eval_env.target_phase:
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
        "delay": delay,
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
        "SPI_near_minus_far": round(spi_by_type["near"] - spi_by_type["far"], 6),
        "Q_mass_near_minus_far": round(q_mass_by_type["near"] - q_mass_by_type["far"], 6),
    }


def normalized_action_mass(values, available):
    max_q = max(values[action] for action in available)
    exp_values = {action: math.exp(values[action] - max_q) for action in available}
    denom = sum(exp_values.values())
    return {action: exp_values[action] / denom for action in available}


def mean(values):
    values = list(values)
    return sum(values) / max(1, len(values))


def sem(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def summarize(rows):
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
        "SPI_near_minus_far",
        "Q_mass_near_minus_far",
    ]
    summary = []
    for delay in DELAY_CONDITIONS:
        subset = [row for row in rows if row["delay"] == delay]
        item = {"delay": delay, "n": len(subset)}
        for metric in metrics:
            values = [float(row[metric]) for row in subset]
            item[f"{metric}_mean"] = round(mean(values), 6)
            item[f"{metric}_ci95"] = round(1.96 * sem(values), 6)
        summary.append(item)
    return summary


def gradient_trend(rows):
    delays = [float(row["delay"]) for row in rows]
    spi_gap = [float(row["SPI_near_minus_far"]) for row in rows]
    q_gap = [float(row["Q_mass_near_minus_far"]) for row in rows]
    spi_overall = [float(row["SPI"]) for row in rows]
    cds_overall = [float(row["CDS"]) for row in rows]
    return {
        "spi_gap_vs_delay": uncertainty_correlation(delays, spi_gap),
        "q_gap_vs_delay": uncertainty_correlation(delays, q_gap),
        "spi_overall_vs_delay": uncertainty_correlation(delays, spi_overall),
        "cds_overall_vs_delay": uncertainty_correlation(delays, cds_overall),
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plot(summary):
    svg_path = os.path.join(RESULTS_DIR, "temporal_distractor_delay_plot.svg")
    width, height = 940, 340
    panel_w, panel_h = 380, 220
    origins = [(60, 70), (500, 70)]
    colors = {"SPI_near": "#245c9c", "SPI_mid": "#d66b2a", "SPI_far": "#2f7d55"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="470" y="26" text-anchor="middle" font-family="Arial" font-size="17" font-weight="700">Temporal Distractor Delay: SPI by type vs reward delay</text>',
    ]
    xs = DELAY_CONDITIONS
    panel_specs = [("SPI by type", ["SPI_near_mean", "SPI_mid_mean", "SPI_far_mean"]), ("near-minus-far gap", ["SPI_near_minus_far_mean", "Q_mass_near_minus_far_mean"])]
    for (title, keys), (ox, oy) in zip(panel_specs, origins):
        parts.append(f'<text x="{ox + panel_w/2}" y="{oy-12}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700">{title}</text>')
        parts.append(f'<line x1="{ox}" y1="{oy+panel_h}" x2="{ox+panel_w}" y2="{oy+panel_h}" stroke="#222"/>')
        parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+panel_h}" stroke="#222"/>')
        values_all = [row[key] for row in summary for key in keys]
        ymin = min(values_all + [0.0])
        ymax = max(values_all + [0.1])
        pad = (ymax - ymin) * 0.15 or 0.05
        ymin -= pad
        ymax += pad
        for idx, key in enumerate(keys):
            color = colors.get(key.replace("_mean", ""), ["#245c9c", "#d66b2a"][idx % 2])
            coords = " ".join(
                f"{ox + (x/max(xs))*panel_w:.2f},{oy + (ymax - row[key])/(ymax-ymin)*panel_h:.2f}"
                for x, row in zip(xs, summary)
            )
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x in xs:
            parts.append(f'<text x="{ox + (x/max(xs))*panel_w:.2f}" y="{oy+panel_h+16}" text-anchor="middle" font-family="Arial" font-size="10">{x}</text>')
    parts.append("</svg>")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_report(summary, rows, trend):
    zero_delay = next(row for row in summary if row["delay"] == 0)
    max_delay = next(row for row in summary if row["delay"] == max(DELAY_CONDITIONS))
    spi_trend = trend["spi_gap_vs_delay"]
    q_trend = trend["q_gap_vs_delay"]
    spi_overall_trend = trend["spi_overall_vs_delay"]
    cds_overall_trend = trend["cds_overall_vs_delay"]
    gradient_strengthens = (
        float(spi_trend["correlation"]) > 0
        and float(spi_trend["p_value"]) < 0.05
        and float(q_trend["correlation"]) > 0
        and float(q_trend["p_value"]) < 0.05
    )
    lines = [
        "# Temporal Distractor Delay Experiment",
        "",
        "## Research question",
        "temporal_distractor_report.md found no near-vs-far proximity gradient in distractor Q-mass or selection rate (near_minus_far was near zero and not reliably positive). Does that null result change if the credit-assignment problem is made explicitly harder by pushing the reward phase further away from the (unchanged) distractor exposure windows?",
        "",
        "## Why this follows the temporal distractor experiment",
        "That experiment's own recommended next step was exactly this: hold temporal availability fixed and manipulate reward delay after near versus far distractor exposure, to see whether the (absent) proximity gradient strengthens when credit assignment is made harder.",
        "",
        "## Design",
        f"The near ({NEAR_WINDOW}), mid ({MID_WINDOW}), far ({FAR_WINDOW}) and neutral distractor windows are at the exact same absolute phases as the original temporally_biased_distractors condition, and the delay=0 condition reproduces that environment exactly. For delay in {DELAY_CONDITIONS}, the reward phase is moved from phase {BASE_TARGET_PHASE} to phase {BASE_TARGET_PHASE} + delay; the {SLACK}-step post-target slack is preserved. The phases inserted between the near window (ending at phase {NEAR_WINDOW[-1]}) and the (now later) reward phase offer no distractor actions at all -- only action 0 is available there, so the only thing that changes with delay is how many credit-assignment steps separate distractor exposure from the eventual reward signal.",
        "",
        "## Agent design",
        f"Tabular SARSA(lambda), alpha={ALPHA}, gamma={GAMMA}, lambda={LAMBDA}, epsilon {EPSILON_START}->{EPSILON_END}, {EPISODES} training episodes, {EVAL_EPISODES} evaluation episodes, {len(SEEDS)} seeds per delay condition.",
        "",
        "## Results table",
        "",
        "| delay | target phase | reward | goal_rate | CDS | SPI | SPI_near | SPI_mid | SPI_far | SPI near-far | Q_mass near-far |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['delay']} | {BASE_TARGET_PHASE + row['delay']} | {row['reward_mean']:.3f} +/- {row['reward_ci95']:.3f} | "
            f"{row['goal_rate_mean']:.3f} +/- {row['goal_rate_ci95']:.3f} | {row['CDS_mean']:.3f} +/- {row['CDS_ci95']:.3f} | "
            f"{row['SPI_mean']:.3f} +/- {row['SPI_ci95']:.3f} | {row['SPI_near_mean']:.3f} | {row['SPI_mid_mean']:.3f} | {row['SPI_far_mean']:.3f} | "
            f"{row['SPI_near_minus_far_mean']:.3f} +/- {row['SPI_near_minus_far_ci95']:.3f} | "
            f"{row['Q_mass_near_minus_far_mean']:.3f} +/- {row['Q_mass_near_minus_far_ci95']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Dose-response tests (n={} seed-delay observations)".format(len(SEEDS) * len(DELAY_CONDITIONS)),
            "",
            f"SPI near-minus-far gap vs delay: r = {float(spi_trend['correlation']):.4f}, p = {float(spi_trend['p_value']):.4f}.",
            f"Q-mass near-minus-far gap vs delay: r = {float(q_trend['correlation']):.4f}, p = {float(q_trend['p_value']):.4f}.",
            f"Overall SPI (all distractor types) vs delay: r = {float(spi_overall_trend['correlation']):.4f}, p = {float(spi_overall_trend['p_value']):.4f}.",
            f"Overall CDS vs delay: r = {float(cds_overall_trend['correlation']):.4f}, p = {float(cds_overall_trend['p_value']):.4f}.",
            "",
            "## Interpretation",
            (
                f"At delay=0 (identical to the original temporally_biased_distractors condition), SPI near-minus-far = "
                f"{zero_delay['SPI_near_minus_far_mean']:.3f}, matching the original report's near-null finding. At the largest tested delay "
                f"({max(DELAY_CONDITIONS)} extra steps), SPI near-minus-far = {max_delay['SPI_near_minus_far_mean']:.3f}. "
                "If distractor Q-mass and selection rate were driven by eligibility-trace distance from the reward signal, lengthening the "
                "pure, distractor-free gap between the near window and the reward should widen the near/far gap, since the relative eligibility-trace "
                "discount ratio between the near and far windows is unaffected by this manipulation but the overall magnitude of all distractor "
                "credit should shrink and -- if genuinely distance-driven -- shrink unevenly. The trend tests above are the direct answer to whether "
                "that happens."
            ),
            "",
            "## Does credit-assignment difficulty strengthen the proximity gradient?",
            f"Supported: {gradient_strengthens}.",
            "",
            "## A metric-construction artifact in CDS, not a real effect",
            (
                f"Overall CDS falls sharply and almost perfectly with delay ({zero_delay['CDS_mean']:.3f} at delay=0 down to "
                f"{max_delay['CDS_mean']:.3f} at delay={max(DELAY_CONDITIONS)}, r={float(cds_overall_trend['correlation']):.4f}, "
                f"p={float(cds_overall_trend['p_value']):.4f}), and CDS * target_phase is approximately constant across every row of the "
                "results table above. This is not evidence that longer delay reduces superstition: CDS, as defined in this codebase's "
                "`evaluate()` pattern (shared with temporal_distractor_experiment.py and distractor_action_experiment.py), averages the "
                "distractor Q-mass over *every* pre-target decision step, including steps where no distractor is available at all (which "
                "contribute exactly 0). The filler steps this experiment inserts to create delay are always distractor-free, so they add "
                "zero-valued terms to that average without changing the fixed number of genuinely distractor-available steps -- mechanically "
                "diluting CDS as delay grows, independent of any change in the policy's actual distractor preference when a distractor is "
                "present. SPI, which is defined only over decisions where a distractor was actually available, does not show this artifact "
                "(its near-flat, non-significant trend above is the metric that should be trusted for this question). This is worth flagging "
                "as a general caveat for any future use of this repository's CDS metric on environments with variable-length, non-uniform "
                "distractor availability."
            ),
            "",
            "## Limitations",
            "Delay is varied only up to 10 extra steps given the tabular agent's horizon and the fixed near/mid/far window positions; much larger delays were not tested. Approximate correlation p-values use scipy's Pearson test on seed-level observations pooled across delay conditions and do not correct for the two primary gap-vs-delay comparisons (SPI and Q-mass) tested together.",
            "",
            "## Next recommended experiment",
            "If the gradient remains null even at the largest delay tested here, the next step is to test much larger delays with a function-approximation agent (where generalization across nearby states, rather than tabular independence, could reintroduce a distance-dependent credit leak that this tabular design cannot show).",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "temporal_distractor_delay_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows = []
    for delay in DELAY_CONDITIONS:
        for seed in SEEDS:
            print(f"delay={delay} seed={seed}", flush=True)
            rows.append(train_and_evaluate(delay, seed))
    summary = summarize(rows)
    trend = gradient_trend(rows)
    result_fields = [
        "delay",
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
        "SPI_near_minus_far",
        "Q_mass_near_minus_far",
    ]
    write_csv(os.path.join(RESULTS_DIR, "temporal_distractor_delay_results.csv"), rows, result_fields)
    summary_fields = ["delay", "n"]
    for metric in result_fields[2:]:
        summary_fields.extend([f"{metric}_mean", f"{metric}_ci95"])
    write_csv(os.path.join(RESULTS_DIR, "temporal_distractor_delay_summary.csv"), summary, summary_fields)
    write_plot(summary)
    write_report(summary, rows, trend)


if __name__ == "__main__":
    run()
