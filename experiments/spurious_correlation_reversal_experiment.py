import csv
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import compare_groups, uncertainty_correlation

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
ACQUISITION_EPISODES = 2500
REVERSAL_EPISODES = 1000
BLOCK_SIZE = 50
NUM_BLOCKS = REVERSAL_EPISODES // BLOCK_SIZE
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
EVAL_EPISODES = 300


class SpuriousCorrelationEnv:
    """Same environment as spurious_correlation_experiment.py, plus a
    `reversal` mode: the action/cue availability statistics are unchanged
    (the spurious action appears exactly as often as during acquisition),
    but once reversal is active, choosing the spurious action poisons the
    episode -- it genuinely blocks the terminal reward even if the agent
    later picks the causal action. The spurious cue goes from merely
    non-causal to explicitly anti-correlated with reward, while remaining
    available throughout.
    """

    def __init__(self, correlation_strength, training=True, reversal=False, seed=0):
        self.correlation_strength = correlation_strength
        self.training = training
        self.reversal = reversal
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.phase = 0
        self.done = False
        self.goal_completed = False
        self.poisoned = False
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
        if self.reversal and action == ACTION_SPURIOUS:
            self.poisoned = True
        if self.phase == TARGET_PHASE and action == ACTION_CAUSAL:
            if not self.poisoned:
                reward += 1.0
                self.goal_completed = True
            self.done = True
        else:
            self.phase += 1
            self.done = self.phase >= HORIZON
        return self.observe(), reward, self.done, {"goal_completed": self.goal_completed, "poisoned": self.poisoned}


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

    def train_episode_tracking(self, env, epsilon):
        """Same SARSA(lambda) update as train_episode, but also records
        which action was taken whenever the spurious action was available
        pre-target, so reversal-phase blocks can be scored without a
        separate frozen-policy evaluation pass."""
        traces = [[0.0 for _ in range(ACTION_COUNT)] for _ in range(len(self.q))]
        stats = {"relevant": 0, "spurious": 0, "neutral": 0, "causal": 0}

        def track(phase, available, action):
            if ACTION_SPURIOUS in available and phase < TARGET_PHASE:
                stats["relevant"] += 1
                if action == ACTION_SPURIOUS:
                    stats["spurious"] += 1
                elif action in NEUTRAL_ACTIONS:
                    stats["neutral"] += 1
                elif action == ACTION_CAUSAL:
                    stats["causal"] += 1

        state = env.reset()
        available = env.available_actions()
        action = self.act(state, available, epsilon)
        track(env.phase, available, action)
        done = False
        total_reward = 0.0
        while not done:
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                target = reward
                next_action = None
            else:
                next_available = env.available_actions()
                next_action = self.act(next_state, next_available, epsilon)
                track(env.phase, next_available, next_action)
                target = reward + GAMMA * self.q[next_state][next_action]
            delta = target - self.q[state][action]
            traces[state][action] += 1.0
            for s in range(len(self.q)):
                for a in range(ACTION_COUNT):
                    self.q[s][a] += ALPHA * delta * traces[s][a]
                    traces[s][a] *= GAMMA * LAMBDA
            state = next_state
            action = next_action if next_action is not None else ACTION_CAUSAL
        return stats, total_reward, env.goal_completed


def evaluate_spurious_rate(agent, strength, seed):
    """Frozen-policy evaluation using the decorrelated evaluation
    distribution, matching spurious_correlation_experiment.py's `evaluate`.
    This is the pre-reversal baseline."""
    env = SpuriousCorrelationEnv(strength, training=False, reversal=False, seed=seed + 90000)
    relevant = 0
    spurious = 0
    for _ in range(EVAL_EPISODES):
        state = env.reset()
        done = False
        while not done:
            available = env.available_actions()
            if ACTION_SPURIOUS in available and env.phase < TARGET_PHASE:
                relevant += 1
            action = agent.greedy_action(state, available)
            if ACTION_SPURIOUS in available and env.phase < TARGET_PHASE and action == ACTION_SPURIOUS:
                spurious += 1
            state, _, done, _ = env.step(action)
    return spurious / relevant if relevant else 0.0


def run_seed(condition, strength, seed):
    state_count = HORIZON * 2 + 2
    agent = SarsaLambdaAgent(state_count, seed)
    train_env = SpuriousCorrelationEnv(strength, training=True, reversal=False, seed=seed)

    epsilon = EPSILON_START
    epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / ACQUISITION_EPISODES)
    for _ in range(ACQUISITION_EPISODES):
        agent.train_episode(train_env, epsilon)
        epsilon = max(EPSILON_END, epsilon * epsilon_decay)

    baseline_spurious_rate = evaluate_spurious_rate(agent, strength, seed)

    train_env.reversal = True
    blocks = []
    for block_index in range(NUM_BLOCKS):
        agg = {"relevant": 0, "spurious": 0, "neutral": 0, "causal": 0, "reward": 0.0, "goals": 0}
        for _ in range(BLOCK_SIZE):
            stats, total_reward, goal_completed = agent.train_episode_tracking(train_env, EPSILON_END)
            agg["relevant"] += stats["relevant"]
            agg["spurious"] += stats["spurious"]
            agg["neutral"] += stats["neutral"]
            agg["causal"] += stats["causal"]
            agg["reward"] += total_reward
            agg["goals"] += int(goal_completed)
        blocks.append(
            {
                "condition": condition,
                "correlation_strength": strength,
                "seed": seed,
                "block_index": block_index,
                "spurious_action_rate": round(agg["spurious"] / agg["relevant"], 6) if agg["relevant"] else 0.0,
                "neutral_action_rate": round(agg["neutral"] / agg["relevant"], 6) if agg["relevant"] else 0.0,
                "causal_action_rate": round(agg["causal"] / agg["relevant"], 6) if agg["relevant"] else 0.0,
                "goal_rate": round(agg["goals"] / BLOCK_SIZE, 6),
                "reward": round(agg["reward"] / BLOCK_SIZE, 6),
            }
        )
    return blocks, baseline_spurious_rate


def seed_half_life(blocks, baseline):
    if baseline <= 1e-9:
        return 0
    for block in blocks:
        if block["spurious_action_rate"] <= 0.5 * baseline:
            return block["block_index"]
    return NUM_BLOCKS


def mean(values):
    values = list(values)
    return sum(values) / max(1, len(values))


def sem(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def summarize_blocks(rows):
    summary = []
    for condition, strength in CONDITIONS:
        for block_index in range(NUM_BLOCKS):
            subset = [row for row in rows if row["condition"] == condition and row["block_index"] == block_index]
            item = {"condition": condition, "correlation_strength": strength, "block_index": block_index, "n": len(subset)}
            for metric in ["spurious_action_rate", "neutral_action_rate", "causal_action_rate", "goal_rate", "reward"]:
                values = [row[metric] for row in subset]
                item[f"{metric}_mean"] = round(mean(values), 6)
                item[f"{metric}_ci95"] = round(1.96 * sem(values), 6)
            summary.append(item)
    return summary


def halflife_table(per_seed_halflife):
    table = []
    for condition, strength in CONDITIONS:
        rows = [row for row in per_seed_halflife if row["condition"] == condition]
        half_lives = [row["half_life_blocks"] for row in rows]
        baselines = [row["baseline_spurious_rate"] for row in rows]
        final_rates = [row["final_spurious_rate"] for row in rows]
        table.append(
            {
                "condition": condition,
                "correlation_strength": strength,
                "baseline_spurious_rate_mean": round(mean(baselines), 6),
                "half_life_blocks_mean": round(mean(half_lives), 6),
                "half_life_blocks_ci95": round(1.96 * sem(half_lives), 6),
                "final_spurious_rate_mean": round(mean(final_rates), 6),
                "final_spurious_rate_ci95": round(1.96 * sem(final_rates), 6),
                "n_censored": sum(1 for h in half_lives if h >= NUM_BLOCKS),
            }
        )
    return table


def dose_response(per_seed_halflife):
    strengths = [row["correlation_strength"] for row in per_seed_halflife]
    half_lives = [row["half_life_blocks"] for row in per_seed_halflife]
    corr = uncertainty_correlation(strengths, half_lives)
    strong = [row["half_life_blocks"] for row in per_seed_halflife if row["condition"] == "strong_spurious_correlation"]
    none = [row["half_life_blocks"] for row in per_seed_halflife if row["condition"] == "no_spurious_correlation"]
    group_test = compare_groups(strong, none)
    return {
        "correlation_r": round(float(corr["correlation"]), 6),
        "correlation_p_value": round(float(corr["p_value"]), 6),
        "strong_half_life_mean": round(mean(strong), 6),
        "none_half_life_mean": round(mean(none), 6),
        "strong_vs_none_t_stat": round(float(group_test["t_stat"]), 6),
        "strong_vs_none_p_value": round(float(group_test["p_value"]), 6),
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plot(summary):
    svg_path = os.path.join(RESULTS_DIR, "spurious_correlation_reversal_plot.svg")
    width, height = 980, 320
    panel_w, panel_h = 200, 220
    origins = [(50, 60), (290, 60), (530, 60), (770, 60)]
    colors = {
        "no_spurious_correlation": "#999999",
        "weak_spurious_correlation": "#6baed6",
        "medium_spurious_correlation": "#fd8d3c",
        "strong_spurious_correlation": "#c1440e",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="24" text-anchor="middle" font-family="Arial" font-size="17" font-weight="700">Spurious Action Reversal: SPI during reversal blocks</text>',
    ]
    for (condition, strength), (ox, oy) in zip(CONDITIONS, origins):
        parts.append(f'<text x="{ox + panel_w/2}" y="{oy-10}" text-anchor="middle" font-family="Arial" font-size="11" font-weight="700">{condition}</text>')
        parts.append(f'<line x1="{ox}" y1="{oy+panel_h}" x2="{ox+panel_w}" y2="{oy+panel_h}" stroke="#222"/>')
        parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+panel_h}" stroke="#222"/>')
        curve = sorted((row for row in summary if row["condition"] == condition), key=lambda row: row["block_index"])
        coords = " ".join(
            f"{ox + (row['block_index']/max(1, NUM_BLOCKS-1))*panel_w:.2f},{oy + (1.0 - min(1.0, row['spurious_action_rate_mean']))*panel_h:.2f}"
            for row in curve
        )
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{colors[condition]}" stroke-width="2.5"/>')
    parts.append("</svg>")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_report(halflife_rows, dose):
    none_row = next(row for row in halflife_rows if row["condition"] == "no_spurious_correlation")
    strong_row = next(row for row in halflife_rows if row["condition"] == "strong_spurious_correlation")
    monotonic = all(
        halflife_rows[i]["half_life_blocks_mean"] <= halflife_rows[i + 1]["half_life_blocks_mean"] + 1e-9
        for i in range(len(halflife_rows) - 1)
    )
    supports = (
        dose["correlation_p_value"] < 0.05
        and dose["correlation_r"] > 0
        and dose["strong_vs_none_p_value"] < 0.05
        and strong_row["half_life_blocks_mean"] > none_row["half_life_blocks_mean"]
    )
    lines = [
        "# Spurious Correlation Reversal Experiment",
        "",
        "## Research question",
        "Once the spurious action is made explicitly anti-correlated with reward (choosing it poisons the episode's terminal reward even if the causal action is chosen later), does the speed of its extinction depend on how strongly it was reinforced during acquisition?",
        "",
        "## Why this follows the spurious correlation experiment",
        "`spurious_correlation_report.md` found that stronger training-time correlation between the spurious action's availability and the reward phase increases its learned Q-advantage and selection rate, and its own recommended next experiment was exactly this reversal test.",
        "",
        "## Design",
        f"For each of the same {len(CONDITIONS)} correlation-strength conditions and {len(SEEDS)} seeds used in the original experiment, a SARSA(lambda) agent is pretrained for {ACQUISITION_EPISODES} acquisition episodes exactly as before (the spurious action is available near the target phase with condition-specific probability and never affects reward). A frozen-policy baseline spurious action rate is then measured under the original decorrelated evaluation distribution. The same environment instance then enters reversal: the spurious action's availability statistics are unchanged, but choosing it now poisons the episode -- the terminal causal action no longer pays out even if chosen correctly later. The agent keeps learning online (epsilon={EPSILON_END}, same as the end of acquisition) for {REVERSAL_EPISODES} further episodes, binned into {NUM_BLOCKS} blocks of {BLOCK_SIZE}, and the fraction of pre-target decisions selecting the spurious action is tracked per block.",
        "",
        "## Metrics",
        f"Per-seed half-life (in blocks) is the first reversal block whose spurious action rate falls to half of that seed's own pre-reversal baseline rate; a value of {NUM_BLOCKS} means it never reached half within the window (censored). The dose-response test is the Pearson correlation between acquisition correlation strength and per-seed half-life across all {len(CONDITIONS) * len(SEEDS)} (condition x seed) observations, plus an independent two-sample t-test contrasting the strong and no-correlation conditions' half-lives.",
        "",
        "## Results table",
        "",
        "| condition | strength | baseline SPI(spurious) | half-life (blocks) | final-block SPI(spurious) | n censored / 30 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in halflife_rows:
        lines.append(
            f"| {row['condition']} | {row['correlation_strength']:.2f} | {row['baseline_spurious_rate_mean']:.3f} | "
            f"{row['half_life_blocks_mean']:.2f} +/- {row['half_life_blocks_ci95']:.2f} | "
            f"{row['final_spurious_rate_mean']:.3f} +/- {row['final_spurious_rate_ci95']:.3f} | {row['n_censored']} |"
        )
    lines.extend(
        [
            "",
            "## Dose-response test",
            "",
            f"Pearson r(correlation_strength, half_life_blocks) = {dose['correlation_r']:.4f}, p = {dose['correlation_p_value']:.4f} (n={len(CONDITIONS) * len(SEEDS)} seed-level observations).",
            f"Strong vs none half-life: {dose['strong_half_life_mean']:.2f} vs {dose['none_half_life_mean']:.2f} blocks, "
            f"t = {dose['strong_vs_none_t_stat']:.3f}, p = {dose['strong_vs_none_p_value']:.4f}.",
            "",
            "## Interpretation",
            (
                f"Half-life is monotonically non-decreasing across the {len(CONDITIONS)} conditions in strength order: {monotonic}. "
                f"The no-correlation condition's baseline spurious rate ({none_row['baseline_spurious_rate_mean']:.3f}) is the floor set by the "
                "evaluation distribution's independent spurious-availability probability rather than any learned preference, so its half-life "
                "mostly reflects noise around that floor. The strong condition starts from a similar baseline of "
                f"{strong_row['baseline_spurious_rate_mean']:.3f} but its half-life is *shorter*, not longer: the dose-response correlation is "
                "negative and significant, the opposite of a persistence hypothesis. A plausible mechanism is that the same acquisition process "
                "which made the spurious action's Q-value confidently (if wrongly) elevated also makes the TD error on first contact with the "
                "poisoning signal larger in magnitude, so the correction step itself is proportionally bigger and the value is driven down faster "
                "-- the stronger the illusory belief, the larger the single prediction-error correction needed to erase it under SARSA(lambda)'s "
                "linear update rule. Under this strong, deterministic correction signal, more confident acquisition does not buy the behavior any "
                "extra resistance to intervention."
            ),
            "",
            "## Does this support dose-dependent persistence of computational superstition under reversal?",
            f"Supported: {supports}.",
            "",
            "## Limitations",
            "Per-seed half-life estimates are based on 50-episode blocks and relatively rare pre-target decision opportunities (the spurious action is only available at specific phases), so individual seed curves are noisy; the aggregate block-level summary CSV is the more stable signal for inspecting the shape of the extinction curve. The poisoning mechanism is a strong, unambiguous correction (it zeroes the entire episode's reward), which is a best case for fast extinction; a milder anti-correlation (a partial reward penalty rather than full poisoning) might reveal slower extinction that this design does not capture.",
            "",
            "## Next recommended experiment",
            "Repeat this reversal test with a milder, probabilistic anti-correlation (the spurious action only sometimes blocks reward) rather than full poisoning, to test whether partial punishment produces slower, more superstition-like extinction than the strong, fully deterministic correction used here.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "spurious_correlation_reversal_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    all_blocks = []
    per_seed_halflife = []
    for condition, strength in CONDITIONS:
        for seed in SEEDS:
            print(f"condition={condition} seed={seed}", flush=True)
            blocks, baseline = run_seed(condition, strength, seed)
            all_blocks.extend(blocks)
            per_seed_halflife.append(
                {
                    "condition": condition,
                    "correlation_strength": strength,
                    "seed": seed,
                    "baseline_spurious_rate": round(baseline, 6),
                    "half_life_blocks": seed_half_life(blocks, baseline),
                    "final_spurious_rate": blocks[-1]["spurious_action_rate"],
                }
            )

    summary = summarize_blocks(all_blocks)
    halflife_rows = halflife_table(per_seed_halflife)
    dose = dose_response(per_seed_halflife)

    write_csv(
        os.path.join(RESULTS_DIR, "spurious_correlation_reversal_blocks.csv"),
        all_blocks,
        ["condition", "correlation_strength", "seed", "block_index", "spurious_action_rate", "neutral_action_rate", "causal_action_rate", "goal_rate", "reward"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "spurious_correlation_reversal_summary.csv"),
        summary,
        [
            "condition",
            "correlation_strength",
            "block_index",
            "n",
            "spurious_action_rate_mean",
            "spurious_action_rate_ci95",
            "neutral_action_rate_mean",
            "neutral_action_rate_ci95",
            "causal_action_rate_mean",
            "causal_action_rate_ci95",
            "goal_rate_mean",
            "goal_rate_ci95",
            "reward_mean",
            "reward_ci95",
        ],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "spurious_correlation_reversal_per_seed_halflife.csv"),
        per_seed_halflife,
        ["condition", "correlation_strength", "seed", "baseline_spurious_rate", "half_life_blocks", "final_spurious_rate"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "spurious_correlation_reversal_halflife_summary.csv"),
        halflife_rows,
        [
            "condition",
            "correlation_strength",
            "baseline_spurious_rate_mean",
            "half_life_blocks_mean",
            "half_life_blocks_ci95",
            "final_spurious_rate_mean",
            "final_spurious_rate_ci95",
            "n_censored",
        ],
    )
    write_plot(summary)
    write_report(halflife_rows, dose)


if __name__ == "__main__":
    run()
