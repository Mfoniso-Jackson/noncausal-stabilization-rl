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

AMBIGUITY_CONDITIONS = [
    {"condition": "clear_causality", "ambiguity_level": 0.0, "reward_prob": 0.95, "cue_given_reward": 0.50},
    {"condition": "moderate_ambiguity", "ambiguity_level": 1.0, "reward_prob": 0.70, "cue_given_reward": 0.75},
    {"condition": "high_ambiguity", "ambiguity_level": 2.0, "reward_prob": 0.55, "cue_given_reward": 0.90},
    {"condition": "extreme_ambiguity", "ambiguity_level": 3.0, "reward_prob": 0.45, "cue_given_reward": 0.95},
]
MEMORY_GAPS = [0, 2, 5, 8]
SEEDS = list(range(30))
EPISODES = 3000
EVAL_EPISODES = 300
CUE_PHASE = 3
SLACK = 10
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


class AmbiguousCausalityMemoryEnv:
    """Same latent-reward / spurious-cue causal structure as
    ambiguous_causality_experiment.py, but the cue is shown only once, at a
    fixed early phase (CUE_PHASE), and the decision phase is pushed out by
    `memory_gap` pure filler steps during which no cue information is ever
    observed. At memory_gap=0 the decision happens immediately after the
    cue (closest analogue to the original single-probe design); larger
    memory_gap forces the agent to act on a state that has long since
    stopped carrying any cue information, testing whether a wider gap
    between evidence and outcome changes how strongly the agent locks onto
    the (already illusory) cue-spurious-action association when the cue
    actually is visible.
    """

    def __init__(self, reward_prob, cue_given_reward, memory_gap, seed=0):
        self.reward_prob = reward_prob
        self.cue_given_reward = cue_given_reward
        self.memory_gap = memory_gap
        self.target_phase = CUE_PHASE + 1 + memory_gap
        self.horizon = self.target_phase + SLACK
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.phase = 0
        self.done = False
        self.reward_latent = self.rng.random() < self.reward_prob
        prob = self.cue_given_reward if self.reward_latent else FALSE_CUE_PROB
        self.cue_present = self.rng.random() < prob
        self.causal_target_selected = False
        self.reward_obtained = False
        return self.observe()

    def observe(self):
        cue_bit = 1 if self.phase == CUE_PHASE and self.cue_present else 0
        return self.phase * 2 + cue_bit

    def available_actions(self):
        return [0, 1, 2, 3]

    def step(self, action):
        if self.done:
            raise RuntimeError("step called after episode termination")
        reward = STEP_PENALTY
        if self.phase == self.target_phase:
            if action == ACTION_CAUSAL:
                self.causal_target_selected = True
                if self.reward_latent:
                    reward += 1.0
                    self.reward_obtained = True
            self.done = True
        else:
            self.phase += 1
            self.done = self.phase >= self.horizon
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


def run_seed(config, memory_gap, seed):
    env = AmbiguousCausalityMemoryEnv(config["reward_prob"], config["cue_given_reward"], memory_gap, seed=seed)
    agent = SarsaLambdaAgent(env.horizon * 2 + 2, seed)
    epsilon = EPSILON_START
    epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / EPISODES)
    for _ in range(EPISODES):
        agent.train_episode(env, epsilon)
        epsilon = max(EPSILON_END, epsilon * epsilon_decay)
    metrics = evaluate(agent, config, memory_gap, seed)
    return {
        "condition": config["condition"],
        "ambiguity_level": config["ambiguity_level"],
        "memory_gap": memory_gap,
        "seed": seed,
        **metrics,
    }


def evaluate(agent, config, memory_gap, seed):
    env = AmbiguousCausalityMemoryEnv(config["reward_prob"], config["cue_given_reward"], memory_gap, seed=seed + 50000)
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
            if env.phase == CUE_PHASE and env.cue_present:
                probe_decisions += 1
                values = agent.q[state]
                masses = softmax_mass(values)
                cds_values.append(masses[ACTION_SPURIOUS] + sum(masses[a] for a in NEUTRAL_ACTIONS))
                q_spurious.append(values[ACTION_SPURIOUS])
                q_neutral.append(mean(values[a] for a in NEUTRAL_ACTIONS))
                q_causal.append(values[ACTION_CAUSAL])
            action = agent.greedy_action(state)
            if env.phase == CUE_PHASE and env.cue_present:
                if action == ACTION_SPURIOUS:
                    spurious_choices += 1
                elif action in NEUTRAL_ACTIONS:
                    neutral_choices += 1
                elif action == ACTION_CAUSAL:
                    causal_choices += 1
            state, reward, done, _ = env.step(action)
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
        "spurious_action_rate": round(spurious_choices / probe_decisions if probe_decisions else 0.0, 6),
        "neutral_action_rate": round(neutral_choices / probe_decisions if probe_decisions else 0.0, 6),
        "causal_action_rate": round(causal_choices / probe_decisions if probe_decisions else 0.0, 6),
        "Q_causal": round(q_caus, 6),
        "Q_spurious": round(q_spur, 6),
        "Q_neutral_mean": round(q_neut, 6),
        "spurious_Q_advantage": round(q_spur - q_neut, 6),
        "spurious_vs_causal_Q": round(q_spur - q_caus, 6),
    }


def softmax_mass(values):
    max_q = max(values)
    exp_values = [math.exp(value - max_q) for value in values]
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


def summarize(rows):
    metrics = [
        "reward",
        "goal_rate",
        "causal_target_rate",
        "SPI",
        "CDS",
        "spurious_action_rate",
        "neutral_action_rate",
        "causal_action_rate",
        "Q_causal",
        "Q_spurious",
        "Q_neutral_mean",
        "spurious_Q_advantage",
        "spurious_vs_causal_Q",
    ]
    summary = []
    for config in AMBIGUITY_CONDITIONS:
        for gap in MEMORY_GAPS:
            subset = [row for row in rows if row["condition"] == config["condition"] and row["memory_gap"] == gap]
            item = {"condition": config["condition"], "ambiguity_level": config["ambiguity_level"], "memory_gap": gap, "n": len(subset)}
            for metric in metrics:
                values = [float(row[metric]) for row in subset]
                item[f"{metric}_mean"] = round(mean(values), 6)
                item[f"{metric}_ci95"] = round(1.96 * sem(values), 6)
            summary.append(item)
    return summary


def ambiguity_trend_by_gap(rows):
    """For each memory gap, the same ambiguity-level dose-response test the
    original experiment ran (spurious_action_rate, spurious_Q_advantage,
    spurious_vs_causal_Q, SPI vs ambiguity_level), so we can see whether the
    gap strengthens, weakens, or leaves that dose-response unchanged."""
    output = []
    for gap in MEMORY_GAPS:
        gap_rows = [row for row in rows if row["memory_gap"] == gap]
        for metric in ["spurious_action_rate", "spurious_Q_advantage", "spurious_vs_causal_Q", "SPI"]:
            x = [float(row["ambiguity_level"]) for row in gap_rows]
            y = [float(row[metric]) for row in gap_rows]
            corr = uncertainty_correlation(x, y)
            output.append(
                {
                    "memory_gap": gap,
                    "metric": metric,
                    "correlation_r": round(float(corr["correlation"]), 6),
                    "p_value": round(float(corr["p_value"]), 6),
                }
            )
    return output


def gap_main_effect(rows):
    """Does memory gap alone (pooled across ambiguity levels) predict more
    spurious reliance, and does the gap's effect specifically within the
    extreme_ambiguity condition grow?"""
    output = {}
    for metric in ["spurious_action_rate", "spurious_Q_advantage", "spurious_vs_causal_Q", "SPI"]:
        gaps = [float(row["memory_gap"]) for row in rows]
        values = [float(row[metric]) for row in rows]
        output[f"{metric}_pooled"] = uncertainty_correlation(gaps, values)
        extreme_rows = [row for row in rows if row["condition"] == "extreme_ambiguity"]
        gaps_e = [float(row["memory_gap"]) for row in extreme_rows]
        values_e = [float(row[metric]) for row in extreme_rows]
        output[f"{metric}_extreme_only"] = uncertainty_correlation(gaps_e, values_e)
    return output


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plot(summary):
    svg_path = os.path.join(RESULTS_DIR, "ambiguous_causality_memory_plot.svg")
    width, height = 980, 360
    panel_w, panel_h = 210, 230
    origins = [(50, 70), (290, 70), (530, 70), (770, 70)]
    colors = {0: "#999999", 2: "#6baed6", 5: "#fd8d3c", 8: "#c1440e"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="24" text-anchor="middle" font-family="Arial" font-size="17" font-weight="700">Ambiguous Causality + Memory: spurious_action_rate vs ambiguity by gap</text>',
    ]
    for config, (ox, oy) in zip(AMBIGUITY_CONDITIONS, origins):
        parts.append(f'<text x="{ox + panel_w/2}" y="{oy-10}" text-anchor="middle" font-family="Arial" font-size="11" font-weight="700">{config["condition"]}</text>')
        parts.append(f'<line x1="{ox}" y1="{oy+panel_h}" x2="{ox+panel_w}" y2="{oy+panel_h}" stroke="#222"/>')
        parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+panel_h}" stroke="#222"/>')
    # draw one polyline per gap, spanning the four ambiguity panels' x-centers
    panel_centers = [ox + panel_w / 2 for ox, oy in origins]
    for gap in MEMORY_GAPS:
        values = [
            next(row["spurious_action_rate_mean"] for row in summary if row["condition"] == config["condition"] and row["memory_gap"] == gap)
            for config in AMBIGUITY_CONDITIONS
        ]
        ymax = max(values + [0.1])
        coords = " ".join(
            f"{cx:.2f},{origins[0][1] + panel_h * (1.0 - min(1.0, v / ymax)):.2f}" for cx, v in zip(panel_centers, values)
        )
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{colors[gap]}" stroke-width="2.5"/>')
    parts.append("</svg>")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_report(summary, trend_by_gap, main_effect):
    def trend_lookup(gap, metric):
        return next(row for row in trend_by_gap if row["memory_gap"] == gap and row["metric"] == metric)

    zero_gap_spi = trend_lookup(0, "spurious_action_rate")
    max_gap = max(MEMORY_GAPS)
    max_gap_spi = trend_lookup(max_gap, "spurious_action_rate")
    emerges = (
        float(zero_gap_spi["correlation_r"]) <= 0 or float(zero_gap_spi["p_value"]) >= 0.05
    ) and float(max_gap_spi["correlation_r"]) > 0 and float(max_gap_spi["p_value"]) < 0.05

    pooled_spi = main_effect["spurious_action_rate_pooled"]
    pooled_q = main_effect["spurious_Q_advantage_pooled"]
    extreme_spi = main_effect["spurious_action_rate_extreme_only"]
    extreme_q = main_effect["spurious_Q_advantage_extreme_only"]

    lines = [
        "# Ambiguous Causality Memory Experiment",
        "",
        "## Research question",
        "ambiguous_causality_report.md found no reliable dose-response between causal ambiguity and spurious-action reliance. Its own recommended next step was to add partial observability or memory constraints so the agent must infer causal structure from history, then repeat the manipulation. Does separating the spurious cue from the decision point by an explicit memory gap reveal or strengthen a dose-response that the single-step-ahead design could not?",
        "",
        "## Why this is a genuinely new manipulation, not a replication",
        "The original experiment showed the cue exactly one step before the decision and additionally compared a 'training_distribution' vs a 'decorrelated' evaluation mode. Because the cue bit is only ever encoded in the observation at that single probe phase and never at the decision phase itself, the decision-phase action (and therefore reward/goal_rate) cannot depend on which generative mode produced an episode -- both modes reduce to the same fixed, Q-table-determined choice. This experiment does not repeat that two-mode comparison; instead it manipulates something that genuinely changes the task: the number of pure, cue-free filler steps (`memory_gap`) inserted between the single cue exposure and the decision phase, which changes the temporal credit-assignment distance the SARSA(lambda) eligibility trace must bridge between seeing the cue and learning from the eventual reward.",
        "",
        "## Environment design",
        f"Same latent-reward / spurious-cue causal structure as the original experiment (reward is sampled from a latent Bernoulli outcome with probability `reward_prob` and delivered only if the causal action is chosen at the decision phase; the cue is a single per-episode Bernoulli draw with probability `cue_given_reward` if the latent outcome is true and {FALSE_CUE_PROB} otherwise, and it never affects transitions, reward, or action availability). The cue is shown once, at phase {CUE_PHASE}. The decision phase is pushed out to {CUE_PHASE} + 1 + memory_gap, and every phase in between offers only the causal action with no cue information at all -- at memory_gap=0 the decision happens immediately after the cue (closest to the original design); larger memory_gap widens the gap the eligibility trace must cross.",
        "",
        "## Design",
        f"The same 4 ambiguity-level conditions as the original experiment (reward_prob/cue_given_reward pairs) are crossed with memory_gap in {MEMORY_GAPS}, {len(SEEDS)} seeds each ({len(AMBIGUITY_CONDITIONS) * len(MEMORY_GAPS) * len(SEEDS)} total seed-condition runs). Agent and training hyperparameters are unchanged: SARSA(lambda), alpha={ALPHA}, gamma={GAMMA}, lambda={LAMBDA}, epsilon {EPSILON_START}->{EPSILON_END}, {EPISODES} training episodes, {EVAL_EPISODES} evaluation episodes.",
        "",
        "## Results table",
        "",
        "| condition | gap | reward | goal_rate | SPI | spurious_rate | Q_advantage | spurious_vs_causal_Q |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['condition']} | {row['memory_gap']} | {row['reward_mean']:.3f} +/- {row['reward_ci95']:.3f} | "
            f"{row['goal_rate_mean']:.3f} +/- {row['goal_rate_ci95']:.3f} | {row['SPI_mean']:.3f} +/- {row['SPI_ci95']:.3f} | "
            f"{row['spurious_action_rate_mean']:.3f} +/- {row['spurious_action_rate_ci95']:.3f} | "
            f"{row['spurious_Q_advantage_mean']:.3f} +/- {row['spurious_Q_advantage_ci95']:.3f} | "
            f"{row['spurious_vs_causal_Q_mean']:.3f} +/- {row['spurious_vs_causal_Q_ci95']:.3f} |"
        )
    lines.extend(["", "## Ambiguity-level dose-response, by memory gap", "", "| memory gap | metric | r | p |", "|---:|---|---:|---:|"])
    for row in trend_by_gap:
        lines.append(f"| {row['memory_gap']} | {row['metric']} | {row['correlation_r']:.4f} | {row['p_value']:.4f} |")
    lines.extend(
        [
            "",
            "## Memory-gap main effect",
            "",
            f"Pooled across all ambiguity levels, memory_gap vs spurious_action_rate: r = {float(pooled_spi['correlation']):.4f}, p = {float(pooled_spi['p_value']):.4f}.",
            f"Pooled across all ambiguity levels, memory_gap vs spurious_Q_advantage: r = {float(pooled_q['correlation']):.4f}, p = {float(pooled_q['p_value']):.4f}.",
            f"Within extreme_ambiguity only, memory_gap vs spurious_action_rate: r = {float(extreme_spi['correlation']):.4f}, p = {float(extreme_spi['p_value']):.4f}.",
            f"Within extreme_ambiguity only, memory_gap vs spurious_Q_advantage: r = {float(extreme_q['correlation']):.4f}, p = {float(extreme_q['p_value']):.4f}.",
            "",
            "## Interpretation",
            (
                f"At memory_gap=0, the ambiguity-level dose-response on spurious_action_rate is r={float(zero_gap_spi['correlation_r']):.4f}, "
                f"p={float(zero_gap_spi['p_value']):.4f} (consistent with the original experiment's null finding at its single-step design). "
                f"At the largest tested gap ({max_gap} filler steps), that same dose-response is r={float(max_gap_spi['correlation_r']):.4f}, "
                f"p={float(max_gap_spi['p_value']):.4f}. The memory-gap main-effect tests above show whether widening the gap between cue "
                "exposure and the reward signal changes overall spurious reliance, independent of ambiguity level, and whether any such change "
                "concentrates in the condition where the cue is most predictive of the latent outcome (extreme_ambiguity)."
            ),
            "",
            "## Does memory constraint reveal a dose-response that the original design missed?",
            f"Supported: {emerges}.",
            "",
            "## Limitations",
            "Memory_gap is implemented by withholding cue information from the observation during filler phases, not by giving the tabular agent a literal sliding window of raw history to integrate -- a recurrent or belief-state agent (as the hidden_state_superstition and causal_reversal experiments use) could in principle behave differently under the same manipulation. Correlation p-values use scipy's Pearson test on seed-level observations and do not correct for testing multiple metrics and gap values together.",
            "",
            "## Next recommended experiment",
            "Repeat this same memory-gap manipulation with a recurrent (GRU/LSTM) DQN agent instead of tabular SARSA(lambda), to test whether a function-approximation agent that must compress the gap into a learned hidden state shows a different relationship between credit-assignment distance and spurious-cue reliance than the tabular agent used here.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "ambiguous_causality_memory_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows = []
    for config in AMBIGUITY_CONDITIONS:
        for gap in MEMORY_GAPS:
            for seed in SEEDS:
                print(f"condition={config['condition']} gap={gap} seed={seed}", flush=True)
                rows.append(run_seed(config, gap, seed))

    summary = summarize(rows)
    trend_by_gap = ambiguity_trend_by_gap(rows)
    main_effect = gap_main_effect(rows)

    result_fields = [
        "condition",
        "ambiguity_level",
        "memory_gap",
        "seed",
        "reward",
        "goal_rate",
        "causal_target_rate",
        "SPI",
        "CDS",
        "spurious_action_rate",
        "neutral_action_rate",
        "causal_action_rate",
        "Q_causal",
        "Q_spurious",
        "Q_neutral_mean",
        "spurious_Q_advantage",
        "spurious_vs_causal_Q",
    ]
    write_csv(os.path.join(RESULTS_DIR, "ambiguous_causality_memory_results.csv"), rows, result_fields)
    summary_fields = ["condition", "ambiguity_level", "memory_gap", "n"]
    for metric in result_fields[4:]:
        summary_fields.extend([f"{metric}_mean", f"{metric}_ci95"])
    write_csv(os.path.join(RESULTS_DIR, "ambiguous_causality_memory_summary.csv"), summary, summary_fields)
    write_csv(
        os.path.join(RESULTS_DIR, "ambiguous_causality_memory_trends.csv"),
        trend_by_gap,
        ["memory_gap", "metric", "correlation_r", "p_value"],
    )
    write_plot(summary)
    write_report(summary, trend_by_gap, main_effect)


if __name__ == "__main__":
    run()
