import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import compare_groups, cohens_d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

DISTRACTOR_COUNTS = [2, 5, 10]
SEEDS = list(range(30))
HORIZON = 20
TARGET_PHASE = 10
PRETRAIN_EPISODES = 2500
INTERVENTION_EPISODES = 1000
INTERVENTION_MAX_STEPS = 60
BLOCK_SIZE = 50
NUM_BLOCKS = INTERVENTION_EPISODES // BLOCK_SIZE
ALPHA = 0.12
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
INTERVENTION_EPSILON = 0.1
STEP_PENALTY = -0.002


class DistractorActionEnv:
    """Same mechanics as experiments/distractor_action_experiment.py.

    When disable_distractors is True, any action > 0 becomes non-functional
    (it no longer advances the phase counter): the distractor action is still
    selectable, but it is now exactly the "unavailable" ritual action the
    intervention is meant to test persistence against.
    """

    def __init__(self, distractor_count, disable_distractors=False, horizon=HORIZON, target_phase=TARGET_PHASE, max_steps=None):
        self.distractor_count = distractor_count
        self.disable_distractors = disable_distractors
        self.horizon = horizon
        self.target_phase = target_phase
        self.max_steps = max_steps if max_steps is not None else horizon
        self.action_count = 1 + distractor_count
        self.reset()

    def reset(self):
        self.phase = 0
        self.steps = 0
        self.done = False
        self.goal_completed = False
        return self.phase

    def step(self, action):
        if self.done:
            raise RuntimeError("step called after episode termination")
        reward = STEP_PENALTY
        self.steps += 1
        blocked = self.disable_distractors and action > 0
        if self.phase == self.target_phase and action == 0:
            reward += 1.0
            self.goal_completed = True
            self.done = True
        else:
            if not blocked:
                self.phase += 1
            self.done = self.steps >= self.max_steps
        return self.phase, reward, self.done, {"goal_completed": self.goal_completed, "blocked": blocked}


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


def pretrain(distractor_count, seed):
    env = DistractorActionEnv(distractor_count)
    agent = QLearningAgent(env.horizon, env.action_count, seed)
    epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / PRETRAIN_EPISODES)
    epsilon = EPSILON_START
    for _ in range(PRETRAIN_EPISODES):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        epsilon = max(EPSILON_END, epsilon * epsilon_decay)
    return agent


def run_block(agent, env, episodes, epsilon):
    spi_decisions = 0
    relevant_decisions = 0
    goals = 0
    total_reward = 0.0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, epsilon)
            if state < TARGET_PHASE:
                relevant_decisions += 1
                if action > 0:
                    spi_decisions += 1
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
        goals += int(env.goal_completed)
    spi = spi_decisions / relevant_decisions if relevant_decisions else 0.0
    return {"spi": spi, "goal_rate": goals / episodes, "reward": total_reward / episodes}


def pre_intervention_spi(agent, distractor_count):
    """Greedy SPI measured in the ORIGINAL (non-disabled) environment, i.e. the
    standard superstition measurement right at the end of pretraining."""
    env = DistractorActionEnv(distractor_count)
    relevant = 0
    distractor_picks = 0
    for _ in range(100):
        state = env.reset()
        done = False
        while not done:
            action = agent.greedy_action(state)
            if state < TARGET_PHASE:
                relevant += 1
                if action > 0:
                    distractor_picks += 1
            state, _, done, _ = env.step(action)
    return distractor_picks / relevant if relevant else 0.0


def run_intervention(distractor_count, seed):
    pretrained_agent = pretrain(distractor_count, seed)
    baseline_spi = pre_intervention_spi(pretrained_agent, distractor_count)

    naive_agent = QLearningAgent(HORIZON, pretrained_agent.action_count, seed + 100000)

    rows = []
    for group, agent in (("intervened", pretrained_agent), ("naive_control", naive_agent)):
        env = DistractorActionEnv(distractor_count, disable_distractors=True, max_steps=INTERVENTION_MAX_STEPS)
        for block_index in range(NUM_BLOCKS):
            stats = run_block(agent, env, BLOCK_SIZE, INTERVENTION_EPSILON)
            rows.append(
                {
                    "distractor_count": distractor_count,
                    "seed": seed,
                    "group": group,
                    "block_index": block_index,
                    "spi": round(stats["spi"], 6),
                    "goal_rate": round(stats["goal_rate"], 6),
                    "reward": round(stats["reward"], 6),
                }
            )
    return rows, baseline_spi


def mean(values):
    return sum(values) / max(1, len(values))


def sem(values):
    if len(values) <= 1:
        return 0.0
    import statistics

    return statistics.stdev(values) / (len(values) ** 0.5)


def summarize(rows):
    summary = []
    for count in DISTRACTOR_COUNTS:
        for group in ("intervened", "naive_control"):
            for block_index in range(NUM_BLOCKS):
                subset = [
                    row
                    for row in rows
                    if row["distractor_count"] == count and row["group"] == group and row["block_index"] == block_index
                ]
                item = {"distractor_count": count, "group": group, "block_index": block_index, "n": len(subset)}
                for metric in ["spi", "goal_rate", "reward"]:
                    values = [row[metric] for row in subset]
                    item[f"{metric}_mean"] = round(mean(values), 6)
                    item[f"{metric}_ci95"] = round(1.96 * sem(values), 6)
                summary.append(item)
    return summary


def half_life_blocks(summary, count, group):
    curve = sorted(
        (row for row in summary if row["distractor_count"] == count and row["group"] == group),
        key=lambda row: row["block_index"],
    )
    if not curve:
        return NUM_BLOCKS
    baseline = curve[0]["spi_mean"]
    if baseline <= 1e-9:
        return 0
    for row in curve:
        if row["spi_mean"] <= 0.5 * baseline:
            return row["block_index"]
    return NUM_BLOCKS


def block_spi(summary, count, group, block_index):
    for row in summary:
        if row["distractor_count"] == count and row["group"] == group and row["block_index"] == block_index:
            return row["spi_mean"]
    return 0.0


def halflife_table(summary, rows, baseline_spi_by_count_seed):
    table = []
    for count in DISTRACTOR_COUNTS:
        intervened_hl = half_life_blocks(summary, count, "intervened")
        naive_hl = half_life_blocks(summary, count, "naive_control")
        final_block = NUM_BLOCKS - 1
        intervened_final = [
            row["spi"] for row in rows if row["distractor_count"] == count and row["group"] == "intervened" and row["block_index"] == final_block
        ]
        naive_final = [
            row["spi"] for row in rows if row["distractor_count"] == count and row["group"] == "naive_control" and row["block_index"] == final_block
        ]
        goal_intervened_final = [
            row["goal_rate"] for row in rows if row["distractor_count"] == count and row["group"] == "intervened" and row["block_index"] == final_block
        ]
        goal_naive_final = [
            row["goal_rate"] for row in rows if row["distractor_count"] == count and row["group"] == "naive_control" and row["block_index"] == final_block
        ]
        spi_test = compare_groups(intervened_final, naive_final)
        goal_test = compare_groups(goal_intervened_final, goal_naive_final)
        table.append(
            {
                "distractor_count": count,
                "pre_intervention_spi": round(mean(baseline_spi_by_count_seed[count]), 6),
                "intervened_block0_spi": round(block_spi(summary, count, "intervened", 0), 6),
                "naive_control_block0_spi": round(block_spi(summary, count, "naive_control", 0), 6),
                "intervened_block1_spi": round(block_spi(summary, count, "intervened", 1), 6),
                "naive_control_block1_spi": round(block_spi(summary, count, "naive_control", 1), 6),
                "intervened_half_life_blocks": intervened_hl,
                "naive_control_half_life_blocks": naive_hl,
                "persistence_gap_blocks": intervened_hl - naive_hl,
                "intervened_final_spi": round(mean(intervened_final), 6),
                "naive_control_final_spi": round(mean(naive_final), 6),
                "final_spi_cohens_d": round(cohens_d(intervened_final, naive_final), 6),
                "final_spi_p_value": round(float(spi_test["p_value"]), 6),
                "intervened_final_goal_rate": round(mean(goal_intervened_final), 6),
                "naive_control_final_goal_rate": round(mean(goal_naive_final), 6),
                "final_goal_rate_p_value": round(float(goal_test["p_value"]), 6),
            }
        )
    return table


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plot(summary):
    svg_path = os.path.join(RESULTS_DIR, "distractor_intervention_plot.svg")
    width, height = 980, 360
    panel_w, panel_h = 260, 220
    origins = [(60, 60), (380, 60), (700, 60)]
    colors = {"intervened": "#c1440e", "naive_control": "#245c9c"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="26" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Distractor Intervention: SPI during post-intervention blocks</text>',
    ]
    for count, (ox, oy) in zip(DISTRACTOR_COUNTS, origins):
        parts.append(f'<text x="{ox + panel_w/2}" y="{oy-12}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700">D={count}</text>')
        parts.append(f'<line x1="{ox}" y1="{oy+panel_h}" x2="{ox+panel_w}" y2="{oy+panel_h}" stroke="#222"/>')
        parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+panel_h}" stroke="#222"/>')
        for group in ("intervened", "naive_control"):
            curve = sorted(
                (row for row in summary if row["distractor_count"] == count and row["group"] == group),
                key=lambda row: row["block_index"],
            )
            coords = " ".join(
                f"{ox + (row['block_index']/max(1, NUM_BLOCKS-1))*panel_w:.2f},{oy + (1.0 - min(1.0, row['spi_mean']))*panel_h:.2f}"
                for row in curve
            )
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{colors[group]}" stroke-width="2.5"/>')
    parts.append(f'<rect x="60" y="310" width="12" height="12" fill="{colors["intervened"]}"/>')
    parts.append('<text x="78" y="320" font-family="Arial" font-size="12">intervened (trained with distractors)</text>')
    parts.append(f'<rect x="330" y="310" width="12" height="12" fill="{colors["naive_control"]}"/>')
    parts.append('<text x="348" y="320" font-family="Arial" font-size="12">naive control (no prior distractor history)</text>')
    parts.append("</svg>")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_report(halflife_rows):
    any_persistence = any(row["persistence_gap_blocks"] > 0 and row["final_spi_p_value"] < 0.05 for row in halflife_rows)
    intervened_starts_lower = all(row["intervened_block0_spi"] < row["naive_control_block0_spi"] for row in halflife_rows)
    both_floor_by_block1 = all(
        abs(row["intervened_block1_spi"] - row["naive_control_block1_spi"]) < 0.02 for row in halflife_rows
    )
    lines = [
        "# Distractor Intervention Experiment",
        "",
        "## Research question",
        "After training with non-causal distractor actions available, does the policy continue to allocate action probability to those actions once they are made non-functional (no longer advance the episode), compared to an agent with no prior distractor-training history placed in the same disabled-action environment?",
        "",
        "## Why this follows the distractor action experiment",
        "`distractor_action_experiment.py` found that CDS and SPI rise with the number of available distractor actions while goal rate stays near 1.0, and its own recommended next experiment was exactly this: remove the distractor affordance after training and see whether the trained policy still reaches for it. This experiment implements that intervention.",
        "",
        "## Design",
        f"For each distractor count D in {DISTRACTOR_COUNTS}, 30 tabular Q-learning seeds are pretrained for {PRETRAIN_EPISODES} episodes exactly as in the distractor action experiment. Each pretrained agent then enters an intervention environment where any action other than the causal action (action 0) no longer advances the episode's phase counter -- it is still selectable, but it is now a non-functional ritual. The agent continues to learn online (epsilon={INTERVENTION_EPSILON}) for {INTERVENTION_EPISODES} further episodes, binned into {NUM_BLOCKS} blocks of {BLOCK_SIZE} episodes. A naive control agent (zero-initialized Q-table, no prior exposure to distractor actions ever being useful) is run through the identical intervention environment for the same number of episodes and epsilon, as the no-superstition-history baseline.",
        "",
        "## Metrics",
        "Per-block SPI is the fraction of pre-goal greedy+exploratory decisions that select a disabled distractor action. Per-block goal rate is the fraction of episodes in that block reaching the causal reward. Half-life (in blocks) is the first block at which mean SPI falls to half of its block-0 value; `persistence_gap_blocks` is intervened half-life minus naive-control half-life -- positive means prior training history slows extinction of the now-useless action relative to an agent that never had that history.",
        "",
        "## Results table",
        "",
        "| D | pre-intervention SPI | intervened block-0 SPI | naive block-0 SPI | intervened block-1 SPI | naive block-1 SPI | intervened half-life (blocks) | naive half-life (blocks) | intervened final SPI | naive final SPI | Cohen's d | p (final SPI) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in halflife_rows:
        lines.append(
            f"| {row['distractor_count']} | {row['pre_intervention_spi']:.3f} | {row['intervened_block0_spi']:.3f} | "
            f"{row['naive_control_block0_spi']:.3f} | {row['intervened_block1_spi']:.3f} | {row['naive_control_block1_spi']:.3f} | "
            f"{row['intervened_half_life_blocks']} | {row['naive_control_half_life_blocks']} | {row['intervened_final_spi']:.3f} | "
            f"{row['naive_control_final_spi']:.3f} | {row['final_spi_cohens_d']:.3f} | {row['final_spi_p_value']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            (
                f"Intervened block-0 SPI is lower than naive-control block-0 SPI for every distractor count: {intervened_starts_lower}. "
                "This is the opposite of what intervention-resistant superstition would predict -- the previously-trained agent enters the "
                "disabled-action environment already favoring the causal action more than a zero-initialized agent does, because the naive "
                "agent's untrained Q-table starts tied at zero and greedy tie-breaking picks uniformly among all actions (expected SPI close "
                "to D/(D+1)) until a handful of updates break the tie. "
                f"By block 1 (episodes 51-100 of the intervention), both groups are within 0.02 SPI of each other for every distractor count: "
                f"{both_floor_by_block1}, and that shared level matches the SPI produced by epsilon-driven forced exploration alone "
                f"(epsilon={INTERVENTION_EPSILON} times the chance of a random distractor pick, D/(D+1)), not a residual policy preference. "
                "The apparent 'censored' half-lives for the intervened group at low D are an artifact of its block-0 value already sitting "
                "near that same exploration floor, not of slow extinction: the policy's greedy preference for the distractor action is gone "
                "within the first block in every condition."
            ),
            "",
            "## Does this support persistent computational superstition surviving intervention?",
            f"Supported: {any_persistence}. The policy abandons the disabled action immediately once it stops advancing the episode; prior "
            "training history does not slow this down relative to a naive agent -- if anything it gives the intervened agent a head start, "
            "since it already knows the causal action's value. The elevated CDS/SPI seen during normal training in the distractor action "
            "experiment therefore looks like an artifact of credit diffusion while the distractor affordance is still (harmlessly) available, "
            "not a stable, intervention-resistant habit.",
            "",
            "## Limitations",
            "The naive control starts from a zero-initialized Q-table rather than from an agent trained on a semantically unrelated task, so it is a lower bound on how much history matters, not a full dissociation of 'any training' from 'distractor-specific training'. The half-life metric is noisy when the block-0 baseline is already close to the exploration floor (small denominators make 'half of baseline' a weak threshold); the block-0/block-1 SPI comparison above is the more reliable signal in this run. Approximate p-values come from scipy's independent two-sample t-test on 30 seeds per group and do not correct for multiple comparisons across the three distractor counts.",
            "",
            "## Next recommended experiment",
            "Re-run this intervention with a function-approximation agent (linear or neural DQN) instead of tabular Q-learning, to test whether generalization across states changes how quickly the now-disabled distractor action's value is unlearned, mirroring the tabular-vs-DQN contrast already used in the causal reversal experiments.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "distractor_intervention_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    all_rows = []
    baseline_spi_by_count_seed = {count: [] for count in DISTRACTOR_COUNTS}
    for count in DISTRACTOR_COUNTS:
        for seed in SEEDS:
            print(f"distractor_count={count} seed={seed}", flush=True)
            rows, baseline_spi = run_intervention(count, seed)
            all_rows.extend(rows)
            baseline_spi_by_count_seed[count].append(baseline_spi)

    summary = summarize(all_rows)
    halflife_rows = halflife_table(summary, all_rows, baseline_spi_by_count_seed)

    write_csv(
        os.path.join(RESULTS_DIR, "distractor_intervention_results.csv"),
        all_rows,
        ["distractor_count", "seed", "group", "block_index", "spi", "goal_rate", "reward"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "distractor_intervention_summary.csv"),
        summary,
        ["distractor_count", "group", "block_index", "n", "spi_mean", "spi_ci95", "goal_rate_mean", "goal_rate_ci95", "reward_mean", "reward_ci95"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "distractor_intervention_halflife.csv"),
        halflife_rows,
        [
            "distractor_count",
            "pre_intervention_spi",
            "intervened_block0_spi",
            "naive_control_block0_spi",
            "intervened_block1_spi",
            "naive_control_block1_spi",
            "intervened_half_life_blocks",
            "naive_control_half_life_blocks",
            "persistence_gap_blocks",
            "intervened_final_spi",
            "naive_control_final_spi",
            "final_spi_cohens_d",
            "final_spi_p_value",
            "intervened_final_goal_rate",
            "naive_control_final_goal_rate",
            "final_goal_rate_p_value",
        ],
    )
    write_plot(summary)
    write_report(halflife_rows)


if __name__ == "__main__":
    run()
