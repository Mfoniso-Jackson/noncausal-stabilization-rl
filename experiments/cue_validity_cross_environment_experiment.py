"""Cross-environment replication: cue-validity persistence in a
structurally different, grid-navigation environment.

Pre-registered design: docs/cue_validity_cross_environment.md. Reuses
SarsaLambdaAgent (experiments/cue_validity_mve_experiment.py) and the
LinearAgentWrapper/DQNAgent pattern (experiments/cue_validity_nonlinear_experiment.py)
unchanged, pointed at env/cue_validity_grid_env.py::CueValidityGridEnv
instead of CueValidityEnv.
"""

import csv
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dqn_agent import DQNAgent
from analysis.stats import compare_groups, cohens_d
from env.cue_validity_grid_env import ACTION_CAUSAL, ACTION_CUE, NEUTRAL_ACTIONS, CueValidityGridEnv
from metrics.cue_validity import cue_dependence_score, proxy_dependence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

AVAILABLE_ACTIONS = (ACTION_CAUSAL, ACTION_CUE) + NEUTRAL_ACTIONS
CONDITIONS = ["tabular", "linear_raw"]
VALIDITY_LEVELS = [0.0, 0.5, 0.95]
SEEDS = list(range(15))
ACQUISITION_EPISODES = 5000
INTERVENTION_EPISODES = 1500
BLOCK_SIZE = 50
NUM_BLOCKS = INTERVENTION_EPISODES // BLOCK_SIZE
EPSILON_START = 1.0
EPSILON_END = 0.04
INTERVENTION_EPSILON = EPSILON_END
GRID_SIZE = 5

ALPHA = 0.08
GAMMA = 0.95
LAMBDA = 0.80


class SarsaLambdaAgent:
    """Unchanged from experiments/cue_validity_mve_experiment.py, except
    for taking state_count directly rather than deriving it from a phase
    horizon."""

    def __init__(self, state_count, seed):
        self.rng = random.Random(seed)
        self.q = [[0.0 for _ in range(4)] for _ in range(state_count)]

    def act(self, state, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(4)
        return self.greedy_action(state)

    def greedy_action(self, state):
        values = self.q[state]
        best = max(values)
        best_actions = [a for a in range(4) if abs(values[a] - best) < 1e-12]
        return self.rng.choice(best_actions)

    def train_episode(self, env, epsilon):
        traces = [[0.0 for _ in range(4)] for _ in range(len(self.q))]
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
                for a in range(4):
                    self.q[s][a] += ALPHA * delta * traces[s][a]
                    traces[s][a] *= GAMMA * LAMBDA
            state = next_state
            action = next_action if next_action is not None else ACTION_CAUSAL


class LinearAgentWrapper:
    """Unchanged from experiments/cue_validity_nonlinear_experiment.py."""

    def __init__(self, seed, feature_dim):
        self.agent = DQNAgent(
            state_size=feature_dim,
            action_size=4,
            seed=seed,
            lr=0.015,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay=1.0,
            replay_size=8000,
            batch_size=32,
            target_update=50,
            train_frequency=4,
        )
        self.epsilon = EPSILON_START
        self.rng = random.Random(seed)

    def act(self, state):
        self.agent.epsilon = self.epsilon
        return self.agent.act(state, training=True)

    def remember(self, state, action, reward, next_state, done):
        self.agent.remember(state, action, reward, next_state, done)

    def train_step(self):
        self.agent.train_step()

    def q_values_for_features(self, features):
        return self.agent.q_values(features)


def make_env(validity, seed):
    return CueValidityGridEnv(validity, grid_size=GRID_SIZE, seed=seed)


def make_agent(condition, env, seed):
    if condition == "tabular":
        return SarsaLambdaAgent(env.state_count, seed)
    return LinearAgentWrapper(seed, 6)


def epsilon_schedule(total_episodes):
    decay = (EPSILON_END / EPSILON_START) ** (1.0 / total_episodes)
    epsilon = EPSILON_START
    for _ in range(total_episodes):
        yield epsilon
        epsilon = max(EPSILON_END, epsilon * decay)


def train_episode_linear(agent, env, epsilon):
    agent.epsilon = epsilon
    env.reset()
    state = env.observation_vector()
    done = False
    while not done:
        action = agent.act(state)
        _, reward, done, _ = env.step(action)
        next_state = env.observation_vector()
        agent.remember(state, action, reward, next_state, done)
        agent.train_step()
        state = next_state


def train_episode(condition, agent, env, epsilon):
    if condition == "tabular":
        agent.train_episode(env, epsilon)
    else:
        train_episode_linear(agent, env, epsilon)


def probe_q_values(condition, agent, validity, cue_present):
    probe_env = CueValidityGridEnv(validity, grid_size=GRID_SIZE, seed=0)
    probe_env.pos = probe_env.decision_cell
    probe_env.cue_present = int(cue_present)
    if condition == "tabular":
        state = probe_env.observe()
        values = agent.q[state]
    else:
        features = probe_env.observation_vector()
        values = agent.q_values_for_features(features)
    return {a: values[a] for a in AVAILABLE_ACTIONS}


def read_metrics(condition, agent, validity):
    q_present = probe_q_values(condition, agent, validity, True)
    q_absent = probe_q_values(condition, agent, validity, False)
    cds = cue_dependence_score(q_present, AVAILABLE_ACTIONS, ACTION_CAUSAL, ACTION_CUE, NEUTRAL_ACTIONS)
    dep = proxy_dependence(q_present, q_absent, AVAILABLE_ACTIONS, ACTION_CAUSAL)
    return cds, dep


def run_arm(label, condition, validity, decorrelate_after_acquisition, seed):
    env = make_env(validity, seed)
    agent = make_agent(condition, env, seed)
    for epsilon in epsilon_schedule(ACQUISITION_EPISODES):
        train_episode(condition, agent, env, epsilon)
    acq_cds, acq_dep = read_metrics(condition, agent, validity)

    if decorrelate_after_acquisition:
        env.decorrelate()

    block_rows = []
    for block_index in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(condition, agent, env, INTERVENTION_EPSILON)
        block_cds, block_dep = read_metrics(condition, agent, validity)
        block_rows.append(
            {
                "label": label,
                "condition": condition,
                "validity": validity,
                "seed": seed,
                "block_index": block_index,
                "cds": round(block_cds, 6),
                "proxy_dependence": round(block_dep, 6),
            }
        )
    summary = {
        "label": label,
        "condition": condition,
        "validity": validity,
        "seed": seed,
        "acquisition_cds": round(acq_cds, 6),
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_block_cds": block_rows[-1]["cds"],
        "final_block_proxy_dependence": block_rows[-1]["proxy_dependence"],
    }
    return summary, block_rows


def run_naive_arm(condition, validity, seed):
    env = make_env(validity, seed)
    env.decorrelate()
    agent = make_agent(condition, env, seed + 1_000_000)

    block_rows = []
    for block_index in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(condition, agent, env, INTERVENTION_EPSILON)
        block_cds, block_dep = read_metrics(condition, agent, validity)
        block_rows.append(
            {
                "label": "naive",
                "condition": condition,
                "validity": validity,
                "seed": seed,
                "block_index": block_index,
                "cds": round(block_cds, 6),
                "proxy_dependence": round(block_dep, 6),
            }
        )
    summary = {
        "label": "naive",
        "condition": condition,
        "validity": validity,
        "seed": seed,
        "acquisition_cds": None,
        "acquisition_proxy_dependence": None,
        "final_block_cds": block_rows[-1]["cds"],
        "final_block_proxy_dependence": block_rows[-1]["proxy_dependence"],
    }
    return summary, block_rows


def mean(values):
    values = list(values)
    return sum(values) / max(1, len(values))


def sem(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def final_dep_values(summaries, condition, validity, label):
    return [
        abs(row["final_block_proxy_dependence"])
        for row in summaries
        if row["condition"] == condition and row["validity"] == validity and row["label"] == label
    ]


def success_criteria(summaries):
    results = []
    for condition in CONDITIONS:
        for validity in (0.5, 0.95):
            decorrelate_values = final_dep_values(summaries, condition, validity, "decorrelate")
            naive_values = final_dep_values(summaries, condition, validity, "naive")
            test = compare_groups(decorrelate_values, naive_values)
            d = cohens_d(decorrelate_values, naive_values)
            results.append(
                {
                    "condition": condition,
                    "validity": validity,
                    "decorrelate_mean": round(mean(decorrelate_values), 6),
                    "naive_mean": round(mean(naive_values), 6),
                    "t_stat": round(float(test["t_stat"]), 6),
                    "p_value": round(float(test["p_value"]), 6),
                    "cohens_d": round(float(d), 6),
                    "has_effect": bool(test["p_value"] < 0.05 and mean(decorrelate_values) > mean(naive_values)),
                }
            )

    floor_results = []
    for condition in CONDITIONS:
        decorrelate_values = final_dep_values(summaries, condition, 0.0, "decorrelate")
        naive_values = final_dep_values(summaries, condition, 0.0, "naive")
        test = compare_groups(decorrelate_values, naive_values)
        d = cohens_d(decorrelate_values, naive_values)
        floor_results.append(
            {
                "condition": condition,
                "decorrelate_mean": round(mean(decorrelate_values), 6),
                "naive_mean": round(mean(naive_values), 6),
                "t_stat": round(float(test["t_stat"]), 6),
                "p_value": round(float(test["p_value"]), 6),
                "cohens_d": round(float(d), 6),
                "no_spurious_effect": bool(test["p_value"] >= 0.05),
            }
        )

    linear_rows = [r for r in results if r["condition"] == "linear_raw"]
    tabular_rows = [r for r in results if r["condition"] == "tabular"]
    criterion_1 = any(r["has_effect"] for r in linear_rows)
    criterion_2 = not any(r["has_effect"] for r in tabular_rows)
    criterion_3 = all(r["no_spurious_effect"] for r in floor_results)
    pattern_confirmed = bool(criterion_1 and criterion_2 and criterion_3)

    return results, floor_results, {
        "criterion_1_linear_effect": criterion_1,
        "criterion_2_tabular_null": criterion_2,
        "criterion_3_floor_clean": criterion_3,
        "pattern_confirmed": pattern_confirmed,
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(summaries, criteria_results, floor_results, verdict):
    def rows_for(condition, label, validity):
        return [row for row in summaries if row["condition"] == condition and row["label"] == label and row["validity"] == validity]

    def mean_field(condition, label, validity, field):
        values = [row[field] for row in rows_for(condition, label, validity) if row[field] is not None]
        return mean(values) if values else float("nan")

    lines = [
        "# Cue Validity Cross-Environment Replication",
        "",
        "## Pre-registration",
        "Design, metrics, and success/failure criteria were written and committed in docs/cue_validity_cross_environment.md before any of the data below was collected. This experiment is the explicit Go/No-Go gate named in the publication plan's Section 12.",
        "",
        "## Design summary",
        f"Environment: CueValidityGridEnv (grid_size={GRID_SIZE}), a structurally different task from CueValidityEnv -- real multi-step 2D navigation to a decision cell, not a phase countdown that ignores the agent's action. condition in {CONDITIONS}, cue_validity in {VALIDITY_LEVELS}, {len(SEEDS)} seeds. {ACQUISITION_EPISODES} acquisition episodes, {INTERVENTION_EPISODES} post-intervention episodes in {NUM_BLOCKS} blocks of {BLOCK_SIZE}.",
        "",
        "## Results table (mean across seeds)",
        "",
        "| condition | validity | acquisition dep | decorrelate final dep | naive final dep |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for validity in VALIDITY_LEVELS:
            acq = mean_field(condition, "decorrelate", validity, "acquisition_proxy_dependence")
            dec = mean_field(condition, "decorrelate", validity, "final_block_proxy_dependence")
            naive = mean_field(condition, "naive", validity, "final_block_proxy_dependence")
            lines.append(f"| {condition} | {validity} | {acq:.3f} | {dec:.3f} | {naive:.3f} |")

    lines.extend(
        [
            "",
            "## Primary test: decorrelate-vs-naive gap, non-floor validity levels",
            "",
            "| condition | validity | decorrelate | naive | t | p | Cohen's d | has effect |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in criteria_results:
        lines.append(
            f"| {row['condition']} | {row['validity']} | {row['decorrelate_mean']:.3f} | {row['naive_mean']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['has_effect']} |"
        )

    lines.extend(
        [
            "",
            "## Floor control (cue_validity=0.0): comparison methodology must not manufacture a spurious gap",
            "",
            "| condition | decorrelate | naive | t | p | Cohen's d | no spurious effect |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in floor_results:
        lines.append(
            f"| {row['condition']} | {row['decorrelate_mean']:.3f} | {row['naive_mean']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['no_spurious_effect']} |"
        )

    lines.extend(
        [
            "",
            "## Verdict (per pre-registered criteria in docs/cue_validity_cross_environment.md)",
            f"Criterion 1 (linear_raw shows a significant effect at >=1 non-floor validity level): {verdict['criterion_1_linear_effect']}.",
            f"Criterion 2 (tabular shows no significant effect at either non-floor validity level): {verdict['criterion_2_tabular_null']}.",
            f"Criterion 3 (floor control shows no spurious effect for either condition): {verdict['criterion_3_floor_clean']}.",
            f"**Pattern confirmed (all three criteria met): {verdict['pattern_confirmed']}.**",
            "",
            "## Limitations",
            "This replication tests only the decorrelate intervention with tabular and linear_raw agents, by pre-registered design -- reversal, neural agents, and interaction-only encoding in this environment are explicitly deferred to a follow-up, contingent on this core pair replicating. A single grid size (5x5) and a single start/decision-cell layout are tested, not a sweep. The episode budget (5000 acquisition) was increased from the 3000 used elsewhere in this lineage to accommodate the added navigation-learning requirement; this was decided from a pre-run timing/learnability smoke test, not tuned post-hoc on this experiment's own persistence results.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_cross_environment_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    summaries = []
    block_rows = []

    for condition in CONDITIONS:
        for validity in VALIDITY_LEVELS:
            for seed in SEEDS:
                print(f"{condition} v{validity} decorrelate seed={seed}", flush=True)
                summary, blocks = run_arm("decorrelate", condition, validity, True, seed)
                summaries.append(summary)
                block_rows.extend(blocks)

                print(f"{condition} v{validity} naive seed={seed}", flush=True)
                summary, blocks = run_naive_arm(condition, validity, seed)
                summaries.append(summary)
                block_rows.extend(blocks)

    criteria_results, floor_results, verdict = success_criteria(summaries)

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_cross_environment_summaries.csv"),
        summaries,
        ["label", "condition", "validity", "seed", "acquisition_cds", "acquisition_proxy_dependence", "final_block_cds", "final_block_proxy_dependence"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_cross_environment_blocks.csv"),
        block_rows,
        ["label", "condition", "validity", "seed", "block_index", "cds", "proxy_dependence"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_cross_environment_criteria.csv"),
        criteria_results,
        ["condition", "validity", "decorrelate_mean", "naive_mean", "t_stat", "p_value", "cohens_d", "has_effect"],
    )
    write_report(summaries, criteria_results, floor_results, verdict)


if __name__ == "__main__":
    run()
