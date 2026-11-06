"""neural_raw extension to the cross-environment replication.

Pre-registered: docs/cue_validity_cross_environment.md, "Extension:
neural_raw in the grid environment." Reuses
experiments/cue_validity_nonlinear_experiment.py's NeuralAgentWrapper/
MLPQNet unchanged, pointed at CueValidityGridEnv's observation_vector()
(6 raw features) instead of CueValidityEnv's. Reuses
experiments/cue_validity_cross_environment_experiment.py's epsilon
schedule, block structure, and metric-reading conventions.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import compare_groups, cohens_d
from env.cue_validity_grid_env import ACTION_CAUSAL, ACTION_CUE, NEUTRAL_ACTIONS, CueValidityGridEnv
from experiments.cue_validity_cross_environment_experiment import (
    ACQUISITION_EPISODES,
    BLOCK_SIZE,
    EPSILON_END,
    GRID_SIZE,
    INTERVENTION_EPSILON,
    NUM_BLOCKS,
    SEEDS,
    VALIDITY_LEVELS,
    epsilon_schedule,
    mean,
)
from experiments.cue_validity_nonlinear_experiment import NeuralAgentWrapper
from metrics.cue_validity import cue_dependence_score, proxy_dependence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
CROSS_ENV_SUMMARY_CSV = os.path.join(RESULTS_DIR, "cue_validity_cross_environment_summaries.csv")

AVAILABLE_ACTIONS = (ACTION_CAUSAL, ACTION_CUE) + NEUTRAL_ACTIONS
CONDITION = "neural_raw"


def make_env(validity, seed):
    return CueValidityGridEnv(validity, grid_size=GRID_SIZE, seed=seed)


def train_episode(agent, env, epsilon):
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


def probe_q_values(validity, agent, cue_present):
    probe_env = CueValidityGridEnv(validity, grid_size=GRID_SIZE, seed=0)
    probe_env.pos = probe_env.decision_cell
    probe_env.cue_present = int(cue_present)
    values = agent.q_values_for_features(probe_env.observation_vector())
    return {a: values[a] for a in AVAILABLE_ACTIONS}


def read_metrics(agent, validity):
    q_present = probe_q_values(validity, agent, True)
    q_absent = probe_q_values(validity, agent, False)
    cds = cue_dependence_score(q_present, AVAILABLE_ACTIONS, ACTION_CAUSAL, ACTION_CUE, NEUTRAL_ACTIONS)
    dep = proxy_dependence(q_present, q_absent, AVAILABLE_ACTIONS, ACTION_CAUSAL)
    return cds, dep


def run_decorrelate_arm(validity, seed):
    env = make_env(validity, seed)
    agent = NeuralAgentWrapper(seed, 6)
    for epsilon in epsilon_schedule(ACQUISITION_EPISODES):
        train_episode(agent, env, epsilon)
    acq_cds, acq_dep = read_metrics(agent, validity)

    env.decorrelate()
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(agent, env, INTERVENTION_EPSILON)
    final_cds, final_dep = read_metrics(agent, validity)

    return {
        "label": "decorrelate",
        "condition": CONDITION,
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_block_proxy_dependence": round(final_dep, 6),
    }


def run_naive_arm(validity, seed):
    env = make_env(validity, seed)
    env.decorrelate()
    agent = NeuralAgentWrapper(seed + 1_000_000, 6)
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(agent, env, INTERVENTION_EPSILON)
    final_cds, final_dep = read_metrics(agent, validity)
    return {
        "label": "naive",
        "condition": CONDITION,
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": None,
        "final_block_proxy_dependence": round(final_dep, 6),
    }


def final_dep_values(summaries, validity, label):
    return [
        abs(row["final_block_proxy_dependence"])
        for row in summaries
        if row["validity"] == validity and row["label"] == label
    ]


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    summaries = []

    for validity in VALIDITY_LEVELS:
        for seed in SEEDS:
            print(f"neural_raw v{validity} decorrelate seed={seed}", flush=True)
            summaries.append(run_decorrelate_arm(validity, seed))

            print(f"neural_raw v{validity} naive seed={seed}", flush=True)
            summaries.append(run_naive_arm(validity, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_grid_neural_summaries.csv"),
        summaries,
        ["label", "condition", "validity", "seed", "acquisition_proxy_dependence", "final_block_proxy_dependence"],
    )

    results = []
    for validity in (0.5, 0.95):
        decorrelate_values = final_dep_values(summaries, validity, "decorrelate")
        naive_values = final_dep_values(summaries, validity, "naive")
        test = compare_groups(decorrelate_values, naive_values)
        d = cohens_d(decorrelate_values, naive_values)
        results.append(
            {
                "validity": validity,
                "decorrelate_mean": round(mean(decorrelate_values), 6),
                "naive_mean": round(mean(naive_values), 6),
                "t_stat": round(float(test["t_stat"]), 6),
                "p_value": round(float(test["p_value"]), 6),
                "cohens_d": round(float(d), 6),
                "has_effect": bool(test["p_value"] < 0.05 and mean(decorrelate_values) > mean(naive_values)),
            }
        )

    floor_decorrelate = final_dep_values(summaries, 0.0, "decorrelate")
    floor_naive = final_dep_values(summaries, 0.0, "naive")
    floor_test = compare_groups(floor_decorrelate, floor_naive)
    floor_d = cohens_d(floor_decorrelate, floor_naive)
    floor_result = {
        "decorrelate_mean": round(mean(floor_decorrelate), 6),
        "naive_mean": round(mean(floor_naive), 6),
        "t_stat": round(float(floor_test["t_stat"]), 6),
        "p_value": round(float(floor_test["p_value"]), 6),
        "cohens_d": round(float(floor_d), 6),
        "no_spurious_effect": bool(floor_test["p_value"] >= 0.05),
    }

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_grid_neural_criteria.csv"),
        results,
        ["validity", "decorrelate_mean", "naive_mean", "t_stat", "p_value", "cohens_d", "has_effect"],
    )

    criterion_met = any(r["has_effect"] for r in results) and floor_result["no_spurious_effect"]

    lines = [
        "# neural_raw Extension to Cross-Environment Replication",
        "",
        "## Pre-registration",
        "Design and success criterion were written and committed in docs/cue_validity_cross_environment.md (\"Extension: neural_raw in the grid environment\") before this data was collected.",
        "",
        "## Results",
        "",
        "| validity | decorrelate | naive | t | p | Cohen's d | has effect |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['validity']} | {row['decorrelate_mean']:.3f} | {row['naive_mean']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['has_effect']} |"
        )
    lines.extend(
        [
            "",
            "## Floor control (cue_validity=0.0)",
            f"decorrelate={floor_result['decorrelate_mean']:.3f}, naive={floor_result['naive_mean']:.3f}, "
            f"t={floor_result['t_stat']:.3f}, p={floor_result['p_value']:.4f}, d={floor_result['cohens_d']:.3f}, "
            f"no spurious effect: {floor_result['no_spurious_effect']}.",
            "",
            "## Verdict",
            f"**neural_raw extends the graded pattern to the grid environment: {criterion_met}.**",
            f"(tabular and linear_raw results, for comparison, are unchanged from cue_validity_cross_environment_report.md: linear_raw showed p<0.0001, d=6.12/5.44 at the same two validity levels.)",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_grid_neural_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
