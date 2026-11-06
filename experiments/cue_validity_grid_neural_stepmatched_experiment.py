"""Step-count-matched retest of neural_raw in the grid environment.

Pre-registered: docs/cue_validity_cross_environment.md, "Step-count-
matched retest." Reuses experiments/cue_validity_grid_neural_experiment.py's
agent/env/metric functions unchanged; only the intervention episode
budget differs (800 episodes in 16 blocks of 50, vs. 1500/30), chosen to
approximately match total training-call count against the original
environment's 1500-episode budget, not nominal episode count.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import compare_groups, cohens_d
from env.cue_validity_grid_env import CueValidityGridEnv
from experiments.cue_validity_cross_environment_experiment import (
    ACQUISITION_EPISODES,
    INTERVENTION_EPSILON,
    SEEDS,
    epsilon_schedule,
    mean,
)
from experiments.cue_validity_grid_neural_experiment import make_env, read_metrics, train_episode
from experiments.cue_validity_nonlinear_experiment import NeuralAgentWrapper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

CONDITION = "neural_raw"
VALIDITY_LEVELS = [0.5, 0.95]
STEPMATCHED_INTERVENTION_EPISODES = 800
STEPMATCHED_BLOCK_SIZE = 50
STEPMATCHED_NUM_BLOCKS = STEPMATCHED_INTERVENTION_EPISODES // STEPMATCHED_BLOCK_SIZE


def run_decorrelate_arm(validity, seed):
    env = make_env(validity, seed)
    agent = NeuralAgentWrapper(seed, 6)
    for epsilon in epsilon_schedule(ACQUISITION_EPISODES):
        train_episode(agent, env, epsilon)
    acq_cds, acq_dep = read_metrics(agent, validity)

    env.decorrelate()
    for _ in range(STEPMATCHED_NUM_BLOCKS):
        for _ in range(STEPMATCHED_BLOCK_SIZE):
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
    for _ in range(STEPMATCHED_NUM_BLOCKS):
        for _ in range(STEPMATCHED_BLOCK_SIZE):
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
            print(f"stepmatched neural_raw v{validity} decorrelate seed={seed}", flush=True)
            summaries.append(run_decorrelate_arm(validity, seed))

            print(f"stepmatched neural_raw v{validity} naive seed={seed}", flush=True)
            summaries.append(run_naive_arm(validity, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_grid_neural_stepmatched_summaries.csv"),
        summaries,
        ["label", "condition", "validity", "seed", "acquisition_proxy_dependence", "final_block_proxy_dependence"],
    )

    results = []
    for validity in VALIDITY_LEVELS:
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

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_grid_neural_stepmatched_criteria.csv"),
        results,
        ["validity", "decorrelate_mean", "naive_mean", "t_stat", "p_value", "cohens_d", "has_effect"],
    )

    step_count_supported = any(r["has_effect"] for r in results)

    lines = [
        "# Step-Count-Matched Retest of neural_raw in Grid Environment",
        "",
        "## Pre-registration",
        "docs/cue_validity_cross_environment.md, \"Step-count-matched retest\" -- written before this data was collected. Flagged in advance as a weak-hypothesis check, not the leading explanation.",
        "",
        "## Design",
        f"neural_raw only, {STEPMATCHED_INTERVENTION_EPISODES}-episode intervention budget in {STEPMATCHED_NUM_BLOCKS} blocks of {STEPMATCHED_BLOCK_SIZE} (vs. 1500/30 in the main extension), chosen to approximately match total training-call count against the original environment's 1500-episode budget.",
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
            "## Verdict",
            f"**Step-count hypothesis supported: {step_count_supported}.**",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_grid_neural_stepmatched_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
