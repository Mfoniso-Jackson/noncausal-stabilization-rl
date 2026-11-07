"""Capacity-sharing vs. step-count test for neural_raw's grid-environment
null at the original 1500-episode intervention budget.

Pre-registered design: docs/cue_validity_grid_capacity_sharing.md. Reuses
experiments/cue_validity_grid_neural_experiment.py's agent/feature/metric
functions unchanged; only the environment's auto_navigate flag differs,
holding step count exactly constant against the already-null
normal-navigation condition.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import compare_groups, cohens_d
from env.cue_validity_grid_env import CueValidityGridEnv
from experiments.cue_validity_cross_environment_experiment import (
    ACQUISITION_EPISODES,
    BLOCK_SIZE,
    GRID_SIZE,
    INTERVENTION_EPSILON,
    NUM_BLOCKS,
    SEEDS,
    epsilon_schedule,
    mean,
)
from experiments.cue_validity_grid_neural_experiment import read_metrics, train_episode
from experiments.cue_validity_nonlinear_experiment import NeuralAgentWrapper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

VALIDITY_LEVELS = [0.5, 0.95]


def make_env(validity, seed):
    return CueValidityGridEnv(validity, grid_size=GRID_SIZE, seed=seed, auto_navigate=True)


def run_decorrelate_arm(validity, seed):
    env = make_env(validity, seed)
    agent = NeuralAgentWrapper(seed, 6)
    for epsilon in epsilon_schedule(ACQUISITION_EPISODES):
        train_episode(agent, env, epsilon)
    _, acq_dep = read_metrics(agent, validity)

    env.decorrelate()
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(agent, env, INTERVENTION_EPSILON)
    _, final_dep = read_metrics(agent, validity)

    return {
        "label": "decorrelate",
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_proxy_dependence": round(final_dep, 6),
    }


def run_naive_arm(validity, seed):
    env = make_env(validity, seed)
    env.decorrelate()
    agent = NeuralAgentWrapper(seed + 1_000_000, 6)
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(agent, env, INTERVENTION_EPSILON)
    _, final_dep = read_metrics(agent, validity)
    return {
        "label": "naive",
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": None,
        "final_proxy_dependence": round(final_dep, 6),
    }


def gap_values(summaries, validity, label):
    return [abs(r["final_proxy_dependence"]) for r in summaries if r["validity"] == validity and r["label"] == label]


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
            print(f"auto_navigate v{validity} decorrelate seed={seed}", flush=True)
            summaries.append(run_decorrelate_arm(validity, seed))
            print(f"auto_navigate v{validity} naive seed={seed}", flush=True)
            summaries.append(run_naive_arm(validity, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_grid_capacity_sharing_summaries.csv"),
        summaries,
        ["label", "validity", "seed", "acquisition_proxy_dependence", "final_proxy_dependence"],
    )

    results = []
    for validity in VALIDITY_LEVELS:
        decorrelate = gap_values(summaries, validity, "decorrelate")
        naive = gap_values(summaries, validity, "naive")
        test = compare_groups(decorrelate, naive)
        d = cohens_d(decorrelate, naive)
        results.append(
            {
                "validity": validity,
                "decorrelate_mean": round(mean(decorrelate), 6),
                "naive_mean": round(mean(naive), 6),
                "t_stat": round(float(test["t_stat"]), 6),
                "p_value": round(float(test["p_value"]), 6),
                "cohens_d": round(float(d), 6),
                "has_effect": bool(test["p_value"] < 0.05 and mean(decorrelate) > mean(naive)),
            }
        )

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_grid_capacity_sharing_criteria.csv"),
        results,
        ["validity", "decorrelate_mean", "naive_mean", "t_stat", "p_value", "cohens_d", "has_effect"],
    )

    capacity_sharing_supported = any(r["has_effect"] for r in results)

    lines = [
        "# Grid Environment Capacity-Sharing vs. Step-Count Test",
        "",
        "## Pre-registration",
        "Design and criteria were written and committed in docs/cue_validity_grid_capacity_sharing.md before any of the data below was collected.",
        "",
        "## Design summary",
        f"neural_raw, auto_navigate=True (step count held exactly constant vs. the already-null normal-navigation 1500-episode condition), cue_validity in {VALIDITY_LEVELS}, {len(SEEDS)} seeds, decorrelate intervention, original 1500-episode intervention budget.",
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
            f"**Capacity-sharing hypothesis supported: {capacity_sharing_supported}.**",
            f"(Comparison: normal navigation at this same 1500-episode budget was null; the 800-episode step-matched normal-navigation condition showed a significant effect.)",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_grid_capacity_sharing_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
