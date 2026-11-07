"""Replay buffer size sweep: linear_raw decorrelate-vs-naive gap across
replay buffer sizes.

Pre-registered design: docs/cue_validity_replay_buffer_sweep.md. Reuses
experiments/cue_validity_proxy_dependence_experiment.py's make_env,
probe_q_values, read_metrics, epsilon_schedule unchanged; only
replay_size varies (LinearAgentWrapper hardcodes 8000).
"""

import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dqn_agent import DQNAgent
from analysis.stats import compare_groups, cohens_d
from experiments.cue_validity_proxy_dependence_experiment import (
    ACQUISITION_EPISODES,
    BLOCK_SIZE,
    EPSILON_END,
    EPSILON_START,
    INTERVENTION_EPSILON,
    NUM_BLOCKS,
    SEEDS,
    epsilon_schedule,
    make_env,
    read_metrics,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

AGENT_TYPE = "linear_raw"
BUFFER_SIZES = [500, 2000, 8000, 20000]


class ReplayBufferAgentWrapper:
    """Identical to LinearAgentWrapper(orthogonal=False) except replay_size
    is a parameter."""

    def __init__(self, seed, replay_size):
        self.orthogonal = False
        self.agent = DQNAgent(
            state_size=6,
            action_size=4,
            seed=seed,
            lr=0.015,
            gamma=0.95,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay=1.0,
            replay_size=replay_size,
            batch_size=32,
            target_update=50,
            train_frequency=4,
        )
        self.rng = random.Random(seed)

    def _features(self, env):
        return env.observation_vector()

    def train_episode(self, env, epsilon):
        self.agent.epsilon = epsilon
        env.reset()
        state = self._features(env)
        done = False
        while not done:
            action = self.agent.act(state, training=True)
            _, reward, done, _ = env.step(action)
            next_state = self._features(env)
            self.agent.remember(state, action, reward, next_state, done)
            self.agent.train_step()
            state = next_state

    def q_values_for_features(self, features):
        return self.agent.q_values(features)


def run_decorrelate_arm(buffer_size, validity, seed):
    env = make_env(validity, seed)
    agent = ReplayBufferAgentWrapper(seed, buffer_size)
    for epsilon in epsilon_schedule(ACQUISITION_EPISODES):
        agent.train_episode(env, epsilon)
    acq_cds, acq_dep = read_metrics(AGENT_TYPE, agent, validity)

    env.decorrelate()
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            agent.train_episode(env, INTERVENTION_EPSILON)
    final_cds, final_dep = read_metrics(AGENT_TYPE, agent, validity)

    return {
        "label": "decorrelate",
        "buffer_size": buffer_size,
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_proxy_dependence": round(final_dep, 6),
    }


def run_naive_arm(buffer_size, validity, seed):
    env = make_env(validity, seed)
    env.decorrelate()
    agent = ReplayBufferAgentWrapper(seed + 1_000_000, buffer_size)
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            agent.train_episode(env, INTERVENTION_EPSILON)
    final_cds, final_dep = read_metrics(AGENT_TYPE, agent, validity)
    return {
        "label": "naive",
        "buffer_size": buffer_size,
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": None,
        "final_proxy_dependence": round(final_dep, 6),
    }


def mean(values):
    values = list(values)
    return sum(values) / max(1, len(values))


def gap_values(summaries, buffer_size, validity, label):
    return [
        abs(row["final_proxy_dependence"])
        for row in summaries
        if row["buffer_size"] == buffer_size and row["validity"] == validity and row["label"] == label
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

    for buffer_size in BUFFER_SIZES:
        for validity in (0.5, 0.95):
            for seed in SEEDS:
                print(f"buffer={buffer_size} v{validity} decorrelate seed={seed}", flush=True)
                summaries.append(run_decorrelate_arm(buffer_size, validity, seed))

                print(f"buffer={buffer_size} v{validity} naive seed={seed}", flush=True)
                summaries.append(run_naive_arm(buffer_size, validity, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_replay_buffer_sweep_summaries.csv"),
        summaries,
        ["label", "buffer_size", "validity", "seed", "acquisition_proxy_dependence", "final_proxy_dependence"],
    )

    results = []
    for buffer_size in BUFFER_SIZES:
        for validity in (0.5, 0.95):
            decorrelate = gap_values(summaries, buffer_size, validity, "decorrelate")
            naive = gap_values(summaries, buffer_size, validity, "naive")
            test = compare_groups(decorrelate, naive)
            d = cohens_d(decorrelate, naive)
            results.append(
                {
                    "buffer_size": buffer_size,
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
        os.path.join(RESULTS_DIR, "cue_validity_replay_buffer_sweep_criteria.csv"),
        results,
        ["buffer_size", "validity", "decorrelate_mean", "naive_mean", "t_stat", "p_value", "cohens_d", "has_effect"],
    )

    staleness_excluded = all(r["has_effect"] for r in results)

    lines = [
        "# Cue Validity Replay Buffer Size Sweep",
        "",
        "## Pre-registration",
        "Design and criteria were written and committed in docs/cue_validity_replay_buffer_sweep.md before any of the data below was collected.",
        "",
        "## Design summary",
        f"linear_raw only, replay_size in {BUFFER_SIZES}, cue_validity in [0.5, 0.95], {len(SEEDS)} seeds, decorrelate intervention only.",
        "",
        "## Results",
        "",
        "| buffer size | validity | decorrelate | naive | t | p | Cohen's d | has effect |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['buffer_size']} | {row['validity']} | {row['decorrelate_mean']:.3f} | {row['naive_mean']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['has_effect']} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            f"**Staleness exclusion holds across all tested buffer sizes: {staleness_excluded}.**",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_replay_buffer_sweep_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
