"""Capacity sweep: neural_raw decorrelate-vs-naive gap across MLP widths.

Pre-registered design: docs/cue_validity_capacity_sweep.md. Reuses
experiments/cue_validity_nonlinear_experiment.py's MLPQNet, make_env,
read_metrics, train_episode, epsilon_schedule unchanged; only
hidden_size varies (the existing NeuralAgentWrapper hardcodes 32).
"""

import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from analysis.stats import compare_groups, cohens_d
from experiments.cue_validity_nonlinear_experiment import (
    ACQUISITION_EPISODES,
    BATCH_SIZE,
    EPSILON_START,
    GAMMA,
    INTERVENTION_EPSILON,
    LEARNING_RATE,
    MLPQNet,
    NUM_BLOCKS,
    BLOCK_SIZE,
    REPLAY_BUFFER_SIZE,
    SEEDS,
    TARGET_UPDATE_INTERVAL,
    TRAIN_EVERY_STEPS,
    VALIDITY_LEVELS,
    epsilon_schedule,
    make_env,
    mean,
    read_metrics,
    train_episode,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

CONDITION = "neural_raw"
WIDTHS = [8, 16, 32, 64, 128]


class CapacityAgentWrapper:
    """Identical to NeuralAgentWrapper except hidden_size is a parameter."""

    def __init__(self, seed, feature_dim, hidden_size):
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.device = torch.device("cpu")
        self.online = MLPQNet(feature_dim, hidden_size=hidden_size).to(self.device)
        self.target = MLPQNet(feature_dim, hidden_size=hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.epsilon = EPSILON_START
        self.env_steps = 0

    def act(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(4)
        with torch.no_grad():
            x = torch.tensor([state], dtype=torch.float32, device=self.device)
            values = self.online(x).cpu().tolist()[0]
        best = max(values)
        best_actions = [a for a in range(4) if abs(values[a] - best) < 1e-9]
        return self.rng.choice(best_actions)

    def remember(self, state, action, reward, next_state, done):
        self.replay.append((state, action, reward, next_state, done))

    def train_step(self):
        self.env_steps += 1
        if self.env_steps % TRAIN_EVERY_STEPS != 0 or len(self.replay) < BATCH_SIZE:
            return
        batch = [self.replay[self.rng.randrange(len(self.replay))] for _ in range(BATCH_SIZE)]
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        q_pred = self.online(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            q_next = self.target(next_states_t).max(dim=1).values
            q_target = rewards_t + (1.0 - dones_t) * GAMMA * q_next
        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        if self.env_steps % TARGET_UPDATE_INTERVAL == 0:
            self.target.load_state_dict(self.online.state_dict())

    def q_values_for_features(self, features):
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32, device=self.device)
            return self.online(x).cpu().tolist()[0]


def run_decorrelate_arm(width, validity, seed):
    env = make_env(validity, seed)
    agent = CapacityAgentWrapper(seed, 6, width)
    for epsilon in epsilon_schedule(ACQUISITION_EPISODES):
        train_episode(CONDITION, agent, env, epsilon)
    acq_cds, acq_dep = read_metrics(CONDITION, agent, validity)

    env.decorrelate()
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(CONDITION, agent, env, INTERVENTION_EPSILON)
    final_cds, final_dep = read_metrics(CONDITION, agent, validity)

    return {
        "label": "decorrelate",
        "width": width,
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_proxy_dependence": round(final_dep, 6),
    }


def run_naive_arm(width, validity, seed):
    env = make_env(validity, seed)
    env.decorrelate()
    agent = CapacityAgentWrapper(seed + 1_000_000, 6, width)
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(CONDITION, agent, env, INTERVENTION_EPSILON)
    final_cds, final_dep = read_metrics(CONDITION, agent, validity)
    return {
        "label": "naive",
        "width": width,
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": None,
        "final_proxy_dependence": round(final_dep, 6),
    }


def gap_values(summaries, width, validity, label):
    return [
        abs(row["final_proxy_dependence"])
        for row in summaries
        if row["width"] == width and row["validity"] == validity and row["label"] == label
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

    for width in WIDTHS:
        for validity in VALIDITY_LEVELS:
            for seed in SEEDS:
                print(f"width={width} v{validity} decorrelate seed={seed}", flush=True)
                summaries.append(run_decorrelate_arm(width, validity, seed))

                print(f"width={width} v{validity} naive seed={seed}", flush=True)
                summaries.append(run_naive_arm(width, validity, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_capacity_sweep_summaries.csv"),
        summaries,
        ["label", "width", "validity", "seed", "acquisition_proxy_dependence", "final_proxy_dependence"],
    )

    results = []
    for width in WIDTHS:
        for validity in VALIDITY_LEVELS:
            decorrelate = gap_values(summaries, width, validity, "decorrelate")
            naive = gap_values(summaries, width, validity, "naive")
            test = compare_groups(decorrelate, naive)
            d = cohens_d(decorrelate, naive)
            results.append(
                {
                    "width": width,
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
        os.path.join(RESULTS_DIR, "cue_validity_capacity_sweep_criteria.csv"),
        results,
        ["width", "validity", "decorrelate_mean", "naive_mean", "t_stat", "p_value", "cohens_d", "has_effect"],
    )

    lines = [
        "# Cue Validity Capacity Sweep",
        "",
        "## Pre-registration",
        "Design and criteria were written and committed in docs/cue_validity_capacity_sweep.md before any of the data below was collected.",
        "",
        "## Design summary",
        f"neural_raw only, hidden_size in {WIDTHS}, cue_validity in {VALIDITY_LEVELS}, {len(SEEDS)} seeds, decorrelate intervention only.",
        "",
        "## Capacity curve",
        "",
        "| width | validity | decorrelate | naive | t | p | Cohen's d | has effect |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['width']} | {row['validity']} | {row['decorrelate_mean']:.3f} | {row['naive_mean']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['has_effect']} |"
        )

    for validity in VALIDITY_LEVELS:
        d_by_width = [(r["width"], r["cohens_d"]) for r in results if r["validity"] == validity]
        d_by_width.sort()
        ds = [d for _, d in d_by_width]
        monotonic_non_increasing = all(ds[i] >= ds[i + 1] - 0.3 for i in range(len(ds) - 1))
        lines.append("")
        lines.append(f"Cohen's d by width at validity={validity}: {d_by_width}. Roughly monotonic non-increasing (allowing noise): {monotonic_non_increasing}.")

    with open(os.path.join(REPORTS_DIR, "cue_validity_capacity_sweep_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
