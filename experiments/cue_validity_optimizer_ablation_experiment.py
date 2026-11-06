"""Cue Validity Optimizer Ablation: SGD (no momentum) vs. Adam.

Pre-registered design: docs/cue_validity_optimizer_ablation.md. Reuses
experiments/cue_validity_nonlinear_experiment.py's run_pretrained_arm /
run_naive_arm, features_for, read_metrics verbatim, substituting a
momentum-free SGD wrapper for NeuralAgentWrapper in the two neural
conditions only.
"""

import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim

from analysis.stats import compare_groups, cohens_d
from experiments.cue_validity_nonlinear_experiment import (
    BATCH_SIZE,
    EPSILON_START,
    FEATURE_DIM,
    GAMMA,
    LEARNING_RATE,
    MLPQNet,
    REPLAY_BUFFER_SIZE,
    SEEDS,
    TARGET_UPDATE_INTERVAL,
    TRAIN_EVERY_STEPS,
    VALIDITY_LEVELS,
    make_env,
    mean,
    read_metrics,
    train_episode,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
ADAM_SUMMARY_CSV = os.path.join(RESULTS_DIR, "cue_validity_nonlinear_summaries.csv")

NEURAL_CONDITIONS = ["neural_raw", "neural_interaction_only"]
from collections import deque


class SGDNeuralAgentWrapper:
    """Identical to NeuralAgentWrapper except for the optimizer: plain SGD
    with momentum=0.0 in place of Adam, at the same learning rate."""

    def __init__(self, seed, feature_dim):
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.device = torch.device("cpu")
        self.online = MLPQNet(feature_dim).to(self.device)
        self.target = MLPQNet(feature_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.SGD(self.online.parameters(), lr=LEARNING_RATE, momentum=0.0)
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


def make_sgd_agent(condition, seed):
    return SGDNeuralAgentWrapper(seed, FEATURE_DIM[condition])


def run_pretrained_arm_sgd(condition, validity, seed):
    env = make_env(validity, seed)
    agent = make_sgd_agent(condition, seed)
    from experiments.cue_validity_nonlinear_experiment import epsilon_schedule, ACQUISITION_EPISODES, NUM_BLOCKS, BLOCK_SIZE, INTERVENTION_EPSILON

    for epsilon in epsilon_schedule(ACQUISITION_EPISODES):
        train_episode(condition, agent, env, epsilon)
    acq_cds, acq_dep = read_metrics(condition, agent, validity)

    env.decorrelate()
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(condition, agent, env, INTERVENTION_EPSILON)
    final_cds, final_dep = read_metrics(condition, agent, validity)

    return {
        "label": "decorrelate",
        "condition": condition,
        "validity": validity,
        "seed": seed,
        "acquisition_cds": round(acq_cds, 6),
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_cds": round(final_cds, 6),
        "final_proxy_dependence": round(final_dep, 6),
    }


def run_naive_arm_sgd(condition, validity, seed):
    env = make_env(validity, seed)
    env.decorrelate()
    agent = make_sgd_agent(condition, seed + 1_000_000)
    from experiments.cue_validity_nonlinear_experiment import NUM_BLOCKS, BLOCK_SIZE, INTERVENTION_EPSILON

    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(condition, agent, env, INTERVENTION_EPSILON)
    final_cds, final_dep = read_metrics(condition, agent, validity)
    return {
        "label": "naive",
        "condition": condition,
        "validity": validity,
        "seed": seed,
        "acquisition_cds": None,
        "acquisition_proxy_dependence": None,
        "final_cds": round(final_cds, 6),
        "final_proxy_dependence": round(final_dep, 6),
    }


def load_adam_gap(condition, validity):
    with open(ADAM_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["validity"] = float(row["validity"])
        row["final_proxy_dependence"] = (
            None if row["final_proxy_dependence"] in ("", "None") else float(row["final_proxy_dependence"])
        )
    decorrelate = [
        abs(row["final_proxy_dependence"])
        for row in rows
        if row["condition"] == condition and row["validity"] == validity and row["label"] == "decorrelate"
    ]
    naive = [
        abs(row["final_proxy_dependence"])
        for row in rows
        if row["condition"] == condition and row["validity"] == validity and row["label"] == "naive"
    ]
    return decorrelate, naive


def sgd_gap(summaries, condition, validity):
    decorrelate = [
        abs(row["final_proxy_dependence"])
        for row in summaries
        if row["condition"] == condition and row["validity"] == validity and row["label"] == "decorrelate"
    ]
    naive = [
        abs(row["final_proxy_dependence"])
        for row in summaries
        if row["condition"] == condition and row["validity"] == validity and row["label"] == "naive"
    ]
    return decorrelate, naive


def gap_summary(decorrelate, naive):
    test = compare_groups(decorrelate, naive)
    d = cohens_d(decorrelate, naive)
    return {
        "decorrelate_mean": round(mean(decorrelate), 6),
        "naive_mean": round(mean(naive), 6),
        "gap": round(mean(decorrelate) - mean(naive), 6),
        "t_stat": round(float(test["t_stat"]), 6),
        "p_value": round(float(test["p_value"]), 6),
        "cohens_d": round(float(d), 6),
        "has_effect": bool(test["p_value"] < 0.05 and mean(decorrelate) > mean(naive)),
    }


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

    for condition in NEURAL_CONDITIONS:
        for validity in VALIDITY_LEVELS:
            for seed in SEEDS:
                print(f"SGD {condition} v{validity} decorrelate seed={seed}", flush=True)
                summaries.append(run_pretrained_arm_sgd(condition, validity, seed))

                print(f"SGD {condition} v{validity} naive seed={seed}", flush=True)
                summaries.append(run_naive_arm_sgd(condition, validity, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_optimizer_ablation_summaries.csv"),
        summaries,
        ["label", "condition", "validity", "seed", "acquisition_cds", "acquisition_proxy_dependence", "final_cds", "final_proxy_dependence"],
    )

    primary_rows = []
    secondary_rows = []
    for condition in NEURAL_CONDITIONS:
        for validity in VALIDITY_LEVELS:
            sgd_decorrelate, sgd_naive = sgd_gap(summaries, condition, validity)
            sgd_result = gap_summary(sgd_decorrelate, sgd_naive)
            sgd_result.update({"condition": condition, "validity": validity, "optimizer": "sgd"})
            primary_rows.append(sgd_result)

            adam_decorrelate, adam_naive = load_adam_gap(condition, validity)
            adam_result = gap_summary(adam_decorrelate, adam_naive)

            compare_test = compare_groups(sgd_decorrelate, adam_decorrelate)
            compare_d = cohens_d(sgd_decorrelate, adam_decorrelate)
            secondary_rows.append(
                {
                    "condition": condition,
                    "validity": validity,
                    "sgd_decorrelate_mean": sgd_result["decorrelate_mean"],
                    "adam_decorrelate_mean": adam_result["decorrelate_mean"],
                    "t_stat": round(float(compare_test["t_stat"]), 6),
                    "p_value": round(float(compare_test["p_value"]), 6),
                    "cohens_d": round(float(compare_d), 6),
                }
            )

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_optimizer_ablation_primary.csv"),
        primary_rows,
        ["condition", "validity", "optimizer", "decorrelate_mean", "naive_mean", "gap", "t_stat", "p_value", "cohens_d", "has_effect"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_optimizer_ablation_secondary.csv"),
        secondary_rows,
        ["condition", "validity", "sgd_decorrelate_mean", "adam_decorrelate_mean", "t_stat", "p_value", "cohens_d"],
    )

    lines = [
        "# Cue Validity Optimizer Ablation (SGD vs. Adam)",
        "",
        "## Pre-registration",
        "Design, metrics, and success/failure criteria were written and committed in docs/cue_validity_optimizer_ablation.md before any of the data below was collected. This experiment is load-bearing for the publication plan's own Go/No-Go criteria (Section 12): an SGD effect loss where Adam showed one is grounds to abandon or pivot the neural portion of the claim.",
        "",
        "## Primary test: does the decorrelate-vs-naive effect survive under SGD (no momentum, same learning rate)?",
        "",
        "| condition | validity | SGD decorrelate | SGD naive | gap | t | p | Cohen's d | effect under SGD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in primary_rows:
        lines.append(
            f"| {row['condition']} | {row['validity']} | {row['decorrelate_mean']:.3f} | {row['naive_mean']:.3f} | {row['gap']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['has_effect']} |"
        )

    neural_raw_sgd_rows = [r for r in primary_rows if r["condition"] == "neural_raw"]
    neural_raw_retains = all(r["has_effect"] for r in neural_raw_sgd_rows)
    neural_raw_loses_any = any(not r["has_effect"] for r in neural_raw_sgd_rows)

    lines.extend(
        [
            "",
            "## Secondary: SGD vs. Adam decorrelate-arm magnitude (direct comparison, not the primary verdict)",
            "",
            "| condition | validity | SGD decorrelate mean | Adam decorrelate mean | t | p | Cohen's d |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in secondary_rows:
        lines.append(
            f"| {row['condition']} | {row['validity']} | {row['sgd_decorrelate_mean']:.3f} | {row['adam_decorrelate_mean']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            f"neural_raw retains a significant decorrelate-vs-naive effect under SGD at both validity levels: {neural_raw_retains}.",
            f"neural_raw loses the effect under SGD at at least one validity level: {neural_raw_loses_any}.",
            "neural_interaction_only had no significant Adam effect to begin with; its SGD result is reported above but is not informative for the momentum question either way.",
            "",
            "## Limitations",
            "The learning rate (1e-3) was not re-tuned for SGD, deliberately, to isolate momentum as the single changed variable -- this means a differently-tuned SGD configuration could behave differently, and this experiment cannot rule that out. Only one architecture and one momentum value (0.0, i.e. plain SGD) were tested, not a momentum sweep.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_optimizer_ablation_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
