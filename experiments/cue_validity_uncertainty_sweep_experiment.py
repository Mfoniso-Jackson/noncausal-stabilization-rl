"""Cue Validity Uncertainty Sweep: a properly controlled test of RQ3
("does uncertainty increase non-causal stabilization"), fixing the
confound in the pre-existing GridCueEnv/run_pipeline.py uncertainty
pipeline documented in docs/symbolic_stabilization_audit.md Section 2.

Pre-registered design: docs/cue_validity_uncertainty_sweep.md. Reuses
agents/dqn_agent.py::DQNAgent and experiments/cue_validity_proxy_dependence_experiment.py's
LinearAgentWrapper, probe_q_values/read_metrics, and epsilon_schedule
unchanged; the only new ingredient is CueValidityEnv's reward_noise
parameter, which is an environment property only and is never read by or
passed to the agent at any call site below.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import compare_groups, cohens_d
from env.cue_validity_env import CueValidityEnv
from experiments.cue_validity_proxy_dependence_experiment import (
    TARGET_PHASE,
    LinearAgentWrapper,
    epsilon_schedule,
    mean,
    read_metrics,
    train_episode,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

NOISE_LEVELS = [0.0, 0.3, 0.6]
VALIDITY_LEVELS = [0.0, 0.95]
SEEDS = list(range(15))
ACQUISITION_EPISODES = 3000
INTERVENTION_EPISODES = 1500
BLOCK_SIZE = 50
NUM_BLOCKS = INTERVENTION_EPISODES // BLOCK_SIZE
INTERVENTION_EPSILON = 0.04


def make_env(validity, reward_noise, seed):
    # reward_noise is an environment constructor argument only; it is never
    # read by, or passed to, the agent at any call site in this file.
    return CueValidityEnv(validity, cue_phase=TARGET_PHASE, target_phase=TARGET_PHASE, reward_noise=reward_noise, seed=seed)


def run_decorrelate_arm(validity, reward_noise, seed):
    env = make_env(validity, reward_noise, seed)
    agent = LinearAgentWrapper(seed, orthogonal=False)
    for epsilon in epsilon_schedule(ACQUISITION_EPISODES):
        train_episode("linear_raw", agent, env, epsilon)
    _, acq_dep = read_metrics("linear_raw", agent, validity)

    env.decorrelate()
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode("linear_raw", agent, env, INTERVENTION_EPSILON)
    _, final_dep = read_metrics("linear_raw", agent, validity)

    return {
        "label": "decorrelate",
        "validity": validity,
        "reward_noise": reward_noise,
        "seed": seed,
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_proxy_dependence": round(final_dep, 6),
    }


def run_naive_arm(validity, reward_noise, seed):
    env = make_env(validity, reward_noise, seed)
    env.decorrelate()
    agent = LinearAgentWrapper(seed + 1_000_000, orthogonal=False)
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode("linear_raw", agent, env, INTERVENTION_EPSILON)
    _, final_dep = read_metrics("linear_raw", agent, validity)
    return {
        "label": "naive",
        "validity": validity,
        "reward_noise": reward_noise,
        "seed": seed,
        "acquisition_proxy_dependence": None,
        "final_proxy_dependence": round(final_dep, 6),
    }


def holm_bonferroni(pvals):
    n = len(pvals)
    indexed = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [None] * n
    running_max = 0.0
    for rank, i in enumerate(indexed):
        m = n - rank
        adj = pvals[i] * m
        running_max = max(running_max, adj)
        adjusted[i] = min(1.0, running_max)
    return [adjusted[i] for i in range(n)]


def gap_values(summaries, validity, reward_noise, label):
    return [
        abs(r["final_proxy_dependence"])
        for r in summaries
        if r["validity"] == validity and r["reward_noise"] == reward_noise and r["label"] == label
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

    for reward_noise in NOISE_LEVELS:
        for validity in VALIDITY_LEVELS:
            for seed in SEEDS:
                print(f"uncertainty-sweep noise={reward_noise} v{validity} decorrelate seed={seed}", flush=True)
                summaries.append(run_decorrelate_arm(validity, reward_noise, seed))
                print(f"uncertainty-sweep noise={reward_noise} v{validity} naive seed={seed}", flush=True)
                summaries.append(run_naive_arm(validity, reward_noise, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_uncertainty_sweep_summaries.csv"),
        summaries,
        ["label", "validity", "reward_noise", "seed", "acquisition_proxy_dependence", "final_proxy_dependence"],
    )

    results = []
    for reward_noise in NOISE_LEVELS:
        for validity in VALIDITY_LEVELS:
            decorrelate = gap_values(summaries, validity, reward_noise, "decorrelate")
            naive = gap_values(summaries, validity, reward_noise, "naive")
            test = compare_groups(decorrelate, naive)
            d = cohens_d(decorrelate, naive)
            results.append(
                {
                    "reward_noise": reward_noise,
                    "validity": validity,
                    "decorrelate_mean": round(mean(decorrelate), 6),
                    "naive_mean": round(mean(naive), 6),
                    "t_stat": round(float(test["t_stat"]), 6),
                    "p_value": round(float(test["p_value"]), 6),
                    "cohens_d": round(float(d), 6),
                }
            )

    results.sort(key=lambda r: (r["reward_noise"], r["validity"]))
    adjusted = holm_bonferroni([r["p_value"] for r in results])
    for r, adj in zip(results, adjusted):
        r["holm_adjusted_p"] = round(adj, 6)
        r["has_effect"] = bool(adj < 0.05 and r["decorrelate_mean"] > r["naive_mean"])

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_uncertainty_sweep_criteria.csv"),
        results,
        ["reward_noise", "validity", "decorrelate_mean", "naive_mean", "t_stat", "p_value", "holm_adjusted_p", "cohens_d", "has_effect"],
    )

    def cell(noise, validity):
        return next(r for r in results if r["reward_noise"] == noise and r["validity"] == validity)

    primary_low = cell(0.0, 0.95)
    primary_high = cell(0.6, 0.95)
    primary_supported = primary_high["cohens_d"] > primary_low["cohens_d"]
    floor_clean = all(not cell(n, 0.0)["has_effect"] for n in NOISE_LEVELS)

    lines = [
        "# Cue Validity Uncertainty Sweep",
        "",
        "## Pre-registration",
        "Design and criteria were written and committed in docs/cue_validity_uncertainty_sweep.md before any of the data below was collected. reward_noise is an environment property only (env/cue_validity_env.py); no agent setting is ever a function of it. Holm-Bonferroni correction applied across the full 6-cell family (3 noise levels x 2 validity levels).",
        "",
        "## Results",
        "",
        "| reward_noise | validity | decorrelate | naive | t | raw p | holm p | Cohen's d | has effect |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['reward_noise']} | {r['validity']} | {r['decorrelate_mean']:.4f} | {r['naive_mean']:.4f} | "
            f"{r['t_stat']:.3f} | {r['p_value']:.4f} | {r['holm_adjusted_p']:.4f} | {r['cohens_d']:.3f} | {r['has_effect']} |"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            f"Primary pre-registered comparison (v=0.95, Cohen's d at noise=0.6 vs. noise=0.0): {primary_high['cohens_d']:.3f} vs. {primary_low['cohens_d']:.3f}. Supported (uncertainty increases persistence): {primary_supported}.",
            f"Floor control (v=0.0) remains null at every noise level, as required for interpretability: {floor_clean}.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_uncertainty_sweep_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return results


if __name__ == "__main__":
    run()
