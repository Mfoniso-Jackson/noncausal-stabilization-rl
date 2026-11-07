"""Properly-powered PPO retest at a fixed, diagnostic-informed acquisition
budget.

Pre-registered design: docs/cue_validity_ppo.md, "Properly-powered
retest" section -- written before this data was collected, after a
diagnostic run (not a pre-registered experiment) showed that PPO's
proxy_dependence follows an inverted-U trajectory driven by policy
entropy's collapse toward convergence, peaking around 550-1050 episodes
across seeds/validity, not growing monotonically with more training.
Reuses experiments/cue_validity_ppo_experiment.py's rollout/probe/metric
functions unchanged; only ACQUISITION_EPISODES differs (800, fixed in
advance, not tuned per seed).
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import PPOAgent
from analysis.stats import bootstrap_diff_ci, compare_groups, cohens_d
from experiments.cue_validity_proxy_dependence_experiment import NUM_BLOCKS, SEEDS, make_env, mean
from experiments.cue_validity_ppo_experiment import probe_logits, read_metrics, run_rollout_block
from metrics.cue_validity import policy_entropy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

VALIDITY_LEVELS = [0.5, 0.95]
CONDITION = "ppo_retest"
ACQUISITION_EPISODES = 800  # fixed from the diagnostic, before running this experiment
BLOCK_SIZE = 50
ACQUISITION_BLOCKS = ACQUISITION_EPISODES // BLOCK_SIZE


def run_decorrelate_arm(validity, seed):
    env = make_env(validity, seed)
    agent = PPOAgent(feature_dim=6, seed=seed)

    for _ in range(ACQUISITION_BLOCKS):
        run_rollout_block(agent, env, BLOCK_SIZE)
    acq_cds, acq_dep = read_metrics(agent, validity)
    q1 = probe_logits(agent, validity, True)
    acq_entropy = policy_entropy(q1, (0, 1, 2, 3))

    env.decorrelate()
    for _ in range(NUM_BLOCKS):
        run_rollout_block(agent, env, BLOCK_SIZE)
    final_cds, final_dep = read_metrics(agent, validity)
    q1_final = probe_logits(agent, validity, True)
    final_entropy = policy_entropy(q1_final, (0, 1, 2, 3))

    return {
        "label": "decorrelate",
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "acquisition_entropy": round(acq_entropy, 6),
        "final_proxy_dependence": round(final_dep, 6),
        "final_entropy": round(final_entropy, 6),
    }


def run_naive_arm(validity, seed):
    env = make_env(validity, seed)
    env.decorrelate()
    agent = PPOAgent(feature_dim=6, seed=seed + 1_000_000)
    for _ in range(NUM_BLOCKS):
        run_rollout_block(agent, env, BLOCK_SIZE)
    final_cds, final_dep = read_metrics(agent, validity)
    q1_final = probe_logits(agent, validity, True)
    final_entropy = policy_entropy(q1_final, (0, 1, 2, 3))
    return {
        "label": "naive",
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": None,
        "acquisition_entropy": None,
        "final_proxy_dependence": round(final_dep, 6),
        "final_entropy": round(final_entropy, 6),
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
            print(f"ppo_retest v{validity} decorrelate seed={seed}", flush=True)
            summaries.append(run_decorrelate_arm(validity, seed))
            print(f"ppo_retest v{validity} naive seed={seed}", flush=True)
            summaries.append(run_naive_arm(validity, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_ppo_retest_summaries.csv"),
        summaries,
        ["label", "validity", "seed", "acquisition_proxy_dependence", "acquisition_entropy", "final_proxy_dependence", "final_entropy"],
    )

    results = []
    for validity in VALIDITY_LEVELS:
        decorrelate = gap_values(summaries, validity, "decorrelate")
        naive = gap_values(summaries, validity, "naive")
        test = compare_groups(decorrelate, naive)
        d = cohens_d(decorrelate, naive)
        boot_lo, boot_hi, boot_point = bootstrap_diff_ci(decorrelate, naive, seed=0)
        results.append(
            {
                "validity": validity,
                "decorrelate_mean": round(mean(decorrelate), 6),
                "naive_mean": round(mean(naive), 6),
                "t_stat": round(float(test["t_stat"]), 6),
                "p_value": round(float(test["p_value"]), 6),
                "cohens_d": round(float(d), 6),
                "bootstrap_diff_lo": round(boot_lo, 6),
                "bootstrap_diff_hi": round(boot_hi, 6),
                "has_effect": bool(test["p_value"] < 0.05 and mean(decorrelate) > mean(naive)),
            }
        )

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_ppo_retest_criteria.csv"),
        results,
        ["validity", "decorrelate_mean", "naive_mean", "t_stat", "p_value", "cohens_d", "bootstrap_diff_lo", "bootstrap_diff_hi", "has_effect"],
    )

    def mean_field(label, validity, field):
        values = [r[field] for r in summaries if r["label"] == label and r["validity"] == validity and r[field] is not None]
        return mean(values) if values else float("nan")

    lines = [
        "# PPO Retest: Fixed 800-Episode Acquisition Budget",
        "",
        "## Pre-registration",
        "Design and criteria were written and committed in docs/cue_validity_ppo.md (\"Properly-powered retest\") before this data was collected, following a diagnostic (not pre-registered) run that found PPO's proxy_dependence follows an inverted-U trajectory, not monotonic growth, driven by policy entropy's collapse toward convergence.",
        "",
        "## Results",
        "",
        "| validity | decorrelate | naive | t | p | Cohen's d | bootstrap diff CI | has effect |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['validity']} | {row['decorrelate_mean']:.3f} | {row['naive_mean']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | "
            f"[{row['bootstrap_diff_lo']:.3f}, {row['bootstrap_diff_hi']:.3f}] | {row['has_effect']} |"
        )

    lines.extend(
        [
            "",
            "## Acquisition-phase diagnostics (mean across seeds, at episode 800)",
            "",
            "| validity | acquisition proxy_dependence | acquisition entropy |",
            "|---:|---:|---:|",
        ]
    )
    for validity in VALIDITY_LEVELS:
        acq_dep = mean_field("decorrelate", validity, "acquisition_proxy_dependence")
        acq_ent = mean_field("decorrelate", validity, "acquisition_entropy")
        lines.append(f"| {validity} | {acq_dep:.4f} | {acq_ent:.4f} |")

    lines.extend(
        [
            "",
            "## Naive-arm entropy (diagnoses whether the naive arm is confounded by its own entropy-collapse trajectory)",
            "",
            "| validity | naive final proxy_dependence | naive final entropy |",
            "|---:|---:|---:|",
        ]
    )
    for validity in VALIDITY_LEVELS:
        naive_dep = mean_field("naive", validity, "final_proxy_dependence")
        naive_ent = mean_field("naive", validity, "final_entropy")
        lines.append(f"| {validity} | {naive_dep:.4f} | {naive_ent:.4f} |")

    any_effect = any(r["has_effect"] for r in results)
    lines.extend(
        [
            "",
            "## Verdict (H4, properly-powered retest)",
            f"PPO shows the linear_raw/neural_raw pattern at the fixed, diagnostic-informed acquisition budget: {any_effect}.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_ppo_retest_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
