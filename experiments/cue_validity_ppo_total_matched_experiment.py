"""Total-episode-matched PPO retest: the methodologically correct
comparison for an algorithm whose proxy_dependence readout is
training-length-dependent via entropy collapse (see
docs/cue_validity_ppo.md).

Pre-registered: docs/cue_validity_ppo.md, "Total-episode-matched
retest." Reuses the already-collected decorrelate arm from
cue_validity_ppo_retest_experiment.py (results/cue_validity_ppo_retest_summaries.csv)
unchanged; only the naive arm is rerun, now trained for the same total
episode count (2300 = 800 acquisition + 1500 intervention) as the
pretrained arm, entirely in the decorrelated environment.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import PPOAgent
from analysis.stats import bootstrap_diff_ci, compare_groups, cohens_d
from experiments.cue_validity_proxy_dependence_experiment import SEEDS, make_env, mean
from experiments.cue_validity_ppo_experiment import read_metrics, run_rollout_block

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
RETEST_SUMMARY_CSV = os.path.join(RESULTS_DIR, "cue_validity_ppo_retest_summaries.csv")

VALIDITY_LEVELS = [0.5, 0.95]
BLOCK_SIZE = 50
TOTAL_EPISODES = 2300  # 800 acquisition + 1500 intervention, matching the decorrelate arm
TOTAL_BLOCKS = TOTAL_EPISODES // BLOCK_SIZE


def run_naive_arm_total_matched(validity, seed):
    env = make_env(validity, seed)
    env.decorrelate()
    agent = PPOAgent(feature_dim=6, seed=seed + 1_000_000)
    for _ in range(TOTAL_BLOCKS):
        run_rollout_block(agent, env, BLOCK_SIZE)
    _, final_dep = read_metrics(agent, validity)
    return {
        "label": "naive_total_matched",
        "validity": validity,
        "seed": seed,
        "final_proxy_dependence": round(final_dep, 6),
    }


def load_decorrelate_gap(validity):
    with open(RETEST_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["validity"] = float(row["validity"])
        row["final_proxy_dependence"] = (
            None if row["final_proxy_dependence"] in ("", "None") else float(row["final_proxy_dependence"])
        )
    return [abs(r["final_proxy_dependence"]) for r in rows if r["validity"] == validity and r["label"] == "decorrelate"]


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    naive_summaries = []

    for validity in VALIDITY_LEVELS:
        for seed in SEEDS:
            print(f"ppo total-matched naive v{validity} seed={seed}", flush=True)
            naive_summaries.append(run_naive_arm_total_matched(validity, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_ppo_total_matched_naive_summaries.csv"),
        naive_summaries,
        ["label", "validity", "seed", "final_proxy_dependence"],
    )

    results = []
    for validity in VALIDITY_LEVELS:
        decorrelate = load_decorrelate_gap(validity)
        naive = [abs(r["final_proxy_dependence"]) for r in naive_summaries if r["validity"] == validity]
        test = compare_groups(decorrelate, naive)
        d = cohens_d(decorrelate, naive)
        boot_lo, boot_hi, boot_point = bootstrap_diff_ci(decorrelate, naive, seed=0)
        results.append(
            {
                "validity": validity,
                "decorrelate_mean": round(mean(decorrelate), 6),
                "naive_total_matched_mean": round(mean(naive), 6),
                "t_stat": round(float(test["t_stat"]), 6),
                "p_value": round(float(test["p_value"]), 6),
                "cohens_d": round(float(d), 6),
                "bootstrap_diff_lo": round(boot_lo, 6),
                "bootstrap_diff_hi": round(boot_hi, 6),
                "has_effect": bool(test["p_value"] < 0.05 and mean(decorrelate) > mean(naive)),
            }
        )

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_ppo_total_matched_criteria.csv"),
        results,
        ["validity", "decorrelate_mean", "naive_total_matched_mean", "t_stat", "p_value", "cohens_d", "bootstrap_diff_lo", "bootstrap_diff_hi", "has_effect"],
    )

    any_effect = any(r["has_effect"] for r in results)
    lines = [
        "# PPO Total-Episode-Matched Retest",
        "",
        "## Pre-registration",
        "Design and criteria were written and committed in docs/cue_validity_ppo.md (\"Total-episode-matched retest\") before this data was collected. Reuses the already-collected decorrelate arm (2300 total episodes); only naive is rerun, now also trained for 2300 total episodes entirely in the decorrelated environment.",
        "",
        "## Results",
        "",
        "| validity | decorrelate | naive (total-matched) | t | p | Cohen's d | bootstrap diff CI | has effect |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['validity']} | {row['decorrelate_mean']:.3f} | {row['naive_total_matched_mean']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | "
            f"[{row['bootstrap_diff_lo']:.3f}, {row['bootstrap_diff_hi']:.3f}] | {row['has_effect']} |"
        )
    lines.extend(
        [
            "",
            "## Verdict (H4, total-episode-matched, methodologically corrected test)",
            f"PPO shows the linear_raw/neural_raw pattern once total episode count is properly matched: {any_effect}.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_ppo_total_matched_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
