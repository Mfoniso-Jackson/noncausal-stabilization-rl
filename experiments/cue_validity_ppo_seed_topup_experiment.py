"""PPO seed top-up for the total-episode-matched persistence comparison.

Pre-registered: docs/cue_validity_ppo.md, "Seed top-up" section, committed
before this script was run. Adds 15 new seeds (15-29) to both arms of the
total-episode-matched comparison, following the same escalation convention
as experiments/cue_validity_neural_raw_seed_topup.py. Combines with the
existing 15 seeds (0-14) from results/cue_validity_ppo_retest_summaries.csv
(decorrelate arm) and results/cue_validity_ppo_total_matched_naive_summaries.csv
(naive_total_matched arm) for a final n=30 per arm per validity level.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import bootstrap_diff_ci, compare_groups, cohens_d
from experiments.cue_validity_proxy_dependence_experiment import mean
from experiments.cue_validity_ppo_retest_experiment import run_decorrelate_arm
from experiments.cue_validity_ppo_total_matched_experiment import run_naive_arm_total_matched

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
RETEST_SUMMARY_CSV = os.path.join(RESULTS_DIR, "cue_validity_ppo_retest_summaries.csv")
TOTAL_MATCHED_NAIVE_CSV = os.path.join(RESULTS_DIR, "cue_validity_ppo_total_matched_naive_summaries.csv")

VALIDITY_LEVELS = [0.5, 0.95]
TOPUP_SEEDS = list(range(15, 30))


def load_original_decorrelate(validity):
    with open(RETEST_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        if row["label"] != "decorrelate":
            continue
        if float(row["validity"]) != validity:
            continue
        out.append(abs(float(row["final_proxy_dependence"])))
    return out


def load_original_naive_total_matched(validity):
    with open(TOTAL_MATCHED_NAIVE_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        if float(row["validity"]) != validity:
            continue
        out.append(abs(float(row["final_proxy_dependence"])))
    return out


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    new_rows = []
    for validity in VALIDITY_LEVELS:
        for seed in TOPUP_SEEDS:
            print(f"ppo topup v{validity} decorrelate seed={seed}", flush=True)
            new_rows.append(run_decorrelate_arm(validity, seed))
            print(f"ppo topup v{validity} naive_total_matched seed={seed}", flush=True)
            new_rows.append(run_naive_arm_total_matched(validity, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_ppo_seed_topup_summaries.csv"),
        new_rows,
        ["label", "validity", "seed", "acquisition_proxy_dependence", "acquisition_entropy", "final_proxy_dependence", "final_entropy"],
    )

    results = []
    for validity in VALIDITY_LEVELS:
        decorrelate = load_original_decorrelate(validity) + [
            abs(r["final_proxy_dependence"]) for r in new_rows if r["validity"] == validity and r["label"] == "decorrelate"
        ]
        naive = load_original_naive_total_matched(validity) + [
            abs(r["final_proxy_dependence"]) for r in new_rows if r["validity"] == validity and r["label"] == "naive_total_matched"
        ]
        test = compare_groups(decorrelate, naive)
        d = cohens_d(decorrelate, naive)
        boot_lo, boot_hi, boot_point = bootstrap_diff_ci(decorrelate, naive, seed=0)
        results.append(
            {
                "validity": validity,
                "n_decorrelate": len(decorrelate),
                "n_naive": len(naive),
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
        os.path.join(RESULTS_DIR, "cue_validity_ppo_seed_topup_criteria.csv"),
        results,
        ["validity", "n_decorrelate", "n_naive", "decorrelate_mean", "naive_total_matched_mean", "t_stat", "p_value", "cohens_d", "bootstrap_diff_lo", "bootstrap_diff_hi", "has_effect"],
    )

    any_effect = any(r["has_effect"] for r in results)
    lines = [
        "# PPO Seed Top-Up (n=30): Total-Episode-Matched Comparison",
        "",
        "## Pre-registration",
        "Design and criteria were written and committed in docs/cue_validity_ppo.md (\"Seed top-up\") before this data was collected. 15 new seeds (15-29) added to both arms, combined with the original 15 (0-14) for n=30 per arm per validity level.",
        "",
        "## Results",
        "",
        "| validity | n | decorrelate | naive (total-matched) | t | p | Cohen's d | bootstrap diff CI | has effect |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['validity']} | {row['n_decorrelate']} | {row['decorrelate_mean']:.3f} | {row['naive_total_matched_mean']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | "
            f"[{row['bootstrap_diff_lo']:.3f}, {row['bootstrap_diff_hi']:.3f}] | {row['has_effect']} |"
        )
    lines.extend(
        [
            "",
            "## Verdict (H4, n=30 seed top-up)",
            f"PPO shows the linear_raw/neural_raw pattern with the full n=30 seed count: {any_effect}.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_ppo_seed_topup_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return results


if __name__ == "__main__":
    run()
