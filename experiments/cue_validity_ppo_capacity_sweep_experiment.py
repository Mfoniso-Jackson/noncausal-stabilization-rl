"""PPO capacity sweep: widths 16 and 64, reusing the already-collected
width-32 (default) result.

Pre-registered design: docs/cue_validity_ppo_capacity_sweep.md. Reuses
experiments/cue_validity_ppo_experiment.py's rollout/probe/metric
functions and the validated methodology from cue_validity_ppo_retest_experiment.py
/ cue_validity_ppo_total_matched_experiment.py (800-episode acquisition,
total-episode-matched naive) unchanged; only PPOAgent's hidden_size
differs.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import PPOAgent
from analysis.stats import compare_groups, cohens_d
from experiments.cue_validity_proxy_dependence_experiment import SEEDS, make_env, mean
from experiments.cue_validity_ppo_experiment import read_metrics, run_rollout_block

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
RETEST_SUMMARY_CSV = os.path.join(RESULTS_DIR, "cue_validity_ppo_retest_summaries.csv")
TOTAL_MATCHED_NAIVE_CSV = os.path.join(RESULTS_DIR, "cue_validity_ppo_total_matched_naive_summaries.csv")

VALIDITY_LEVELS = [0.5, 0.95]
NEW_WIDTHS = [16, 64]
REUSED_WIDTH = 32
BLOCK_SIZE = 50
ACQUISITION_EPISODES = 800
ACQUISITION_BLOCKS = ACQUISITION_EPISODES // BLOCK_SIZE
INTERVENTION_BLOCKS = 1500 // BLOCK_SIZE
TOTAL_BLOCKS = 2300 // BLOCK_SIZE


def run_decorrelate_arm(width, validity, seed):
    env = make_env(validity, seed)
    agent = PPOAgent(feature_dim=6, seed=seed, hidden_size=width)
    for _ in range(ACQUISITION_BLOCKS):
        run_rollout_block(agent, env, BLOCK_SIZE)
    _, acq_dep = read_metrics(agent, validity)

    env.decorrelate()
    for _ in range(INTERVENTION_BLOCKS):
        run_rollout_block(agent, env, BLOCK_SIZE)
    _, final_dep = read_metrics(agent, validity)

    return {
        "label": "decorrelate",
        "width": width,
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_proxy_dependence": round(final_dep, 6),
    }


def run_naive_arm_total_matched(width, validity, seed):
    env = make_env(validity, seed)
    env.decorrelate()
    agent = PPOAgent(feature_dim=6, seed=seed + 1_000_000, hidden_size=width)
    for _ in range(TOTAL_BLOCKS):
        run_rollout_block(agent, env, BLOCK_SIZE)
    _, final_dep = read_metrics(agent, validity)
    return {
        "label": "naive_total_matched",
        "width": width,
        "validity": validity,
        "seed": seed,
        "final_proxy_dependence": round(final_dep, 6),
    }


def load_reused_width32(validity):
    with open(RETEST_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        decorrelate_rows = list(csv.DictReader(f))
    for row in decorrelate_rows:
        row["validity"] = float(row["validity"])
        row["final_proxy_dependence"] = None if row["final_proxy_dependence"] in ("", "None") else float(row["final_proxy_dependence"])
    decorrelate = [abs(r["final_proxy_dependence"]) for r in decorrelate_rows if r["validity"] == validity and r["label"] == "decorrelate"]

    with open(TOTAL_MATCHED_NAIVE_CSV, newline="", encoding="utf-8") as f:
        naive_rows = list(csv.DictReader(f))
    for row in naive_rows:
        row["validity"] = float(row["validity"])
        row["final_proxy_dependence"] = float(row["final_proxy_dependence"])
    naive = [abs(r["final_proxy_dependence"]) for r in naive_rows if r["validity"] == validity]
    return decorrelate, naive


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

    for width in NEW_WIDTHS:
        for validity in VALIDITY_LEVELS:
            for seed in SEEDS:
                print(f"width={width} v{validity} decorrelate seed={seed}", flush=True)
                summaries.append(run_decorrelate_arm(width, validity, seed))
                print(f"width={width} v{validity} naive_total_matched seed={seed}", flush=True)
                summaries.append(run_naive_arm_total_matched(width, validity, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_ppo_capacity_sweep_summaries.csv"),
        summaries,
        ["label", "width", "validity", "seed", "acquisition_proxy_dependence", "final_proxy_dependence"],
    )

    results = []
    for width in NEW_WIDTHS:
        for validity in VALIDITY_LEVELS:
            decorrelate = [abs(r["final_proxy_dependence"]) for r in summaries if r["width"] == width and r["validity"] == validity and r["label"] == "decorrelate"]
            naive = [abs(r["final_proxy_dependence"]) for r in summaries if r["width"] == width and r["validity"] == validity and r["label"] == "naive_total_matched"]
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
                }
            )
    for validity in VALIDITY_LEVELS:
        decorrelate, naive = load_reused_width32(validity)
        test = compare_groups(decorrelate, naive)
        d = cohens_d(decorrelate, naive)
        results.append(
            {
                "width": REUSED_WIDTH,
                "validity": validity,
                "decorrelate_mean": round(mean(decorrelate), 6),
                "naive_mean": round(mean(naive), 6),
                "t_stat": round(float(test["t_stat"]), 6),
                "p_value": round(float(test["p_value"]), 6),
                "cohens_d": round(float(d), 6),
            }
        )

    results.sort(key=lambda r: (r["width"], r["validity"]))
    adjusted = holm_bonferroni([r["p_value"] for r in results])
    for r, adj in zip(results, adjusted):
        r["holm_adjusted_p"] = round(adj, 6)
        r["has_effect"] = bool(adj < 0.05 and r["decorrelate_mean"] > r["naive_mean"])

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_ppo_capacity_sweep_criteria.csv"),
        results,
        ["width", "validity", "decorrelate_mean", "naive_mean", "t_stat", "p_value", "holm_adjusted_p", "cohens_d", "has_effect"],
    )

    lines = [
        "# PPO Capacity Sweep",
        "",
        "## Pre-registration",
        "Design and criteria were written and committed in docs/cue_validity_ppo_capacity_sweep.md before any of the data below was collected. Width 32 reuses already-collected results; widths 16 and 64 are new. Holm-Bonferroni correction applied across the full 6-cell family.",
        "",
        "## Capacity curve (total-episode-matched comparison)",
        "",
        "| width | validity | decorrelate | naive | t | raw p | holm p | Cohen's d | has effect |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['width']} | {r['validity']} | {r['decorrelate_mean']:.3f} | {r['naive_mean']:.3f} | "
            f"{r['t_stat']:.3f} | {r['p_value']:.4f} | {r['holm_adjusted_p']:.4f} | {r['cohens_d']:.3f} | {r['has_effect']} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            f"Any width shows a significant effect after correction: {any(r['has_effect'] for r in results)}.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_ppo_capacity_sweep_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
