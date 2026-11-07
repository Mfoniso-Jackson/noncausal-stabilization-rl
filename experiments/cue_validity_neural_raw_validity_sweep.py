"""neural_raw validity sweep: fills in cue_validity in {0.6, 0.75, 0.9}
between the two already-tested levels (0.5, 0.95) to test whether the
gradient-asymmetry effect's apparent validity-dependence (seen only at the
two endpoints) tracks headroom below the adaptation ceiling.

Pre-registered design: docs/cue_validity_neural_raw_validity_sweep.md.
Reuses experiments/cue_validity_nonlinear_experiment.py's decorrelate-side
arms and experiments/cue_validity_neural_reversal_experiment.py's
reversal-side arms verbatim; combines with the already-committed 0.5/0.95
data rather than re-running them.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import pearsonr

from analysis.stats import compare_groups, cohens_d
from experiments.cue_validity_neural_reversal_experiment import (
    mean,
    run_naive_reversed_arm,
    run_reversed_arm,
)
from experiments.cue_validity_nonlinear_experiment import (
    read_metrics,
    run_naive_arm,
    run_pretrained_arm,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
NONLINEAR_SUMMARY_CSV = os.path.join(RESULTS_DIR, "cue_validity_nonlinear_summaries.csv")
TOPUP_SUMMARY_CSV = os.path.join(RESULTS_DIR, "cue_validity_neural_raw_topup_summaries.csv")

CONDITION = "neural_raw"
NEW_VALIDITY_LEVELS = [0.6, 0.75, 0.9]
EXISTING_VALIDITY_LEVELS = [0.5, 0.95]
ALL_VALIDITY_LEVELS = [0.5, 0.6, 0.75, 0.9, 0.95]
SEEDS = list(range(15))


def fraction_adapted(acq, final, naive_final):
    denom = acq - naive_final
    if abs(denom) < 1e-6:
        return None
    return (acq - final) / denom


def decorrelation_fraction_adapted_new(validity):
    acq_by_seed = {}
    final_by_seed = {}
    for seed in SEEDS:
        row = run_pretrained_arm(CONDITION, validity, seed)
        acq_by_seed[seed] = row["acquisition_proxy_dependence"]
        final_by_seed[seed] = row["final_proxy_dependence"]
    naive_by_seed = {}
    for seed in SEEDS:
        row = run_naive_arm(CONDITION, validity, seed)
        naive_by_seed[seed] = row["final_proxy_dependence"]
    fractions = []
    for s in SEEDS:
        f = fraction_adapted(acq_by_seed[s], final_by_seed[s], naive_by_seed[s])
        if f is not None:
            fractions.append(f)
    return fractions


def reversal_fraction_adapted_new(validity):
    acq_by_seed = {}
    final_by_seed = {}
    for seed in SEEDS:
        summary, _ = run_reversed_arm(CONDITION, validity, seed)
        acq_by_seed[seed] = summary["acquisition_proxy_dependence"]
        final_by_seed[seed] = summary["final_block_proxy_dependence"]
    naive_by_seed = {}
    for seed in SEEDS:
        summary, _ = run_naive_reversed_arm(CONDITION, validity, seed)
        naive_by_seed[seed] = summary["final_block_proxy_dependence"]
    fractions = []
    for s in SEEDS:
        f = fraction_adapted(acq_by_seed[s], final_by_seed[s], naive_by_seed[s])
        if f is not None:
            fractions.append(f)
    return fractions


def load_existing_decorrelation_fractions(validity):
    with open(NONLINEAR_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["validity"] = float(row["validity"])
        row["seed"] = int(row["seed"])
        for key in ("acquisition_proxy_dependence", "final_proxy_dependence"):
            row[key] = None if row[key] in ("", "None") else float(row[key])
    acq_by_seed = {
        row["seed"]: row["acquisition_proxy_dependence"]
        for row in rows
        if row["condition"] == CONDITION and row["validity"] == validity and row["label"] == "decorrelate"
    }
    final_by_seed = {
        row["seed"]: row["final_proxy_dependence"]
        for row in rows
        if row["condition"] == CONDITION and row["validity"] == validity and row["label"] == "decorrelate"
    }
    naive_by_seed = {
        row["seed"]: row["final_proxy_dependence"]
        for row in rows
        if row["condition"] == CONDITION and row["validity"] == validity and row["label"] == "naive"
    }
    fractions = []
    for s in sorted(acq_by_seed):
        if s not in naive_by_seed:
            continue
        f = fraction_adapted(acq_by_seed[s], final_by_seed[s], naive_by_seed[s])
        if f is not None:
            fractions.append(f)
    return fractions


def load_existing_reversal_fractions(validity):
    with open(TOPUP_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["validity"] = float(row["validity"])
        row["seed"] = int(row["seed"])
        row["acquisition_proxy_dependence"] = (
            None if row["acquisition_proxy_dependence"] in ("", "None") else float(row["acquisition_proxy_dependence"])
        )
        row["final_block_proxy_dependence"] = float(row["final_block_proxy_dependence"])
    acq_by_seed = {
        row["seed"]: row["acquisition_proxy_dependence"]
        for row in rows
        if row["condition"] == CONDITION and row["validity"] == validity and row["label"] == "reversed"
    }
    final_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in rows
        if row["condition"] == CONDITION and row["validity"] == validity and row["label"] == "reversed"
    }
    naive_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in rows
        if row["condition"] == CONDITION and row["validity"] == validity and row["label"] == "naive_reversed"
    }
    fractions = []
    for s in sorted(acq_by_seed):
        if s not in naive_by_seed:
            continue
        f = fraction_adapted(acq_by_seed[s], final_by_seed[s], naive_by_seed[s])
        if f is not None:
            fractions.append(f)
    return fractions


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    per_level = {}
    for validity in EXISTING_VALIDITY_LEVELS:
        decorrelation_fracs = load_existing_decorrelation_fractions(validity)
        reversal_fracs = load_existing_reversal_fractions(validity)
        per_level[validity] = (decorrelation_fracs, reversal_fracs, "reused")

    for validity in NEW_VALIDITY_LEVELS:
        print(f"neural_raw v{validity} decorrelation side (new)", flush=True)
        decorrelation_fracs = decorrelation_fraction_adapted_new(validity)
        print(f"neural_raw v{validity} reversal side (new)", flush=True)
        reversal_fracs = reversal_fraction_adapted_new(validity)
        per_level[validity] = (decorrelation_fracs, reversal_fracs, "new")

    level_rows = []
    for validity in ALL_VALIDITY_LEVELS:
        decorrelation_fracs, reversal_fracs, source = per_level[validity]
        test = compare_groups(reversal_fracs, decorrelation_fracs)
        d = cohens_d(reversal_fracs, decorrelation_fracs)
        decorrelation_mean = mean(decorrelation_fracs)
        reversal_mean = mean(reversal_fracs)
        level_rows.append(
            {
                "validity": validity,
                "source": source,
                "n_decorrelation_seeds": len(decorrelation_fracs),
                "n_reversal_seeds": len(reversal_fracs),
                "decorrelation_fraction_adapted": round(decorrelation_mean, 6),
                "reversal_fraction_adapted": round(reversal_mean, 6),
                "headroom": round(1.0 - decorrelation_mean, 6),
                "asymmetry_effect": round(reversal_mean - decorrelation_mean, 6),
                "t_stat": round(float(test["t_stat"]), 6),
                "p_value": round(float(test["p_value"]), 6),
                "cohens_d": round(float(d), 6),
                "asymmetry_supported": bool(test["p_value"] < 0.05 and reversal_mean > decorrelation_mean),
            }
        )

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_neural_raw_validity_sweep.csv"),
        level_rows,
        [
            "validity",
            "source",
            "n_decorrelation_seeds",
            "n_reversal_seeds",
            "decorrelation_fraction_adapted",
            "reversal_fraction_adapted",
            "headroom",
            "asymmetry_effect",
            "t_stat",
            "p_value",
            "cohens_d",
            "asymmetry_supported",
        ],
    )

    headroom_values = [row["headroom"] for row in level_rows]
    effect_values = [row["asymmetry_effect"] for row in level_rows]
    r, p = pearsonr(headroom_values, effect_values)

    lines = [
        "# neural_raw Validity Sweep",
        "",
        "## Pre-registration",
        "Design, metrics, and success/failure criteria were written and committed in docs/cue_validity_neural_raw_validity_sweep.md before any of the new data below was collected.",
        "",
        "## Design summary",
        f"condition=neural_raw only. New cue_validity levels {NEW_VALIDITY_LEVELS} (15 seeds each, all four arms freshly run). Existing levels {EXISTING_VALIDITY_LEVELS} reused from already-committed results (15 decorrelation seeds, 15 or 30 reversal seeds per the seed top-up).",
        "",
        "## Five-point curve",
        "",
        "| validity | source | n decorr. seeds | n reversal seeds | decorrelation frac. adapted | reversal frac. adapted | headroom | asymmetry effect | t | p | Cohen's d | asymmetry supported |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in level_rows:
        lines.append(
            f"| {row['validity']} | {row['source']} | {row['n_decorrelation_seeds']} | {row['n_reversal_seeds']} | "
            f"{row['decorrelation_fraction_adapted']:.3f} | {row['reversal_fraction_adapted']:.3f} | {row['headroom']:.3f} | "
            f"{row['asymmetry_effect']:.3f} | {row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['asymmetry_supported']} |"
        )

    lines.extend(
        [
            "",
            "## Primary (pre-registered) test: headroom vs. asymmetry effect, Pearson correlation across the 5 levels",
            f"r = {r:.4f}, p = {p:.4f} (n=5 points; explicitly pre-registered as underpowered and descriptive, not confirmatory, regardless of outcome).",
            f"Headroom hypothesis supported (r > 0, p < 0.05): {bool(r > 0 and p < 0.05)}.",
        ]
    )

    with open(os.path.join(REPORTS_DIR, "cue_validity_neural_raw_validity_sweep_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return level_rows, r, p


if __name__ == "__main__":
    run()
