"""neural_raw seed top-up for the neural reversal experiment.

Pre-registered: docs/cue_validity_neural_reversal.md, "Seed top-up for
neural_raw" section, committed before this script was run. Adds 15 new
seeds (15-29) to the reversed/naive_reversed arms for neural_raw only,
combines with the original 15 (0-14) from
results/cue_validity_neural_reversal_summaries.csv, and recomputes the
fraction-adapted comparison against the unchanged 15-seed decorrelation
baseline.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import compare_groups, cohens_d
from experiments.cue_validity_neural_reversal_experiment import (
    mean,
    per_seed_fraction_adapted_decorrelation,
    per_seed_fraction_adapted_reversal,
    run_naive_reversed_arm,
    run_reversed_arm,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
ORIGINAL_SUMMARY_CSV = os.path.join(RESULTS_DIR, "cue_validity_neural_reversal_summaries.csv")

CONDITION = "neural_raw"
VALIDITY_LEVELS = [0.5, 0.95]
TOPUP_SEEDS = list(range(15, 30))


def load_original_neural_raw_rows():
    with open(ORIGINAL_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        if row["condition"] != CONDITION:
            continue
        row["validity"] = float(row["validity"])
        row["seed"] = int(row["seed"])
        row["acquisition_proxy_dependence"] = (
            None if row["acquisition_proxy_dependence"] in ("", "None") else float(row["acquisition_proxy_dependence"])
        )
        row["final_block_proxy_dependence"] = float(row["final_block_proxy_dependence"])
        out.append(row)
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

    combined = load_original_neural_raw_rows()
    new_rows = []

    for validity in VALIDITY_LEVELS:
        for seed in TOPUP_SEEDS:
            print(f"{CONDITION} v{validity} reversed seed={seed} (topup)", flush=True)
            summary, _ = run_reversed_arm(CONDITION, validity, seed)
            new_rows.append(summary)

            print(f"{CONDITION} v{validity} naive_reversed seed={seed} (topup)", flush=True)
            summary, _ = run_naive_reversed_arm(CONDITION, validity, seed)
            new_rows.append(summary)

    combined.extend(new_rows)

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_neural_raw_topup_summaries.csv"),
        combined,
        ["label", "condition", "validity", "seed", "acquisition_proxy_dependence", "final_block_proxy_dependence"],
    )

    results = []
    for validity in VALIDITY_LEVELS:
        reversal_fracs = per_seed_fraction_adapted_reversal(combined, CONDITION, validity)
        decorrelation_fracs = per_seed_fraction_adapted_decorrelation(CONDITION, validity)
        test = compare_groups(reversal_fracs, decorrelation_fracs)
        d = cohens_d(reversal_fracs, decorrelation_fracs)
        results.append(
            {
                "condition": CONDITION,
                "validity": validity,
                "n_reversal_seeds": len(reversal_fracs),
                "n_decorrelation_seeds": len(decorrelation_fracs),
                "reversal_fraction_adapted": round(mean(reversal_fracs), 6),
                "decorrelation_fraction_adapted": round(mean(decorrelation_fracs), 6),
                "t_stat": round(float(test["t_stat"]), 6),
                "p_value": round(float(test["p_value"]), 6),
                "cohens_d": round(float(d), 6),
                "asymmetry_supported": bool(test["p_value"] < 0.05 and mean(reversal_fracs) > mean(decorrelation_fracs)),
            }
        )

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_neural_raw_topup_normalized.csv"),
        results,
        [
            "condition",
            "validity",
            "n_reversal_seeds",
            "n_decorrelation_seeds",
            "reversal_fraction_adapted",
            "decorrelation_fraction_adapted",
            "t_stat",
            "p_value",
            "cohens_d",
            "asymmetry_supported",
        ],
    )

    lines = [
        "# neural_raw Seed Top-Up Result",
        "",
        "Pre-registered in docs/cue_validity_neural_reversal.md (\"Seed top-up for neural_raw\") before these 15 additional seeds (15-29) were run. Combined with the original 15 (0-14); decorrelation baseline unchanged at 15 seeds.",
        "",
        "| condition | validity | n reversal seeds | n decorrelation seeds | reversal fraction adapted | decorrelation fraction adapted | t | p | Cohen's d | asymmetry supported |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['condition']} | {row['validity']} | {row['n_reversal_seeds']} | {row['n_decorrelation_seeds']} | "
            f"{row['reversal_fraction_adapted']:.3f} | {row['decorrelation_fraction_adapted']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['asymmetry_supported']} |"
        )
    with open(os.path.join(REPORTS_DIR, "cue_validity_neural_raw_topup_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return results


if __name__ == "__main__":
    run()
