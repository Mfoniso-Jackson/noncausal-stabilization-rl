"""Cue Validity Neural Reversal experiment runner.

Pre-registered design: docs/cue_validity_neural_reversal.md. Combines
experiments/cue_validity_nonlinear_experiment.py's agents, feature
encodings, and metric-reading (make_agent, features_for, train_episode,
read_metrics) with experiments/cue_validity_reversal_experiment.py's
reversal intervention and naive-reversed-control methodology. Reads the
nonlinear experiment's already-committed decorrelation results as the
comparison baseline, exactly as the linear reversal experiment read the
proxy-dependence experiment's.
"""

import csv
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import compare_groups, cohens_d
from experiments.cue_validity_nonlinear_experiment import (
    BLOCK_SIZE,
    INTERVENTION_EPSILON,
    NEW_CONDITIONS,
    NUM_BLOCKS,
    SEEDS,
    VALIDITY_LEVELS,
    epsilon_schedule,
    make_agent,
    make_env,
    read_metrics,
    train_episode,
)

ACQUISITION_EPISODES = 3000

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
DECORRELATION_SUMMARY_CSV = os.path.join(RESULTS_DIR, "cue_validity_nonlinear_summaries.csv")


def run_reversed_arm(condition, validity, seed):
    env = make_env(validity, seed)
    agent = make_agent(condition, seed)
    for epsilon in epsilon_schedule(ACQUISITION_EPISODES):
        train_episode(condition, agent, env, epsilon)
    acq_cds, acq_dep = read_metrics(condition, agent, validity)

    env.reverse()

    block_rows = []
    for block_index in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(condition, agent, env, INTERVENTION_EPSILON)
        block_cds, block_dep = read_metrics(condition, agent, validity)
        block_rows.append(
            {
                "label": "reversed",
                "condition": condition,
                "validity": validity,
                "seed": seed,
                "block_index": block_index,
                "cds": round(block_cds, 6),
                "proxy_dependence": round(block_dep, 6),
            }
        )
    summary = {
        "label": "reversed",
        "condition": condition,
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_block_proxy_dependence": block_rows[-1]["proxy_dependence"],
    }
    return summary, block_rows


def run_naive_reversed_arm(condition, validity, seed):
    env = make_env(validity, seed)
    env.reverse()
    agent = make_agent(condition, seed + 1_000_000)

    block_rows = []
    for block_index in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(condition, agent, env, INTERVENTION_EPSILON)
        block_cds, block_dep = read_metrics(condition, agent, validity)
        block_rows.append(
            {
                "label": "naive_reversed",
                "condition": condition,
                "validity": validity,
                "seed": seed,
                "block_index": block_index,
                "cds": round(block_cds, 6),
                "proxy_dependence": round(block_dep, 6),
            }
        )
    summary = {
        "label": "naive_reversed",
        "condition": condition,
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": None,
        "final_block_proxy_dependence": block_rows[-1]["proxy_dependence"],
    }
    return summary, block_rows


def mean(values):
    values = list(values)
    return sum(values) / max(1, len(values))


def sem(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def per_seed_gap(summaries, condition, validity):
    reversed_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in summaries
        if row["condition"] == condition and row["validity"] == validity and row["label"] == "reversed"
    }
    naive_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in summaries
        if row["condition"] == condition and row["validity"] == validity and row["label"] == "naive_reversed"
    }
    return [reversed_by_seed[s] - naive_by_seed[s] for s in sorted(reversed_by_seed) if s in naive_by_seed]


def load_decorrelation_rows():
    with open(DECORRELATION_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["validity"] = float(row["validity"])
        row["seed"] = int(row["seed"])
        for key in ("acquisition_proxy_dependence", "final_proxy_dependence"):
            row[key] = None if row[key] in ("", "None") else float(row[key])
    return rows


def per_seed_fraction_adapted_reversal(summaries, condition, validity):
    acq_by_seed = {
        row["seed"]: row["acquisition_proxy_dependence"]
        for row in summaries
        if row["condition"] == condition and row["validity"] == validity and row["label"] == "reversed"
    }
    final_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in summaries
        if row["condition"] == condition and row["validity"] == validity and row["label"] == "reversed"
    }
    naive_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in summaries
        if row["condition"] == condition and row["validity"] == validity and row["label"] == "naive_reversed"
    }
    fractions = []
    for s in sorted(acq_by_seed):
        if s not in naive_by_seed:
            continue
        denom = acq_by_seed[s] - naive_by_seed[s]
        if abs(denom) < 1e-6:
            continue
        fractions.append((acq_by_seed[s] - final_by_seed[s]) / denom)
    return fractions


def per_seed_fraction_adapted_decorrelation(condition, validity):
    rows = load_decorrelation_rows()
    acq_by_seed = {
        row["seed"]: row["acquisition_proxy_dependence"]
        for row in rows
        if row["condition"] == condition and row["validity"] == validity and row["label"] == "decorrelate"
    }
    final_by_seed = {
        row["seed"]: row["final_proxy_dependence"]
        for row in rows
        if row["condition"] == condition and row["validity"] == validity and row["label"] == "decorrelate"
    }
    naive_by_seed = {
        row["seed"]: row["final_proxy_dependence"]
        for row in rows
        if row["condition"] == condition and row["validity"] == validity and row["label"] == "naive"
    }
    fractions = []
    for s in sorted(acq_by_seed):
        if s not in naive_by_seed:
            continue
        denom = acq_by_seed[s] - naive_by_seed[s]
        if abs(denom) < 1e-6:
            continue
        fractions.append((acq_by_seed[s] - final_by_seed[s]) / denom)
    return fractions


def normalized_adaptation_comparison(summaries):
    """Primary, pre-registered comparison: fraction of the required change
    (acquisition value to naive-control's fully-adapted value) achieved by
    the final block, reversal vs. decorrelation, per condition and
    validity."""
    results = []
    for condition in NEW_CONDITIONS:
        for validity in VALIDITY_LEVELS:
            reversal_fracs = per_seed_fraction_adapted_reversal(summaries, condition, validity)
            decorrelation_fracs = per_seed_fraction_adapted_decorrelation(condition, validity)
            test = compare_groups(reversal_fracs, decorrelation_fracs)
            d = cohens_d(reversal_fracs, decorrelation_fracs)
            results.append(
                {
                    "condition": condition,
                    "validity": validity,
                    "reversal_fraction_adapted": round(mean(reversal_fracs), 6),
                    "decorrelation_fraction_adapted": round(mean(decorrelation_fracs), 6),
                    "t_stat": round(float(test["t_stat"]), 6),
                    "p_value": round(float(test["p_value"]), 6),
                    "cohens_d": round(float(d), 6),
                    "asymmetry_supported": bool(test["p_value"] < 0.05 and mean(reversal_fracs) > mean(decorrelation_fracs)),
                }
            )
    return results


def raw_gap_table(summaries):
    """Secondary, reported for comparability only -- not used to decide
    support for the mechanism (see pre-registration)."""
    results = []
    for condition in NEW_CONDITIONS:
        for validity in VALIDITY_LEVELS:
            gap = [abs(g) for g in per_seed_gap(summaries, condition, validity)]
            results.append(
                {
                    "condition": condition,
                    "validity": validity,
                    "reversal_gap_mean": round(mean(gap), 6),
                }
            )
    return results


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(summaries, normalized_results, raw_gaps):
    def rows_for(condition, label, validity):
        return [row for row in summaries if row["condition"] == condition and row["label"] == label and row["validity"] == validity]

    def mean_field(condition, label, validity, field):
        values = [row[field] for row in rows_for(condition, label, validity) if row[field] is not None]
        return mean(values) if values else float("nan")

    raw_gap_by_key = {(row["condition"], row["validity"]): row["reversal_gap_mean"] for row in raw_gaps}

    lines = [
        "# Cue Validity Neural Reversal Experiment",
        "",
        "## Pre-registration",
        "Design, metrics, and success/failure criteria were written and committed in docs/cue_validity_neural_reversal.md before any of the data below was collected.",
        "",
        "## Design summary",
        f"cue_validity in {VALIDITY_LEVELS}, condition in {NEW_CONDITIONS}, {len(SEEDS)} seeds. Same agents, feature encodings, and episode budget as the nonlinear experiment; env.reverse() replaces env.decorrelate() as the intervention. Primary comparison (pre-registered): per-seed fraction of required adaptation achieved, reversal vs. the already-collected decorrelation data from results/cue_validity_nonlinear_summaries.csv.",
        "",
        "## Results table (mean across seeds, signed proxy_dependence)",
        "",
        "| condition | validity | acquisition dep | reversed final dep | naive-reversed final dep |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in NEW_CONDITIONS:
        for validity in VALIDITY_LEVELS:
            acq = mean_field(condition, "reversed", validity, "acquisition_proxy_dependence")
            rev_final = mean_field(condition, "reversed", validity, "final_block_proxy_dependence")
            naive_final = mean_field(condition, "naive_reversed", validity, "final_block_proxy_dependence")
            lines.append(f"| {condition} | {validity} | {acq:.3f} | {rev_final:.3f} | {naive_final:.3f} |")

    lines.extend(
        [
            "",
            "## Primary (pre-registered) test: fraction of required adaptation achieved, reversal vs. decorrelation",
            "",
            "| condition | validity | reversal fraction adapted | decorrelation fraction adapted | t | p | Cohen's d | asymmetry supported |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in normalized_results:
        lines.append(
            f"| {row['condition']} | {row['validity']} | {row['reversal_fraction_adapted']:.3f} | {row['decorrelation_fraction_adapted']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['asymmetry_supported']} |"
        )

    lines.extend(
        [
            "",
            "## Secondary, reported for comparability only: raw |reversed final - naive-reversed final| gap",
            "Not used to decide support for the mechanism (see pre-registration: reversal and decorrelation have different-distance fully-adapted targets, so raw gaps are not a fair speed comparison).",
            "",
            "| condition | validity | raw reversal gap |",
            "|---|---:|---:|",
        ]
    )
    for condition in NEW_CONDITIONS:
        for validity in VALIDITY_LEVELS:
            lines.append(f"| {condition} | {validity} | {raw_gap_by_key[(condition, validity)]:.3f} |")

    any_supported = any(row["asymmetry_supported"] for row in normalized_results)
    per_condition_supported = {
        condition: any(row["asymmetry_supported"] for row in normalized_results if row["condition"] == condition)
        for condition in NEW_CONDITIONS
    }

    lines.extend(
        [
            "",
            "## Verdict",
            f"Gradient-asymmetry mechanism supported (reversal fraction-adapted significantly larger than decorrelation fraction-adapted) in at least one condition: {any_supported}.",
        ]
    )
    for condition in NEW_CONDITIONS:
        lines.append(f"- {condition}: {per_condition_supported[condition]}")

    lines.extend(
        [
            "",
            "## Limitations",
            (
                "neural_interaction_only showed no significant decorrelate-vs-naive effect in the nonlinear experiment, so there is no "
                "persistent dependency for reversal to extinguish faster than there; its fraction-adapted comparison is reported for "
                "completeness but a null or noisy result there is expected, not a failure of the mechanism. The naive-reversed control, "
                "like both parent experiments' naive controls, starts from a fresh agent rather than one with unrelated prior training, "
                "so it is a lower bound on how much prior history matters. This experiment and the nonlinear experiment it reuses "
                "decorrelation data from were run as separate processes, not interleaved, so any uncontrolled environmental difference "
                "between the two runs would not be caught by this design."
            ),
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_neural_reversal_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    summaries = []
    block_rows = []

    for condition in NEW_CONDITIONS:
        for validity in VALIDITY_LEVELS:
            for seed in SEEDS:
                print(f"{condition} v{validity} reversed seed={seed}", flush=True)
                summary, blocks = run_reversed_arm(condition, validity, seed)
                summaries.append(summary)
                block_rows.extend(blocks)

                print(f"{condition} v{validity} naive_reversed seed={seed}", flush=True)
                summary, blocks = run_naive_reversed_arm(condition, validity, seed)
                summaries.append(summary)
                block_rows.extend(blocks)

    normalized_results = normalized_adaptation_comparison(summaries)
    raw_gaps = raw_gap_table(summaries)

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_neural_reversal_summaries.csv"),
        summaries,
        ["label", "condition", "validity", "seed", "acquisition_proxy_dependence", "final_block_proxy_dependence"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_neural_reversal_blocks.csv"),
        block_rows,
        ["label", "condition", "validity", "seed", "block_index", "cds", "proxy_dependence"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_neural_reversal_normalized.csv"),
        normalized_results,
        ["condition", "validity", "reversal_fraction_adapted", "decorrelation_fraction_adapted", "t_stat", "p_value", "cohens_d", "asymmetry_supported"],
    )
    write_report(summaries, normalized_results, raw_gaps)


if __name__ == "__main__":
    run()
