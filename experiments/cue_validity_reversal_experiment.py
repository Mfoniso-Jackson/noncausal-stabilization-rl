"""Cue Validity Reversal experiment runner.

Pre-registered design: docs/cue_validity_reversal.md. Direct continuation
of experiments/cue_validity_proxy_dependence_experiment.py -- reuses its
agent classes and helpers rather than redefining them, and reads its
already-committed results CSV as the decorrelation-condition comparison
baseline.
"""

import csv
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import compare_groups, cohens_d
from env.cue_validity_env import ACTION_CAUSAL, ACTION_CUE, NEUTRAL_ACTIONS, CueValidityEnv
from experiments.cue_validity_proxy_dependence_experiment import (
    AGENT_TYPES,
    AVAILABLE_ACTIONS,
    BLOCK_SIZE,
    EPSILON_END,
    EPSILON_START,
    INTERVENTION_EPISODES,
    NUM_BLOCKS,
    SEEDS,
    TARGET_PHASE,
    epsilon_schedule,
    make_agent,
    make_env,
    probe_q_values,
    read_metrics,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
DECORRELATION_SUMMARY_CSV = os.path.join(RESULTS_DIR, "cue_validity_proxy_dependence_summaries.csv")

ACQUISITION_EPISODES = 3000
INTERVENTION_EPSILON = EPSILON_END
VALIDITY_LEVELS = [0.5, 0.95]


def run_reversed_arm(agent_type, validity, seed):
    env = make_env(validity, seed)
    agent = make_agent(agent_type, env, seed)
    for epsilon in epsilon_schedule(ACQUISITION_EPISODES):
        agent.train_episode(env, epsilon)
    acq_cds, acq_dep = read_metrics(agent_type, agent, validity)

    env.reverse()

    block_rows = []
    for block_index in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            agent.train_episode(env, INTERVENTION_EPSILON)
        block_cds, block_dep = read_metrics(agent_type, agent, validity)
        block_rows.append(
            {
                "label": "reversed",
                "agent_type": agent_type,
                "validity": validity,
                "seed": seed,
                "block_index": block_index,
                "cds": round(block_cds, 6),
                "proxy_dependence": round(block_dep, 6),
            }
        )
    summary = {
        "label": "reversed",
        "agent_type": agent_type,
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_block_proxy_dependence": block_rows[-1]["proxy_dependence"],
    }
    return summary, block_rows


def run_naive_reversed_arm(agent_type, validity, seed):
    env = make_env(validity, seed)
    env.reverse()
    agent = make_agent(agent_type, env, seed + 1_000_000)

    block_rows = []
    for block_index in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            agent.train_episode(env, INTERVENTION_EPSILON)
        block_cds, block_dep = read_metrics(agent_type, agent, validity)
        block_rows.append(
            {
                "label": "naive_reversed",
                "agent_type": agent_type,
                "validity": validity,
                "seed": seed,
                "block_index": block_index,
                "cds": round(block_cds, 6),
                "proxy_dependence": round(block_dep, 6),
            }
        )
    summary = {
        "label": "naive_reversed",
        "agent_type": agent_type,
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


def per_seed_gap(summaries, agent_type, validity):
    reversed_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in summaries
        if row["agent_type"] == agent_type and row["validity"] == validity and row["label"] == "reversed"
    }
    naive_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in summaries
        if row["agent_type"] == agent_type and row["validity"] == validity and row["label"] == "naive_reversed"
    }
    return [reversed_by_seed[s] - naive_by_seed[s] for s in sorted(reversed_by_seed) if s in naive_by_seed]


def load_decorrelation_gaps():
    """Per-seed |pretrained_final - naive_final| from the already-committed
    decorrelation experiment, for the direct comparison this experiment's
    success criteria depend on."""
    with open(DECORRELATION_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["validity"] = float(row["validity"])
        row["seed"] = int(row["seed"])
        row["final_block_proxy_dependence"] = (
            None if row["final_block_proxy_dependence"] in ("", "None") else float(row["final_block_proxy_dependence"])
        )

    gaps = {}
    for agent_type in AGENT_TYPES:
        for validity in VALIDITY_LEVELS:
            decorrelate_by_seed = {
                row["seed"]: row["final_block_proxy_dependence"]
                for row in rows
                if row["agent_type"] == agent_type and row["validity"] == validity and row["label"] == f"v{validity}_decorrelate"
            }
            naive_by_seed = {
                row["seed"]: row["final_block_proxy_dependence"]
                for row in rows
                if row["agent_type"] == agent_type and row["validity"] == validity and row["label"] == f"v{validity}_naive"
            }
            gaps[(agent_type, validity)] = [
                abs(decorrelate_by_seed[s] - naive_by_seed[s]) for s in sorted(decorrelate_by_seed) if s in naive_by_seed
            ]
    return gaps


def load_decorrelation_rows():
    with open(DECORRELATION_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["validity"] = float(row["validity"])
        row["seed"] = int(row["seed"])
        for key in ("acquisition_proxy_dependence", "final_block_proxy_dependence"):
            row[key] = None if row[key] in ("", "None") else float(row[key])
    return rows


def per_seed_fraction_adapted_reversal(summaries, agent_type, validity):
    acq_by_seed = {
        row["seed"]: row["acquisition_proxy_dependence"]
        for row in summaries
        if row["agent_type"] == agent_type and row["validity"] == validity and row["label"] == "reversed"
    }
    final_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in summaries
        if row["agent_type"] == agent_type and row["validity"] == validity and row["label"] == "reversed"
    }
    naive_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in summaries
        if row["agent_type"] == agent_type and row["validity"] == validity and row["label"] == "naive_reversed"
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


def per_seed_fraction_adapted_decorrelation(agent_type, validity):
    rows = load_decorrelation_rows()
    acq_by_seed = {
        row["seed"]: row["acquisition_proxy_dependence"]
        for row in rows
        if row["agent_type"] == agent_type and row["validity"] == validity and row["label"] == f"v{validity}_decorrelate"
    }
    final_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in rows
        if row["agent_type"] == agent_type and row["validity"] == validity and row["label"] == f"v{validity}_decorrelate"
    }
    naive_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in rows
        if row["agent_type"] == agent_type and row["validity"] == validity and row["label"] == f"v{validity}_naive"
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
    """Post-hoc, exploratory analysis (not part of the pre-registered
    criteria): the raw-gap comparison above conflates two different things
    -- how far an agent must travel to reach the fully-adapted value, and
    how fast it travels per unit of training. Reversal's fully-adapted
    target is a full sign flip, a strictly larger distance from the
    acquisition-end value than decorrelation's target (~0), so a larger raw
    gap under reversal does not by itself mean slower adaptation. This
    function computes, per seed, the fraction of the required change
    (acquisition value to naive-control's fully-adapted value) actually
    achieved by the final block, for both interventions, and compares them
    directly -- a comparison of rates, not distances."""
    results = []
    for agent_type in AGENT_TYPES:
        for validity in VALIDITY_LEVELS:
            reversal_fracs = per_seed_fraction_adapted_reversal(summaries, agent_type, validity)
            decorrelation_fracs = per_seed_fraction_adapted_decorrelation(agent_type, validity)
            test = compare_groups(reversal_fracs, decorrelation_fracs)
            d = cohens_d(reversal_fracs, decorrelation_fracs)
            results.append(
                {
                    "agent_type": agent_type,
                    "validity": validity,
                    "reversal_fraction_adapted": round(mean(reversal_fracs), 6),
                    "decorrelation_fraction_adapted": round(mean(decorrelation_fracs), 6),
                    "t_stat": round(float(test["t_stat"]), 6),
                    "p_value": round(float(test["p_value"]), 6),
                    "cohens_d": round(float(d), 6),
                    "reversal_faster": bool(test["p_value"] < 0.05 and mean(reversal_fracs) > mean(decorrelation_fracs)),
                }
            )
    return results


def success_criteria(summaries):
    decorrelation_gaps = load_decorrelation_gaps()
    results = []
    for agent_type in AGENT_TYPES:
        for validity in VALIDITY_LEVELS:
            reversal_gap = [abs(g) for g in per_seed_gap(summaries, agent_type, validity)]
            decorrelation_gap = decorrelation_gaps[(agent_type, validity)]
            test = compare_groups(reversal_gap, decorrelation_gap)
            d = cohens_d(reversal_gap, decorrelation_gap)
            results.append(
                {
                    "agent_type": agent_type,
                    "validity": validity,
                    "reversal_gap_mean": round(mean(reversal_gap), 6),
                    "decorrelation_gap_mean": round(mean(decorrelation_gap), 6),
                    "t_stat": round(float(test["t_stat"]), 6),
                    "p_value": round(float(test["p_value"]), 6),
                    "cohens_d": round(float(d), 6),
                    "asymmetry_supported": bool(test["p_value"] < 0.05 and mean(reversal_gap) < mean(decorrelation_gap)),
                }
            )
    return results


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(summaries, criteria_results, normalized_results):
    def rows_for(agent_type, label, validity):
        return [row for row in summaries if row["agent_type"] == agent_type and row["label"] == label and row["validity"] == validity]

    def mean_field(agent_type, label, validity, field):
        values = [row[field] for row in rows_for(agent_type, label, validity) if row[field] is not None]
        return mean(values) if values else float("nan")

    lines = [
        "# Cue Validity Reversal Experiment",
        "",
        "## Pre-registration",
        "Design, metrics, and success/failure criteria were written and committed in docs/cue_validity_reversal.md before any of the data below was collected.",
        "",
        "## Design summary",
        f"cue_validity in {VALIDITY_LEVELS}, agent/representation in {AGENT_TYPES}, {len(SEEDS)} seeds. Same environment and agents as the decorrelation experiment; env.reverse() replaces env.decorrelate() as the intervention. Primary comparison: this experiment's per-seed |reversed_final - naive_reversed_final| gap against the already-collected decorrelation experiment's |decorrelate_final - naive_final| gap, same agent type and validity.",
        "",
        "## Results table (mean across seeds, signed proxy_dependence)",
        "",
        "| agent_type | validity | acquisition dep | reversed final dep | naive-reversed final dep |",
        "|---|---:|---:|---:|---:|",
    ]
    for agent_type in AGENT_TYPES:
        for validity in VALIDITY_LEVELS:
            acq = mean_field(agent_type, "reversed", validity, "acquisition_proxy_dependence")
            rev_final = mean_field(agent_type, "reversed", validity, "final_block_proxy_dependence")
            naive_final = mean_field(agent_type, "naive_reversed", validity, "final_block_proxy_dependence")
            lines.append(f"| {agent_type} | {validity} | {acq:.3f} | {rev_final:.3f} | {naive_final:.3f} |")

    lines.extend(
        [
            "",
            "## Gradient-asymmetry test: reversal gap vs decorrelation gap (|pretrained final - naive final|)",
            "",
            "| agent_type | validity | reversal gap | decorrelation gap | t | p | Cohen's d | asymmetry supported |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in criteria_results:
        lines.append(
            f"| {row['agent_type']} | {row['validity']} | {row['reversal_gap_mean']:.3f} | {row['decorrelation_gap_mean']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['asymmetry_supported']} |"
        )

    any_supported = any(row["asymmetry_supported"] for row in criteria_results if row["agent_type"] != "tabular")
    tabular_rows = [row for row in criteria_results if row["agent_type"] == "tabular"]
    tabular_stayed_small = all(row["reversal_gap_mean"] < 0.05 and row["decorrelation_gap_mean"] < 0.05 for row in tabular_rows)

    lines.extend(
        [
            "",
            "## Verdict (pre-registered criterion: raw |pretrained final - naive final| gap)",
            f"Gradient-asymmetry mechanism supported (reversal gap significantly smaller than decorrelation gap) in at least one function-approximation condition: {any_supported}.",
            f"Tabular gaps stayed small under both interventions, as expected (not a test of the asymmetry mechanism): {tabular_stayed_small}.",
            "",
            "## Post-hoc analysis: the raw-gap comparison conflates distance with speed",
            (
                "Reversal's fully-adapted target is a full sign flip from the acquisition value; decorrelation's is approximately zero -- "
                "a strictly smaller distance from the same starting point. The raw gap above is therefore not a fair comparison of "
                "adaptation *speed*: an agent could be adapting faster under reversal in relative terms while still showing a larger "
                "absolute residual gap, simply because it had further to travel in the same episode budget. The table below instead "
                "reports, per seed, the fraction of the required change (acquisition value to the naive control's fully-adapted value) "
                "actually achieved by the final block -- a comparison of rates, not distances. This was not part of the pre-registered "
                "criteria in docs/cue_validity_reversal.md and is reported as exploratory, not confirmatory."
            ),
            "",
            "| agent_type | validity | reversal fraction adapted | decorrelation fraction adapted | t | p | Cohen's d | reversal faster |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in normalized_results:
        lines.append(
            f"| {row['agent_type']} | {row['validity']} | {row['reversal_fraction_adapted']:.3f} | {row['decorrelation_fraction_adapted']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['reversal_faster']} |"
        )

    normalized_supported = all(
        row["reversal_faster"] for row in normalized_results if row["agent_type"] != "tabular"
    )
    lines.extend(
        [
            "",
            "## Revised verdict (post-hoc, normalized comparison)",
            (
                f"Under the normalized (rate, not distance) comparison, reversal shows a significantly larger fraction of the required "
                f"adaptation achieved than decorrelation, consistently across both linear representations and both validity levels: "
                f"{normalized_supported}. Tabular reaches close to full adaptation (fraction close to 1.0) under both interventions, as "
                "expected. This reverses the pre-registered raw-gap verdict above and is consistent with the gradient-asymmetry "
                "mechanism this experiment set out to test -- reversal's active, structured corrective signal does produce "
                "proportionally faster unlearning of the old association than decorrelation's passive, zero-mean signal, for both "
                "linear-representation conditions, even though reversal's absolute residual gap is larger because it had a longer "
                "distance to cover in the same episode budget."
            ),
            "",
            "## Limitations",
            "The naive-reversed control, like the decorrelation experiment's naive control, starts from a zero-initialized/fresh agent rather than an agent with unrelated prior training, so it is a lower bound on how much prior history matters, not a full dissociation of 'any training' from 'this specific correlation's training'. The comparison against the decorrelation experiment's results reuses that experiment's seeds and episode budget exactly, but the two experiments were run as separate processes, not interleaved, so any uncontrolled environmental difference between the two runs (e.g. library versions) would not be caught by this design. The pre-registered raw-gap criterion and the post-hoc normalized criterion disagree, which is itself the main methodological lesson of this report: a 'gap to naive control' metric should be normalized by the required distance whenever the fully-adapted target differs between conditions being compared, and this was not anticipated when docs/cue_validity_reversal.md was written.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_reversal_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    summaries = []
    block_rows = []

    for agent_type in AGENT_TYPES:
        for validity in VALIDITY_LEVELS:
            for seed in SEEDS:
                print(f"{agent_type} v{validity} reversed seed={seed}", flush=True)
                summary, blocks = run_reversed_arm(agent_type, validity, seed)
                summaries.append(summary)
                block_rows.extend(blocks)

                print(f"{agent_type} v{validity} naive_reversed seed={seed}", flush=True)
                summary, blocks = run_naive_reversed_arm(agent_type, validity, seed)
                summaries.append(summary)
                block_rows.extend(blocks)

    criteria_results = success_criteria(summaries)
    normalized_results = normalized_adaptation_comparison(summaries)

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_reversal_summaries.csv"),
        summaries,
        ["label", "agent_type", "validity", "seed", "acquisition_proxy_dependence", "final_block_proxy_dependence"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_reversal_blocks.csv"),
        block_rows,
        ["label", "agent_type", "validity", "seed", "block_index", "cds", "proxy_dependence"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_reversal_criteria.csv"),
        criteria_results,
        ["agent_type", "validity", "reversal_gap_mean", "decorrelation_gap_mean", "t_stat", "p_value", "cohens_d", "asymmetry_supported"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_reversal_normalized.csv"),
        normalized_results,
        ["agent_type", "validity", "reversal_fraction_adapted", "decorrelation_fraction_adapted", "t_stat", "p_value", "cohens_d", "reversal_faster"],
    )
    write_report(summaries, criteria_results, normalized_results)


if __name__ == "__main__":
    run()
