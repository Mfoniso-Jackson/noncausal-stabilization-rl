"""Grid environment: neural_raw reversal and neural_interaction_only
extension, completing the representation x intervention matrix.

Pre-registered design: docs/cue_validity_grid_neural_reversal_interaction.md.
Reuses experiments/cue_validity_grid_reversal_interaction_experiment.py's
interaction_only_features/features_for/make_env/holm_bonferroni and
experiments/cue_validity_nonlinear_experiment.py's NeuralAgentWrapper
unchanged. Uses the already-established 800-episode step-matched
intervention budget for neural_raw in this environment (1500 is known
null), applied from the start rather than re-discovered.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import compare_groups, cohens_d
from experiments.cue_validity_cross_environment_experiment import ACQUISITION_EPISODES, SEEDS, epsilon_schedule, mean
from experiments.cue_validity_grid_reversal_interaction_experiment import (
    FEATURE_DIM,
    VALIDITY_LEVELS,
    features_for,
    holm_bonferroni,
    make_env,
)
from experiments.cue_validity_nonlinear_experiment import NeuralAgentWrapper
from metrics.cue_validity import cue_dependence_score, proxy_dependence
from env.cue_validity_grid_env import ACTION_CAUSAL, ACTION_CUE, NEUTRAL_ACTIONS, CueValidityGridEnv
from experiments.cue_validity_grid_neural_experiment import GRID_SIZE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
STEPMATCHED_NEURAL_SUMMARY_CSV = os.path.join(RESULTS_DIR, "cue_validity_grid_neural_stepmatched_summaries.csv")

AVAILABLE_ACTIONS = (ACTION_CAUSAL, ACTION_CUE) + NEUTRAL_ACTIONS
INTERVENTION_EPSILON = 0.04
BLOCK_SIZE = 50
NUM_BLOCKS = 800 // BLOCK_SIZE  # step-matched budget, established in docs/cue_validity_cross_environment.md


def train_episode(encoding, agent, env, epsilon):
    agent.epsilon = epsilon
    env.reset()
    state = features_for(encoding, env)
    done = False
    while not done:
        action = agent.act(state)
        _, reward, done, _ = env.step(action)
        next_state = features_for(encoding, env)
        agent.remember(state, action, reward, next_state, done)
        agent.train_step()
        state = next_state


def probe_q_values(encoding, agent, validity, cue_present):
    probe_env = CueValidityGridEnv(validity, grid_size=GRID_SIZE, seed=0)
    probe_env.pos = probe_env.decision_cell
    probe_env.cue_present = int(cue_present)
    values = agent.q_values_for_features(features_for(encoding, probe_env))
    return {a: values[a] for a in AVAILABLE_ACTIONS}


def read_metrics(encoding, agent, validity):
    q_present = probe_q_values(encoding, agent, validity, True)
    q_absent = probe_q_values(encoding, agent, validity, False)
    cds = cue_dependence_score(q_present, AVAILABLE_ACTIONS, ACTION_CAUSAL, ACTION_CUE, NEUTRAL_ACTIONS)
    dep = proxy_dependence(q_present, q_absent, AVAILABLE_ACTIONS, ACTION_CAUSAL)
    return cds, dep


# ---- Reversal arm (neural_raw) ----

def run_reversed_arm(validity, seed):
    env = make_env(validity, seed)
    agent = NeuralAgentWrapper(seed, FEATURE_DIM["raw"])
    for epsilon in epsilon_schedule(ACQUISITION_EPISODES):
        train_episode("raw", agent, env, epsilon)
    _, acq_dep = read_metrics("raw", agent, validity)

    env.reverse()
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode("raw", agent, env, INTERVENTION_EPSILON)
    _, final_dep = read_metrics("raw", agent, validity)

    return {
        "label": "reversed",
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_block_proxy_dependence": round(final_dep, 6),
    }


def run_naive_reversed_arm(validity, seed):
    env = make_env(validity, seed)
    env.reverse()
    agent = NeuralAgentWrapper(seed + 1_000_000, FEATURE_DIM["raw"])
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode("raw", agent, env, INTERVENTION_EPSILON)
    _, final_dep = read_metrics("raw", agent, validity)
    return {
        "label": "naive_reversed",
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": None,
        "final_block_proxy_dependence": round(final_dep, 6),
    }


# ---- Interaction-only arm (neural_interaction_only, decorrelate) ----

def run_decorrelate_arm_interaction(validity, seed):
    env = make_env(validity, seed)
    agent = NeuralAgentWrapper(seed, FEATURE_DIM["interaction_only"])
    for epsilon in epsilon_schedule(ACQUISITION_EPISODES):
        train_episode("interaction_only", agent, env, epsilon)
    _, acq_dep = read_metrics("interaction_only", agent, validity)

    env.decorrelate()
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode("interaction_only", agent, env, INTERVENTION_EPSILON)
    _, final_dep = read_metrics("interaction_only", agent, validity)

    return {
        "label": "decorrelate",
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_proxy_dependence": round(final_dep, 6),
    }


def run_naive_arm_interaction(validity, seed):
    env = make_env(validity, seed)
    env.decorrelate()
    agent = NeuralAgentWrapper(seed + 1_000_000, FEATURE_DIM["interaction_only"])
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode("interaction_only", agent, env, INTERVENTION_EPSILON)
    _, final_dep = read_metrics("interaction_only", agent, validity)
    return {
        "label": "naive",
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": None,
        "final_proxy_dependence": round(final_dep, 6),
    }


# ---- Analysis helpers ----

def fraction_adapted(acq, final, naive_final):
    denom = acq - naive_final
    if abs(denom) < 1e-6:
        return None
    return (acq - final) / denom


def per_seed_fraction_adapted_reversal(summaries, validity):
    acq_by_seed = {r["seed"]: r["acquisition_proxy_dependence"] for r in summaries if r["validity"] == validity and r["label"] == "reversed"}
    final_by_seed = {r["seed"]: r["final_block_proxy_dependence"] for r in summaries if r["validity"] == validity and r["label"] == "reversed"}
    naive_by_seed = {r["seed"]: r["final_block_proxy_dependence"] for r in summaries if r["validity"] == validity and r["label"] == "naive_reversed"}
    fractions = []
    for s in sorted(acq_by_seed):
        if s not in naive_by_seed:
            continue
        f = fraction_adapted(acq_by_seed[s], final_by_seed[s], naive_by_seed[s])
        if f is not None:
            fractions.append(f)
    return fractions


def load_stepmatched_fraction_adapted_neural_raw(validity):
    with open(STEPMATCHED_NEURAL_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["validity"] = float(row["validity"])
        row["seed"] = int(row["seed"])
        for key in ("acquisition_proxy_dependence", "final_block_proxy_dependence"):
            row[key] = None if row[key] in ("", "None") else float(row[key])
    acq_by_seed = {r["seed"]: r["acquisition_proxy_dependence"] for r in rows if r["condition"] == "neural_raw" and r["validity"] == validity and r["label"] == "decorrelate"}
    final_by_seed = {r["seed"]: r["final_block_proxy_dependence"] for r in rows if r["condition"] == "neural_raw" and r["validity"] == validity and r["label"] == "decorrelate"}
    naive_by_seed = {r["seed"]: r["final_block_proxy_dependence"] for r in rows if r["condition"] == "neural_raw" and r["validity"] == validity and r["label"] == "naive"}
    fractions = []
    for s in sorted(acq_by_seed):
        if s not in naive_by_seed:
            continue
        f = fraction_adapted(acq_by_seed[s], final_by_seed[s], naive_by_seed[s])
        if f is not None:
            fractions.append(f)
    return fractions


def load_stepmatched_gap_neural_raw(validity):
    with open(STEPMATCHED_NEURAL_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["validity"] = float(row["validity"])
        row["final_block_proxy_dependence"] = None if row["final_block_proxy_dependence"] in ("", "None") else float(row["final_block_proxy_dependence"])
    decorrelate = [abs(r["final_block_proxy_dependence"]) for r in rows if r["condition"] == "neural_raw" and r["validity"] == validity and r["label"] == "decorrelate"]
    naive = [abs(r["final_block_proxy_dependence"]) for r in rows if r["condition"] == "neural_raw" and r["validity"] == validity and r["label"] == "naive"]
    return decorrelate, naive


def gap_values_interaction(summaries, validity, label):
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

    reversal_summaries = []
    for validity in VALIDITY_LEVELS:
        for seed in SEEDS:
            print(f"neural reversal v{validity} reversed seed={seed}", flush=True)
            reversal_summaries.append(run_reversed_arm(validity, seed))
            print(f"neural reversal v{validity} naive_reversed seed={seed}", flush=True)
            reversal_summaries.append(run_naive_reversed_arm(validity, seed))

    interaction_summaries = []
    for validity in VALIDITY_LEVELS:
        for seed in SEEDS:
            print(f"neural interaction v{validity} decorrelate seed={seed}", flush=True)
            interaction_summaries.append(run_decorrelate_arm_interaction(validity, seed))
            print(f"neural interaction v{validity} naive seed={seed}", flush=True)
            interaction_summaries.append(run_naive_arm_interaction(validity, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_grid_neural_reversal_summaries.csv"),
        reversal_summaries,
        ["label", "validity", "seed", "acquisition_proxy_dependence", "final_block_proxy_dependence"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_grid_neural_interaction_summaries.csv"),
        interaction_summaries,
        ["label", "validity", "seed", "acquisition_proxy_dependence", "final_proxy_dependence"],
    )

    reversal_results = []
    for validity in VALIDITY_LEVELS:
        reversal_fracs = per_seed_fraction_adapted_reversal(reversal_summaries, validity)
        decorrelation_fracs = load_stepmatched_fraction_adapted_neural_raw(validity)
        test = compare_groups(reversal_fracs, decorrelation_fracs)
        d = cohens_d(reversal_fracs, decorrelation_fracs)
        reversal_results.append(
            {
                "validity": validity,
                "reversal_fraction_adapted": round(mean(reversal_fracs), 6),
                "decorrelation_fraction_adapted": round(mean(decorrelation_fracs), 6),
                "t_stat": round(float(test["t_stat"]), 6),
                "p_value": round(float(test["p_value"]), 6),
                "cohens_d": round(float(d), 6),
            }
        )
    reversal_adj = holm_bonferroni([r["p_value"] for r in reversal_results])
    for r, adj in zip(reversal_results, reversal_adj):
        r["holm_adjusted_p"] = round(adj, 6)
        r["asymmetry_supported"] = bool(adj < 0.05 and r["reversal_fraction_adapted"] > r["decorrelation_fraction_adapted"])

    interaction_results = []
    for validity in VALIDITY_LEVELS:
        interaction_decorrelate = gap_values_interaction(interaction_summaries, validity, "decorrelate")
        interaction_naive = gap_values_interaction(interaction_summaries, validity, "naive")
        raw_decorrelate, raw_naive = load_stepmatched_gap_neural_raw(validity)

        own_test = compare_groups(interaction_decorrelate, interaction_naive)
        own_d = cohens_d(interaction_decorrelate, interaction_naive)

        raw_gap = [d_ - mean(raw_naive) for d_ in raw_decorrelate]
        interaction_gap = [d_ - mean(interaction_naive) for d_ in interaction_decorrelate]
        coord_test = compare_groups(raw_gap, interaction_gap)
        coord_d = cohens_d(raw_gap, interaction_gap)

        interaction_results.append(
            {
                "validity": validity,
                "interaction_decorrelate_mean": round(mean(interaction_decorrelate), 6),
                "interaction_naive_mean": round(mean(interaction_naive), 6),
                "own_p_value": round(float(own_test["p_value"]), 6),
                "own_cohens_d": round(float(own_d), 6),
                "coordinate_test_p_value": round(float(coord_test["p_value"]), 6),
                "coordinate_test_cohens_d": round(float(coord_d), 6),
            }
        )
    own_adj = holm_bonferroni([r["own_p_value"] for r in interaction_results])
    coord_adj = holm_bonferroni([r["coordinate_test_p_value"] for r in interaction_results])
    for r, oa, ca in zip(interaction_results, own_adj, coord_adj):
        r["own_holm_adjusted_p"] = round(oa, 6)
        r["coordinate_holm_adjusted_p"] = round(ca, 6)
        r["has_own_effect"] = bool(oa < 0.05 and r["interaction_decorrelate_mean"] > r["interaction_naive_mean"])
        r["coordinate_dependent"] = bool(ca < 0.05 and r["coordinate_test_cohens_d"] > 0)

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_grid_neural_reversal_criteria.csv"),
        reversal_results,
        ["validity", "reversal_fraction_adapted", "decorrelation_fraction_adapted", "t_stat", "p_value", "holm_adjusted_p", "cohens_d", "asymmetry_supported"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_grid_neural_interaction_criteria.csv"),
        interaction_results,
        ["validity", "interaction_decorrelate_mean", "interaction_naive_mean", "own_p_value", "own_holm_adjusted_p", "own_cohens_d",
         "coordinate_test_p_value", "coordinate_holm_adjusted_p", "coordinate_test_cohens_d", "has_own_effect", "coordinate_dependent"],
    )

    lines = [
        "# Grid Environment: neural_raw Reversal and neural_interaction_only Extension",
        "",
        "## Pre-registration",
        "Design and criteria were written and committed in docs/cue_validity_grid_neural_reversal_interaction.md before any of the data below was collected. Uses the already-established step-matched 800-episode budget for neural_raw in this environment, applied from the start. Holm-Bonferroni correction applied within each 2-cell family.",
        "",
        "## Reversal (neural_raw): fraction-of-required-adaptation, reversal vs. step-matched decorrelation",
        "",
        "| validity | reversal frac. adapted | decorrelation frac. adapted | t | raw p | holm p | Cohen's d | asymmetry supported |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in reversal_results:
        lines.append(
            f"| {r['validity']} | {r['reversal_fraction_adapted']:.3f} | {r['decorrelation_fraction_adapted']:.3f} | "
            f"{r['t_stat']:.3f} | {r['p_value']:.4f} | {r['holm_adjusted_p']:.4f} | {r['cohens_d']:.3f} | {r['asymmetry_supported']} |"
        )

    lines.extend(
        [
            "",
            "## Interaction-only (neural_interaction_only): own decorrelate-vs-naive effect, and coordinate-dependence test vs. neural_raw (step-matched)",
            "",
            "| validity | decorrelate | naive | own raw p | own holm p | has own effect | coord raw p | coord holm p | coordinate dependent |",
            "|---:|---:|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for r in interaction_results:
        lines.append(
            f"| {r['validity']} | {r['interaction_decorrelate_mean']:.3f} | {r['interaction_naive_mean']:.3f} | "
            f"{r['own_p_value']:.4f} | {r['own_holm_adjusted_p']:.4f} | {r['has_own_effect']} | "
            f"{r['coordinate_test_p_value']:.4f} | {r['coordinate_holm_adjusted_p']:.4f} | {r['coordinate_dependent']} |"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            f"Reversal: neural_raw gradient-asymmetry mechanism supported in the grid environment (after correction): {any(r['asymmetry_supported'] for r in reversal_results)}.",
            f"Interaction-only: neural_interaction_only shows its own decorrelate-vs-naive effect (after correction): {any(r['has_own_effect'] for r in interaction_results)}.",
            f"Interaction-only: coordinate-dependent relative to neural_raw (after correction): {any(r['coordinate_dependent'] for r in interaction_results)}.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_grid_neural_reversal_interaction_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
