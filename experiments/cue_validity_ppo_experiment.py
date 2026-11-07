"""Cue Validity PPO experiment runner.

Pre-registered design: docs/cue_validity_ppo.md. Reuses
env/cue_validity_env.py, metrics/cue_validity.py, and
experiments/cue_validity_proxy_dependence_experiment.py's make_env,
epsilon-free block/seed conventions unchanged; the only new piece is
plugging agents.ppo_agent.PPOAgent into the same pipeline.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import PPOAgent
from analysis.stats import compare_groups, cohens_d
from env.cue_validity_env import ACTION_CAUSAL, ACTION_CUE, NEUTRAL_ACTIONS, CueValidityEnv
from experiments.cue_validity_proxy_dependence_experiment import (
    ACQUISITION_EPISODES,
    BLOCK_SIZE,
    NUM_BLOCKS,
    SEEDS,
    TARGET_PHASE,
    make_env,
    mean,
)
from metrics.cue_validity import cue_dependence_score, proxy_dependence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

AVAILABLE_ACTIONS = (ACTION_CAUSAL, ACTION_CUE) + NEUTRAL_ACTIONS
VALIDITY_LEVELS = [0.5, 0.95]
CONDITION = "ppo"


def run_rollout_block(agent, env, num_episodes):
    for _ in range(num_episodes):
        env.reset()
        state = env.observation_vector()
        done = False
        while not done:
            action = agent.act(state)
            _, reward, done, _ = env.step(action)
            agent.record_outcome(reward, done)
            state = env.observation_vector()
    agent.update(last_value=0.0)


def probe_logits(agent, validity, cue_present):
    probe_env = CueValidityEnv(validity, cue_phase=TARGET_PHASE, target_phase=TARGET_PHASE, seed=0)
    probe_env.phase = TARGET_PHASE
    probe_env.cue_present = int(cue_present)
    logits = agent.policy_logits_for_features(probe_env.observation_vector())
    return {a: logits[a] for a in AVAILABLE_ACTIONS}


def read_metrics(agent, validity):
    q_present = probe_logits(agent, validity, True)
    q_absent = probe_logits(agent, validity, False)
    cds = cue_dependence_score(q_present, AVAILABLE_ACTIONS, ACTION_CAUSAL, ACTION_CUE, NEUTRAL_ACTIONS)
    dep = proxy_dependence(q_present, q_absent, AVAILABLE_ACTIONS, ACTION_CAUSAL)
    return cds, dep


def run_decorrelate_arm(validity, seed):
    env = make_env(validity, seed)
    agent = PPOAgent(feature_dim=6, seed=seed)

    acquisition_blocks = ACQUISITION_EPISODES // BLOCK_SIZE
    for _ in range(acquisition_blocks):
        run_rollout_block(agent, env, BLOCK_SIZE)
    acq_cds, acq_dep = read_metrics(agent, validity)

    env.decorrelate()
    for _ in range(NUM_BLOCKS):
        run_rollout_block(agent, env, BLOCK_SIZE)
    final_cds, final_dep = read_metrics(agent, validity)

    return {
        "label": "decorrelate",
        "condition": CONDITION,
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_proxy_dependence": round(final_dep, 6),
    }


def run_naive_arm(validity, seed):
    env = make_env(validity, seed)
    env.decorrelate()
    agent = PPOAgent(feature_dim=6, seed=seed + 1_000_000)
    for _ in range(NUM_BLOCKS):
        run_rollout_block(agent, env, BLOCK_SIZE)
    final_cds, final_dep = read_metrics(agent, validity)
    return {
        "label": "naive",
        "condition": CONDITION,
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": None,
        "final_proxy_dependence": round(final_dep, 6),
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
            print(f"ppo v{validity} decorrelate seed={seed}", flush=True)
            summaries.append(run_decorrelate_arm(validity, seed))
            print(f"ppo v{validity} naive seed={seed}", flush=True)
            summaries.append(run_naive_arm(validity, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_ppo_summaries.csv"),
        summaries,
        ["label", "condition", "validity", "seed", "acquisition_proxy_dependence", "final_proxy_dependence"],
    )

    results = []
    for validity in VALIDITY_LEVELS:
        decorrelate = gap_values(summaries, validity, "decorrelate")
        naive = gap_values(summaries, validity, "naive")
        test = compare_groups(decorrelate, naive)
        d = cohens_d(decorrelate, naive)
        results.append(
            {
                "validity": validity,
                "decorrelate_mean": round(mean(decorrelate), 6),
                "naive_mean": round(mean(naive), 6),
                "t_stat": round(float(test["t_stat"]), 6),
                "p_value": round(float(test["p_value"]), 6),
                "cohens_d": round(float(d), 6),
                "has_effect": bool(test["p_value"] < 0.05 and mean(decorrelate) > mean(naive)),
            }
        )

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_ppo_criteria.csv"),
        results,
        ["validity", "decorrelate_mean", "naive_mean", "t_stat", "p_value", "cohens_d", "has_effect"],
    )

    any_effect = any(r["has_effect"] for r in results)
    all_null = not any_effect

    lines = [
        "# Cue Validity PPO Experiment",
        "",
        "## Pre-registration",
        "Design and criteria were written and committed in docs/cue_validity_ppo.md before any of the data below was collected.",
        "",
        "## Design summary",
        f"ppo (agents.ppo_agent.PPOAgent), cue_validity in {VALIDITY_LEVELS}, {len(SEEDS)} seeds, decorrelate intervention. Same episode budget as the rest of this lineage (3000 acquisition, 1500 post-intervention in 30 blocks of 50); each block is one PPO rollout+update.",
        "",
        "## Results",
        "",
        "| validity | decorrelate | naive | t | p | Cohen's d | has effect |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['validity']} | {row['decorrelate_mean']:.3f} | {row['naive_mean']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['has_effect']} |"
        )
    lines.extend(
        [
            "",
            "## Verdict (H4: do different algorithms show different patterns?)",
            f"PPO shows the linear_raw/neural_raw pattern (significant effect at >=1 validity level): {any_effect}.",
            f"PPO shows the tabular pattern (null at both levels): {all_null}.",
            "",
            "## Limitations",
            "PPO's hyperparameters (lr, clip_eps, epochs, entropy_coef, etc.) are agents/ppo_agent.py's defaults, not tuned for this specific task -- a differently-tuned PPO could behave differently. Only one seed-per-rollout-shuffle ordering and one network architecture (1 hidden layer, 32 units) are tested.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_ppo_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
