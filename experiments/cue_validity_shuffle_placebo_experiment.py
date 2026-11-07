"""Shuffled/placebo control experiment.

Pre-registered design: docs/cue_validity_shuffle_placebo.md. Reuses
experiments/cue_validity_proxy_dependence_experiment.py's
LinearAgentWrapper, make_agent, probe_q_values, read_metrics,
epsilon_schedule unchanged; only the environment differs
(ShuffleDecorrelateEnv, defined here, replaces decorrelate() with a
permutation of the realized acquisition cue sequence).
"""

import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import compare_groups, cohens_d
from env.cue_validity_env import ACTION_CAUSAL, CueValidityEnv
from experiments.cue_validity_proxy_dependence_experiment import (
    ACQUISITION_EPISODES,
    BLOCK_SIZE,
    INTERVENTION_EPSILON,
    NUM_BLOCKS,
    SEEDS,
    TARGET_PHASE,
    epsilon_schedule,
    make_agent,
    probe_q_values,
    read_metrics,
    train_episode,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
DECORRELATION_SUMMARY_CSV = os.path.join(RESULTS_DIR, "cue_validity_proxy_dependence_summaries.csv")

AGENT_TYPE = "linear_raw"
VALIDITY_LEVELS = [0.5, 0.95]


class ShuffleDecorrelateEnv(CueValidityEnv):
    def __init__(self, *args, **kwargs):
        self._logging = True
        self._acquisition_cue_log = []
        self._shuffle_rng = None
        self._shuffle_pool = []
        super().__init__(*args, **kwargs)

    def reset(self):
        obs = super().reset()
        if self._logging:
            self._acquisition_cue_log.append(self.cue_present)
        elif self._shuffle_rng is not None:
            if not self._shuffle_pool:
                self._refill_pool()
            self.cue_present = self._shuffle_pool.pop()
            obs = self.observe()
        return obs

    def start_shuffle_phase(self, shuffle_seed, source_log=None):
        self._logging = False
        self._shuffle_rng = random.Random(shuffle_seed)
        self._source_log = list(source_log if source_log is not None else self._acquisition_cue_log)
        self._refill_pool()

    def _refill_pool(self):
        pool = list(self._source_log)
        self._shuffle_rng.shuffle(pool)
        self._shuffle_pool = pool


def make_env(validity, seed):
    return ShuffleDecorrelateEnv(validity, cue_phase=TARGET_PHASE, target_phase=TARGET_PHASE, seed=seed)


def run_shuffle_arm(validity, seed):
    env = make_env(validity, seed)
    agent = make_agent(AGENT_TYPE, env, seed)
    for epsilon in epsilon_schedule(ACQUISITION_EPISODES):
        train_episode(AGENT_TYPE, agent, env, epsilon)
    acq_cds, acq_dep = read_metrics(AGENT_TYPE, agent, validity)

    env.start_shuffle_phase(shuffle_seed=seed + 2_000_000)

    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(AGENT_TYPE, agent, env, INTERVENTION_EPSILON)
    _, final_dep = read_metrics(AGENT_TYPE, agent, validity)

    return {
        "label": "shuffle",
        "agent_type": AGENT_TYPE,
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_block_proxy_dependence": round(final_dep, 6),
    }


def run_naive_shuffle_arm(validity, seed):
    # Build a separate acquisition-phase cue log, from a different seed,
    # to serve as this naive agent's shuffle pool -- it never sees the
    # evaluation seed's own realized sequence.
    source_env = ShuffleDecorrelateEnv(validity, cue_phase=TARGET_PHASE, target_phase=TARGET_PHASE, seed=seed + 3_000_000)
    for _ in range(ACQUISITION_EPISODES):
        source_env.reset()
        done = False
        while not done:
            _, _, done, _ = source_env.step(ACTION_CAUSAL)
    source_log = source_env._acquisition_cue_log

    env = make_env(validity, seed)
    env.start_shuffle_phase(shuffle_seed=seed + 4_000_000, source_log=source_log)
    agent = make_agent(AGENT_TYPE, env, seed + 1_000_000)

    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(AGENT_TYPE, agent, env, INTERVENTION_EPSILON)
    _, final_dep = read_metrics(AGENT_TYPE, agent, validity)

    return {
        "label": "naive_shuffle",
        "agent_type": AGENT_TYPE,
        "validity": validity,
        "seed": seed,
        "acquisition_proxy_dependence": None,
        "final_block_proxy_dependence": round(final_dep, 6),
    }


def mean(values):
    values = list(values)
    return sum(values) / max(1, len(values))


def per_seed_shuffle_gap(summaries, validity):
    shuffle_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in summaries
        if row["validity"] == validity and row["label"] == "shuffle"
    }
    naive_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in summaries
        if row["validity"] == validity and row["label"] == "naive_shuffle"
    }
    return [abs(shuffle_by_seed[s] - naive_by_seed[s]) for s in sorted(shuffle_by_seed) if s in naive_by_seed]


def load_decorrelate_gap(validity):
    with open(DECORRELATION_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["validity"] = float(row["validity"])
        row["seed"] = int(row["seed"])
        row["final_block_proxy_dependence"] = (
            None if row["final_block_proxy_dependence"] in ("", "None") else float(row["final_block_proxy_dependence"])
        )
    decorrelate_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in rows
        if row["agent_type"] == AGENT_TYPE and row["validity"] == validity and row["label"] == f"v{validity}_decorrelate"
    }
    naive_by_seed = {
        row["seed"]: row["final_block_proxy_dependence"]
        for row in rows
        if row["agent_type"] == AGENT_TYPE and row["validity"] == validity and row["label"] == f"v{validity}_naive"
    }
    return [abs(decorrelate_by_seed[s] - naive_by_seed[s]) for s in sorted(decorrelate_by_seed) if s in naive_by_seed]


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
            print(f"v{validity} shuffle seed={seed}", flush=True)
            summaries.append(run_shuffle_arm(validity, seed))

            print(f"v{validity} naive_shuffle seed={seed}", flush=True)
            summaries.append(run_naive_shuffle_arm(validity, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_shuffle_placebo_summaries.csv"),
        summaries,
        ["label", "agent_type", "validity", "seed", "acquisition_proxy_dependence", "final_block_proxy_dependence"],
    )

    results = []
    for validity in VALIDITY_LEVELS:
        shuffle_gap = per_seed_shuffle_gap(summaries, validity)
        decorrelate_gap = load_decorrelate_gap(validity)
        test = compare_groups(shuffle_gap, decorrelate_gap)
        d = cohens_d(shuffle_gap, decorrelate_gap)
        results.append(
            {
                "validity": validity,
                "shuffle_gap_mean": round(mean(shuffle_gap), 6),
                "decorrelate_gap_mean": round(mean(decorrelate_gap), 6),
                "t_stat": round(float(test["t_stat"]), 6),
                "p_value": round(float(test["p_value"]), 6),
                "cohens_d": round(float(d), 6),
                "placebo_concern_ruled_out": bool(test["p_value"] >= 0.05),
            }
        )

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_shuffle_placebo_criteria.csv"),
        results,
        ["validity", "shuffle_gap_mean", "decorrelate_gap_mean", "t_stat", "p_value", "cohens_d", "placebo_concern_ruled_out"],
    )

    all_ruled_out = all(r["placebo_concern_ruled_out"] for r in results)

    lines = [
        "# Cue Validity Shuffled/Placebo Control",
        "",
        "## Pre-registration",
        "Design and criteria were written and committed in docs/cue_validity_shuffle_placebo.md before any of the data below was collected.",
        "",
        "## Design summary",
        f"linear_raw only, cue_validity in {VALIDITY_LEVELS}, {len(SEEDS)} seeds. Shuffle arm replaces decorrelate()'s fresh i.i.d. resampling with a permutation of the realized acquisition cue sequence. Compared against the already-committed decorrelate arm from cue_validity_proxy_dependence_experiment.py.",
        "",
        "## Results",
        "",
        "| validity | shuffle gap | decorrelate gap | t | p | Cohen's d | placebo concern ruled out |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['validity']} | {row['shuffle_gap_mean']:.3f} | {row['decorrelate_gap_mean']:.3f} | "
            f"{row['t_stat']:.3f} | {row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['placebo_concern_ruled_out']} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            f"**Placebo-mechanism concern ruled out at both validity levels: {all_ruled_out}.** "
            "(Shuffle and decorrelate gaps are statistically indistinguishable, meaning decorrelate()'s own resampling mechanism is not itself contributing to the measured persistence effect beyond the mere removal of cue information.)" if all_ruled_out else
            f"**Placebo-mechanism concern NOT ruled out at all levels: {all_ruled_out}.** See per-level results above.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_shuffle_placebo_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
