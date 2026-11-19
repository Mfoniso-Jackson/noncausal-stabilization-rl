"""Cue Validity Recurrent Architecture Comparison experiment runner.

Pre-registered design: docs/cue_validity_recurrent_architecture.md. Reuses
env/cue_validity_env.py, metrics/cue_validity.py,
experiments/cue_validity_proxy_dependence_experiment.py::LinearAgentWrapper,
and experiments/cue_validity_nonlinear_experiment.py::NeuralAgentWrapper
unchanged (both imported directly, not reimplemented) for linear_raw and
neural_raw. Adds RecurrentAgentWrapper (a GRU over a rolling window of
observation vectors) with every hyperparameter that can be matched to
neural_raw held identical -- the fix to the confound documented in
docs/symbolic_stabilization_audit.md Section 6 and this experiment's own
pre-registration.
"""

import csv
import os
import random
import sys
from collections import deque

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mplconfig"))

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import compare_groups, cohens_d
from env.cue_validity_env import ACTION_CAUSAL, ACTION_CUE, NEUTRAL_ACTIONS, CueValidityEnv
from experiments.cue_validity_nonlinear_experiment import NeuralAgentWrapper
from experiments.cue_validity_proxy_dependence_experiment import LinearAgentWrapper
from metrics.cue_validity import cue_dependence_score, proxy_dependence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

AVAILABLE_ACTIONS = (ACTION_CAUSAL, ACTION_CUE) + NEUTRAL_ACTIONS
TARGET_PHASE = 4
VALIDITY_LEVELS = [0.5, 0.95]
CONDITIONS = ["linear_raw", "neural_raw", "recurrent_gru"]
SEEDS = list(range(15))
ACQUISITION_EPISODES = 3000
INTERVENTION_EPISODES = 1500
BLOCK_SIZE = 50
NUM_BLOCKS = INTERVENTION_EPISODES // BLOCK_SIZE
EPSILON_START = 1.0
EPSILON_END = 0.04
INTERVENTION_EPSILON = EPSILON_END

# Matched to neural_raw (cue_validity_nonlinear_experiment.py) exactly.
GAMMA = 0.95
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 8000
TARGET_UPDATE_INTERVAL = 200
TRAIN_EVERY_STEPS = 4
HIDDEN_SIZE = 32
SEQUENCE_LENGTH = 5  # the one architecture-necessary addition; see pre-registration

torch.set_num_threads(1)


def make_env(validity, seed):
    return CueValidityEnv(validity, cue_phase=TARGET_PHASE, target_phase=TARGET_PHASE, seed=seed)


class GRUQNet(nn.Module):
    def __init__(self, input_dim, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 4)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class RecurrentAgentWrapper:
    """Self-contained GRU agent: manages its own rolling observation
    window internally (mirroring LinearAgentWrapper's self-contained
    train_episode(env, epsilon) pattern), so the window-management logic
    never has to be threaded through the generic single-vector
    train_episode/features_for loop used for linear_raw/neural_raw."""

    def __init__(self, seed, feature_dim):
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.device = torch.device("cpu")
        self.online = GRUQNet(feature_dim).to(self.device)
        self.target = GRUQNet(feature_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.env_steps = 0
        self.window = deque(maxlen=SEQUENCE_LENGTH)

    def _reset_window(self, first_vector):
        self.window.clear()
        for _ in range(SEQUENCE_LENGTH):
            self.window.append(list(first_vector))

    def _window_snapshot(self):
        return [list(v) for v in self.window]

    def _greedy_action(self, seq, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(4)
        with torch.no_grad():
            x = torch.tensor([seq], dtype=torch.float32, device=self.device)
            values = self.online(x).cpu().tolist()[0]
        best = max(values)
        best_actions = [a for a in range(4) if abs(values[a] - best) < 1e-9]
        return self.rng.choice(best_actions)

    def train_episode(self, env, epsilon):
        env.reset()
        self._reset_window(env.observation_vector())
        state_seq = self._window_snapshot()
        done = False
        while not done:
            action = self._greedy_action(state_seq, epsilon)
            _, reward, done, _ = env.step(action)
            self.window.append(list(env.observation_vector()))
            next_state_seq = self._window_snapshot()
            self.replay.append((state_seq, action, reward, next_state_seq, done))
            self.env_steps += 1
            if self.env_steps % TRAIN_EVERY_STEPS == 0:
                self._train_step()
            if self.env_steps % TARGET_UPDATE_INTERVAL == 0:
                self.target.load_state_dict(self.online.state_dict())
            state_seq = next_state_seq

    def _train_step(self):
        if len(self.replay) < BATCH_SIZE:
            return
        batch = [self.replay[self.rng.randrange(len(self.replay))] for _ in range(BATCH_SIZE)]
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        q_pred = self.online(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            q_next = self.target(next_states_t).max(dim=1).values
            q_target = rewards_t + (1.0 - dones_t) * GAMMA * q_next
        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()

    def q_values_for_features(self, features):
        """Probe-time padding: repeat the probed single feature vector
        across the full window, since a synthetic probe state has no real
        preceding history -- matches neural_recurrent_reversal_experiment.py's
        own precedent for this, stated in the pre-registration."""
        seq = [list(features)] * SEQUENCE_LENGTH
        with torch.no_grad():
            x = torch.tensor([seq], dtype=torch.float32, device=self.device)
            return self.online(x).cpu().tolist()[0]


def make_agent(condition, seed):
    if condition == "linear_raw":
        return LinearAgentWrapper(seed, orthogonal=False)
    if condition == "neural_raw":
        return NeuralAgentWrapper(seed, 6)
    if condition == "recurrent_gru":
        return RecurrentAgentWrapper(seed, 6)
    raise ValueError(condition)


def train_episode(condition, agent, env, epsilon):
    # LinearAgentWrapper and RecurrentAgentWrapper are both self-contained
    # (their own train_episode(env, epsilon) method); only NeuralAgentWrapper
    # exposes the flat act/remember/train_step interface used by the
    # generic loop below.
    if condition in ("linear_raw", "recurrent_gru"):
        agent.train_episode(env, epsilon)
        return
    agent.epsilon = epsilon
    env.reset()
    state = env.observation_vector()
    done = False
    while not done:
        action = agent.act(state)
        _, reward, done, _ = env.step(action)
        next_state = env.observation_vector()
        agent.remember(state, action, reward, next_state, done)
        agent.train_step()
        state = next_state


def probe_q_values(condition, agent, validity, cue_present):
    probe_env = CueValidityEnv(validity, cue_phase=TARGET_PHASE, target_phase=TARGET_PHASE, seed=0)
    probe_env.phase = TARGET_PHASE
    probe_env.cue_present = int(cue_present)
    features = probe_env.observation_vector()
    values = agent.q_values_for_features(features)
    return {a: values[a] for a in AVAILABLE_ACTIONS}


def read_metrics(condition, agent, validity):
    q_present = probe_q_values(condition, agent, validity, True)
    q_absent = probe_q_values(condition, agent, validity, False)
    cds = cue_dependence_score(q_present, AVAILABLE_ACTIONS, ACTION_CAUSAL, ACTION_CUE, NEUTRAL_ACTIONS)
    dep = proxy_dependence(q_present, q_absent, AVAILABLE_ACTIONS, ACTION_CAUSAL)
    return cds, dep


def epsilon_schedule(total_episodes):
    decay = (EPSILON_END / EPSILON_START) ** (1.0 / total_episodes)
    epsilon = EPSILON_START
    for _ in range(total_episodes):
        yield epsilon
        epsilon = max(EPSILON_END, epsilon * decay)


def run_decorrelate_arm(condition, validity, seed):
    env = make_env(validity, seed)
    agent = make_agent(condition, seed)
    for epsilon in epsilon_schedule(ACQUISITION_EPISODES):
        train_episode(condition, agent, env, epsilon)
    acq_cds, acq_dep = read_metrics(condition, agent, validity)

    env.decorrelate()
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(condition, agent, env, INTERVENTION_EPSILON)
    final_cds, final_dep = read_metrics(condition, agent, validity)

    return {
        "label": "decorrelate",
        "condition": condition,
        "validity": validity,
        "seed": seed,
        "acquisition_cds": round(acq_cds, 6),
        "acquisition_proxy_dependence": round(acq_dep, 6),
        "final_cds": round(final_cds, 6),
        "final_proxy_dependence": round(final_dep, 6),
    }


def run_naive_arm(condition, validity, seed):
    env = make_env(validity, seed)
    env.decorrelate()
    agent = make_agent(condition, seed + 1_000_000)
    for _ in range(NUM_BLOCKS):
        for _ in range(BLOCK_SIZE):
            train_episode(condition, agent, env, INTERVENTION_EPSILON)
    final_cds, final_dep = read_metrics(condition, agent, validity)
    return {
        "label": "naive",
        "condition": condition,
        "validity": validity,
        "seed": seed,
        "acquisition_cds": None,
        "acquisition_proxy_dependence": None,
        "final_cds": round(final_cds, 6),
        "final_proxy_dependence": round(final_dep, 6),
    }


def mean(values):
    values = list(values)
    return sum(values) / max(1, len(values))


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


def gap_values(summaries, condition, validity, label):
    return [abs(r["final_proxy_dependence"]) for r in summaries if r["condition"] == condition and r["validity"] == validity and r["label"] == label]


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

    for condition in CONDITIONS:
        for validity in VALIDITY_LEVELS:
            for seed in SEEDS:
                print(f"recurrent-arch {condition} v{validity} decorrelate seed={seed}", flush=True)
                summaries.append(run_decorrelate_arm(condition, validity, seed))
                print(f"recurrent-arch {condition} v{validity} naive seed={seed}", flush=True)
                summaries.append(run_naive_arm(condition, validity, seed))

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_recurrent_architecture_summaries.csv"),
        summaries,
        ["label", "condition", "validity", "seed", "acquisition_cds", "acquisition_proxy_dependence", "final_cds", "final_proxy_dependence"],
    )

    results = []
    for condition in CONDITIONS:
        for validity in VALIDITY_LEVELS:
            decorrelate = gap_values(summaries, condition, validity, "decorrelate")
            naive = gap_values(summaries, condition, validity, "naive")
            test = compare_groups(decorrelate, naive)
            d = cohens_d(decorrelate, naive)
            results.append(
                {
                    "condition": condition,
                    "validity": validity,
                    "decorrelate_mean": round(mean(decorrelate), 6),
                    "naive_mean": round(mean(naive), 6),
                    "t_stat": round(float(test["t_stat"]), 6),
                    "p_value": round(float(test["p_value"]), 6),
                    "cohens_d": round(float(d), 6),
                }
            )

    results.sort(key=lambda r: (CONDITIONS.index(r["condition"]), r["validity"]))
    adjusted = holm_bonferroni([r["p_value"] for r in results])
    for r, adj in zip(results, adjusted):
        r["holm_adjusted_p"] = round(adj, 6)
        r["has_effect"] = bool(adj < 0.05 and r["decorrelate_mean"] > r["naive_mean"])

    write_csv(
        os.path.join(RESULTS_DIR, "cue_validity_recurrent_architecture_criteria.csv"),
        results,
        ["condition", "validity", "decorrelate_mean", "naive_mean", "t_stat", "p_value", "holm_adjusted_p", "cohens_d", "has_effect"],
    )

    lines = [
        "# Cue Validity Recurrent Architecture Comparison",
        "",
        "## Pre-registration",
        "Design and criteria were written and committed in docs/cue_validity_recurrent_architecture.md before any of the data below was collected. linear_raw and neural_raw rerun fresh in this script for full internal comparability; recurrent_gru uses hyperparameters matched to neural_raw wherever architecturally possible. Holm-Bonferroni correction applied across the full 6-cell family.",
        "",
        "## Results",
        "",
        "| condition | validity | decorrelate | naive | t | raw p | holm p | Cohen's d | has effect |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['condition']} | {r['validity']} | {r['decorrelate_mean']:.4f} | {r['naive_mean']:.4f} | "
            f"{r['t_stat']:.3f} | {r['p_value']:.4f} | {r['holm_adjusted_p']:.4f} | {r['cohens_d']:.3f} | {r['has_effect']} |"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            f"recurrent_gru shows the linear_raw/neural_raw pattern: {any(r['has_effect'] for r in results if r['condition'] == 'recurrent_gru')}.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "cue_validity_recurrent_architecture_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return results


if __name__ == "__main__":
    run()
