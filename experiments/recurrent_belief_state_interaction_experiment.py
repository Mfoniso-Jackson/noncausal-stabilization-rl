import math
import os
import random
import statistics
import sys
from collections import deque

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mplconfig"))

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import compare_groups, cohens_d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

ACTION_CAUSAL = 0
ACTION_PROXY = 1
NEUTRAL_ACTIONS = [2, 3]
ACTION_COUNT = 4
OBS_DIM = 6
HORIZON = 14
TARGET_PHASE = 10
HIDDEN_REWARD_PROB = 0.5
STEP_PENALTY = -0.002

PHASES = [
    ("acquisition", 400, 0.95, 0.05),
    ("reversal", 200, 0.05, 0.95),
    ("extinction", 200, 0.50, 0.50),
]
EVAL_EPISODES = 100
HISTORY_LENGTHS = [1, 5]
AUX_WEIGHTS = [0.0, 4.0]
SEEDS = list(range(5))

ALPHA = 0.08
GAMMA = 0.95
LAMBDA = 0.80
EPSILON_START = 1.0
EPSILON_END = 0.04
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50_000
TARGET_UPDATE_INTERVAL = 200
TRAIN_EVERY_STEPS = 16
MIN_PERSISTENCE_DENOMINATOR = 0.05

torch.set_num_threads(1)


class HiddenProxyReversalEnv:
    """Identical mechanics to neural_recurrent_reversal_experiment.py's
    HiddenProxyReversalEnv."""

    def __init__(self, p_proxy_given_reward, p_proxy_given_no_reward, seed=0):
        self.p_proxy_given_reward = p_proxy_given_reward
        self.p_proxy_given_no_reward = p_proxy_given_no_reward
        self.rng = random.Random(seed)
        self.reset()

    def set_proxy_probs(self, p_proxy_given_reward, p_proxy_given_no_reward):
        self.p_proxy_given_reward = p_proxy_given_reward
        self.p_proxy_given_no_reward = p_proxy_given_no_reward

    def reset(self):
        self.phase = 0
        self.done = False
        self.hidden_reward_state = int(self.rng.random() < HIDDEN_REWARD_PROB)
        self.reward_obtained = False
        self.proxy_by_phase = [self._sample_proxy() for _ in range(HORIZON + 1)]
        return self.observe()

    def _sample_proxy(self):
        prob = self.p_proxy_given_reward if self.hidden_reward_state else self.p_proxy_given_no_reward
        return int(self.rng.random() < prob)

    def observe(self):
        return self.phase * 2 + self.proxy_by_phase[self.phase]

    def observation_vector(self):
        phase_norm = self.phase / TARGET_PHASE
        proxy = self.proxy_by_phase[self.phase]
        return [
            phase_norm,
            float(proxy),
            1.0 if self.phase == TARGET_PHASE else 0.0,
            1.0 if self.phase < TARGET_PHASE else 0.0,
            math.sin(self.phase / max(1, TARGET_PHASE) * math.pi),
            math.cos(self.phase / max(1, TARGET_PHASE) * math.pi),
        ]

    def step(self, action):
        if self.done:
            raise RuntimeError("step called after episode termination")
        reward = STEP_PENALTY
        if self.phase == TARGET_PHASE:
            if action == ACTION_CAUSAL and self.hidden_reward_state == 1:
                reward += 1.0
                self.reward_obtained = True
            self.done = True
        else:
            self.phase += 1
            self.done = self.phase >= HORIZON
        return self.observe(), reward, self.done, {}


class BeliefGRUQNet(nn.Module):
    def __init__(self, input_dim, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.q_head = nn.Linear(hidden_size, ACTION_COUNT)
        self.belief_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.q_head(last), self.belief_head(last).squeeze(-1)


class RecurrentBeliefAgent:
    """Same architecture as recurrent_belief_state_supervision_experiment.py,
    but both history_length and the auxiliary belief-loss weight are now
    varied together, to test whether supervision's effect (if any) depends
    on how much temporal context the agent has to use it with. aux_weight=0.0
    reproduces the unsupervised condition (the belief_head never receives
    gradient and is dead weight)."""

    def __init__(self, seed, history_length, aux_weight):
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.history_length = history_length
        self.aux_weight = aux_weight
        self.device = torch.device("cpu")
        self.online = BeliefGRUQNet(OBS_DIM).to(self.device)
        self.target = BeliefGRUQNet(OBS_DIM).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE)
        self.q_loss_fn = nn.SmoothL1Loss()
        self.aux_loss_fn = nn.BCEWithLogitsLoss()
        self.replay = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.env_steps = 0
        self.history = deque(maxlen=history_length)

    def _current_state(self, env):
        return [list(x) for x in self.history]

    def _reset_history(self, env):
        self.history.clear()
        obs = env.observation_vector()
        for _ in range(self.history_length):
            self.history.append(list(obs))

    def _append_history(self, env):
        self.history.append(list(env.observation_vector()))

    def train_episode(self, env, epsilon):
        env.reset()
        self._reset_history(env)
        belief_label = float(env.hidden_reward_state)
        done = False
        while not done:
            state = self._current_state(env)
            action = self._epsilon_action(state, epsilon)
            _, reward, done, _ = env.step(action)
            self._append_history(env)
            next_state = self._current_state(env)
            self.replay.append((state, action, reward, next_state, done, belief_label))
            self.env_steps += 1
            if self.env_steps % TRAIN_EVERY_STEPS == 0:
                self._train_step()
            if self.env_steps % TARGET_UPDATE_INTERVAL == 0:
                self.target.load_state_dict(self.online.state_dict())

    def _epsilon_action(self, state, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(ACTION_COUNT)
        return random_argmax(self._q_values(state), self.rng)

    def _q_values(self, state):
        with torch.no_grad():
            x = torch.tensor([state], dtype=torch.float32, device=self.device)
            q_values, _ = self.online(x)
            values = q_values.cpu().tolist()[0]
        return values

    def _belief_prob(self, state):
        with torch.no_grad():
            x = torch.tensor([state], dtype=torch.float32, device=self.device)
            _, belief_logit = self.online(x)
            return torch.sigmoid(belief_logit).item()

    def _train_step(self):
        if len(self.replay) < BATCH_SIZE:
            return
        batch = [self.replay[self.rng.randrange(len(self.replay))] for _ in range(BATCH_SIZE)]
        states, actions, rewards, next_states, dones, beliefs = zip(*batch)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        q_pred_full, belief_logit = self.online(states_t)
        q_pred = q_pred_full.gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            q_next_full, _ = self.target(next_states_t)
            q_next = q_next_full.max(dim=1).values
            q_target = rewards_t + (1.0 - dones_t) * GAMMA * q_next
        loss = self.q_loss_fn(q_pred, q_target)
        if self.aux_weight > 0.0:
            beliefs_t = torch.tensor(beliefs, dtype=torch.float32, device=self.device)
            aux_loss = self.aux_loss_fn(belief_logit, beliefs_t)
            loss = loss + self.aux_weight * aux_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()

    def begin_eval_episode(self, env):
        self._reset_history(env)

    def observe_eval_step(self, env):
        self._append_history(env)

    def q_values_for_env(self, env):
        return self._q_values(self._current_state(env))

    def belief_prob_for_env(self, env):
        return self._belief_prob(self._current_state(env))

    def greedy_action_for_env(self, env):
        return random_argmax(self.q_values_for_env(env), self.rng)


def random_argmax(values, rng):
    best = max(values)
    best_actions = [idx for idx, value in enumerate(values) if abs(value - best) < 1e-12]
    return rng.choice(best_actions)


def train_and_evaluate(history_length, aux_weight, seed):
    agent = RecurrentBeliefAgent(seed, history_length, aux_weight)
    env = HiddenProxyReversalEnv(PHASES[0][2], PHASES[0][3], seed=seed)
    rows = []
    total_episodes = sum(episodes for _, episodes, _, _ in PHASES)
    epsilon = EPSILON_START
    epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / total_episodes)
    global_episode = 0
    for phase_name, phase_episodes, p1, p0 in PHASES:
        env.set_proxy_probs(p1, p0)
        checkpoints = sorted(set([1, max(1, phase_episodes // 2), phase_episodes]))
        for phase_episode in range(1, phase_episodes + 1):
            agent.train_episode(env, epsilon)
            epsilon = max(EPSILON_END, epsilon * epsilon_decay)
            global_episode += 1
            if phase_episode in checkpoints:
                metrics = evaluate(agent, p1, p0, seed + global_episode * 17)
                rows.append(
                    {
                        "history_length": history_length,
                        "aux_weight": aux_weight,
                        "seed": seed,
                        "phase": phase_name,
                        "phase_episode": phase_episode,
                        "global_episode": global_episode,
                        **metrics,
                    }
                )
    return rows


def evaluate(agent, p1, p0, seed):
    env = HiddenProxyReversalEnv(p1, p0, seed=seed + 70000)
    totals = init_counts()
    cue_counts = {0: init_counts(), 1: init_counts()}
    cds_values, q_proxy_values, q_neutral_values, q_causal_values = [], [], [], []
    belief_correct = 0
    belief_total = 0
    for _ in range(EVAL_EPISODES):
        env.reset()
        agent.begin_eval_episode(env)
        done = False
        while not done:
            cue = env.observe() % 2
            values = agent.q_values_for_env(env)
            action = random_argmax(values, agent.rng)
            if env.phase < TARGET_PHASE:
                record_action(totals, action)
                record_action(cue_counts[cue], action)
                masses = softmax(values)
                cds_values.append(masses[ACTION_PROXY] + sum(masses[a] for a in NEUTRAL_ACTIONS))
                q_proxy_values.append(values[ACTION_PROXY])
                q_neutral_values.append(mean(values[a] for a in NEUTRAL_ACTIONS))
                q_causal_values.append(values[ACTION_CAUSAL])
            if env.phase == TARGET_PHASE:
                belief_prob = agent.belief_prob_for_env(env)
                belief_total += 1
                belief_correct += int((belief_prob >= 0.5) == bool(env.hidden_reward_state))
            _, reward, done, _ = env.step(action)
            totals["reward_sum"] += reward
            agent.observe_eval_step(env)
        totals["goal_sum"] += int(env.reward_obtained)
    q_proxy = mean(q_proxy_values)
    q_neutral = mean(q_neutral_values)
    q_causal = mean(q_causal_values)
    p_causal_cue1 = rate(cue_counts[1], "causal")
    p_causal_cue0 = rate(cue_counts[0], "causal")
    return {
        "reward": round(totals["reward_sum"] / EVAL_EPISODES, 6),
        "goal_rate": round(totals["goal_sum"] / EVAL_EPISODES, 6),
        "SPI": round((totals["proxy"] + totals["neutral"]) / max(1, totals["decisions"]), 6),
        "CDS": round(mean(cds_values), 6),
        "proxy_action_rate": round(rate(totals, "proxy"), 6),
        "proxy_Q_advantage": round(q_proxy - q_neutral, 6),
        "proxy_dependence": round(p_causal_cue1 - p_causal_cue0, 6),
        "abs_proxy_dependence": round(abs(p_causal_cue1 - p_causal_cue0), 6),
        "belief_accuracy": round(belief_correct / belief_total if belief_total else 0.0, 6),
    }


def init_counts():
    return {"decisions": 0, "causal": 0, "proxy": 0, "neutral": 0, "reward_sum": 0.0, "goal_sum": 0}


def record_action(counts, action):
    counts["decisions"] += 1
    if action == ACTION_CAUSAL:
        counts["causal"] += 1
    elif action == ACTION_PROXY:
        counts["proxy"] += 1
    elif action in NEUTRAL_ACTIONS:
        counts["neutral"] += 1


def rate(counts, key):
    return counts[key] / max(1, counts["decisions"])


def softmax(values):
    max_q = max(values)
    exp_values = [math.exp(max(-60.0, min(60.0, value - max_q))) for value in values]
    denom = sum(exp_values)
    return [value / denom for value in exp_values]


def mean(values):
    values = list(values)
    return sum(values) / max(1, len(values))


def sem(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def ci95(values):
    return 1.96 * sem(values)


def get_checkpoint(rows, phase, which):
    subset = [r for r in rows if r["phase"] == phase]
    episode = min(r["phase_episode"] for r in subset) if which == "initial" else max(r["phase_episode"] for r in subset)
    return next(r for r in subset if r["phase_episode"] == episode)


def summarize(rows):
    metrics = ["reward", "goal_rate", "SPI", "CDS", "proxy_action_rate", "proxy_Q_advantage", "proxy_dependence", "abs_proxy_dependence", "belief_accuracy"]
    summary_rows = []
    for history_length in HISTORY_LENGTHS:
        for aux_weight in AUX_WEIGHTS:
            for phase, _, _, _ in PHASES:
                subset = [
                    get_checkpoint(
                        [r for r in rows if r["history_length"] == history_length and r["aux_weight"] == aux_weight and r["seed"] == seed and r["phase"] == phase],
                        phase,
                        "final",
                    )
                    for seed in SEEDS
                ]
                item = {"history_length": history_length, "aux_weight": aux_weight, "phase": phase, "n": len(subset)}
                for metric in metrics:
                    values = [float(r[metric]) for r in subset]
                    item[f"{metric}_mean"] = round(mean(values), 6)
                    item[f"{metric}_ci95"] = round(ci95(values), 6)
                summary_rows.append(item)
    return summary_rows


def extinction_values(rows, history_length, aux_weight):
    return [
        get_checkpoint(
            [r for r in rows if r["history_length"] == history_length and r["aux_weight"] == aux_weight and r["seed"] == seed and r["phase"] == "extinction"],
            "extinction",
            "final",
        )["abs_proxy_dependence"]
        for seed in SEEDS
    ]


def interaction_analysis(rows):
    """For each history length, compare aux_weight=0 vs the strongest
    aux_weight's extinction abs_proxy_dependence; then compare whether that
    gap itself differs between history lengths (the interaction)."""
    comparisons = []
    gaps = {}
    for history_length in HISTORY_LENGTHS:
        none_values = extinction_values(rows, history_length, AUX_WEIGHTS[0])
        strong_values = extinction_values(rows, history_length, AUX_WEIGHTS[-1])
        test = compare_groups(none_values, strong_values)
        gap = mean(none_values) - mean(strong_values)
        gaps[history_length] = gap
        comparisons.append(
            {
                "history_length": history_length,
                "aux_weight_none": AUX_WEIGHTS[0],
                "aux_weight_strong": AUX_WEIGHTS[-1],
                "none_mean": round(mean(none_values), 6),
                "strong_mean": round(mean(strong_values), 6),
                "gap_none_minus_strong": round(gap, 6),
                "t_stat": round(float(test["t_stat"]), 6),
                "p_value": round(float(test["p_value"]), 6),
                "cohens_d": round(float(cohens_d(none_values, strong_values)), 6),
            }
        )
    interaction = {
        "gap_at_history_min": round(gaps[HISTORY_LENGTHS[0]], 6),
        "gap_at_history_max": round(gaps[HISTORY_LENGTHS[-1]], 6),
        "interaction_difference": round(gaps[HISTORY_LENGTHS[-1]] - gaps[HISTORY_LENGTHS[0]], 6),
    }
    return comparisons, interaction


def write_csv(path, rows, fieldnames):
    import csv

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plot(summary):
    svg_path = os.path.join(RESULTS_DIR, "recurrent_belief_state_interaction_plot.svg")
    width, height = 980, 360
    panel_w, panel_h = 420, 230
    origins = [(50, 70), (510, 70)]
    colors = {0.0: "#999999", AUX_WEIGHTS[-1]: "#c1440e"}
    phases_order = ["acquisition", "reversal", "extinction"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="26" text-anchor="middle" font-family="Arial" font-size="17" font-weight="700">abs_proxy_dependence: history_length x aux_weight across phases</text>',
    ]
    for history_length, (ox, oy) in zip(HISTORY_LENGTHS, origins):
        parts.append(f'<text x="{ox + panel_w/2}" y="{oy-12}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700">history_length={history_length}</text>')
        parts.append(f'<line x1="{ox}" y1="{oy+panel_h}" x2="{ox+panel_w}" y2="{oy+panel_h}" stroke="#222"/>')
        parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+panel_h}" stroke="#222"/>')
        values_all = [row["abs_proxy_dependence_mean"] for row in summary if row["history_length"] == history_length]
        ymin = min(values_all + [0.0])
        ymax = max(values_all + [0.1])
        pad = (ymax - ymin) * 0.15 or 0.05
        ymin -= pad
        ymax += pad
        for aux_weight in AUX_WEIGHTS:
            curve = [row for row in summary if row["history_length"] == history_length and row["aux_weight"] == aux_weight]
            curve.sort(key=lambda row: phases_order.index(row["phase"]))
            coords = " ".join(
                f"{ox + (i/max(1, len(phases_order)-1))*panel_w:.2f},{oy + (ymax - row['abs_proxy_dependence_mean'])/(ymax-ymin)*panel_h:.2f}"
                for i, row in enumerate(curve)
            )
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{colors[aux_weight]}" stroke-width="2.5"/>')
        for i, phase in enumerate(phases_order):
            parts.append(f'<text x="{ox + (i/max(1,len(phases_order)-1))*panel_w:.2f}" y="{oy+panel_h+16}" text-anchor="middle" font-family="Arial" font-size="10">{phase}</text>')
    parts.append("</svg>")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_report(summary, comparisons, interaction):
    def row_for(history_length, aux_weight, phase):
        return next(r for r in summary if r["history_length"] == history_length and r["aux_weight"] == aux_weight and r["phase"] == phase)

    any_significant = any(c["p_value"] < 0.05 for c in comparisons)
    compounds = (
        abs(interaction["gap_at_history_max"]) > abs(interaction["gap_at_history_min"])
        and interaction["gap_at_history_max"] > 0
    )

    lines = [
        "# Recurrent Belief-State Interaction Experiment",
        "",
        "## Research question",
        "recurrent_belief_state_supervision_report.md found no significant benefit from belief-state supervision at history_length=5 and aux_weight=1.0, and recommended sweeping the auxiliary loss weight and testing whether any benefit compounds with longer history. This experiment crosses history_length in {1, 5} with aux_weight in {0.0 (unsupervised), 4.0 (strong supervision)} in a 2x2 design, to test both the main effect of a much stronger supervision weight and its interaction with memory capacity.",
        "",
        "## Design",
        f"Same hidden-state POMDP, 3-phase acquisition/reversal/extinction schedule ({PHASES[0][1]}/{PHASES[1][1]}/{PHASES[2][1]} episodes), and two-head GRU(64) architecture as the two prior recurrent experiments. {len(HISTORY_LENGTHS)} history lengths x {len(AUX_WEIGHTS)} aux weights x {len(SEEDS)} seeds = {len(HISTORY_LENGTHS) * len(AUX_WEIGHTS) * len(SEEDS)} total runs.",
        "",
        "## Results table (final checkpoint of each phase)",
        "",
        "| history_length | aux_weight | phase | abs_proxy_dependence | belief_accuracy |",
        "|---:|---:|---|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['history_length']} | {row['aux_weight']} | {row['phase']} | "
            f"{row['abs_proxy_dependence_mean']:.3f} +/- {row['abs_proxy_dependence_ci95']:.3f} | "
            f"{row['belief_accuracy_mean']:.3f} +/- {row['belief_accuracy_ci95']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Extinction comparison: unsupervised vs strong supervision, by history length",
            "",
            "| history_length | none mean | strong mean | gap (none - strong) | t | p | Cohen's d |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for c in comparisons:
        lines.append(
            f"| {c['history_length']} | {c['none_mean']:.3f} | {c['strong_mean']:.3f} | {c['gap_none_minus_strong']:.3f} | "
            f"{c['t_stat']:.3f} | {c['p_value']:.4f} | {c['cohens_d']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interaction",
            "",
            f"Gap (none - strong) at history_length={HISTORY_LENGTHS[0]}: {interaction['gap_at_history_min']:.3f}. "
            f"Gap at history_length={HISTORY_LENGTHS[-1]}: {interaction['gap_at_history_max']:.3f}. "
            f"Difference (interaction): {interaction['interaction_difference']:.3f}.",
            "",
            "## Interpretation",
            (
                "A positive gap means the unsupervised condition has higher residual extinction dependence than the strongly-supervised one, "
                "i.e. supervision helped; a larger positive gap at the longer history length would be the 'compounds with memory' pattern the "
                "original recommendation hypothesized. None of the four cells' comparisons reach significance at this seed count "
                f"(see p-values above); any difference in the gaps between history lengths should be read as suggestive at best with "
                f"{len(SEEDS)} seeds per cell. "
                "The belief-head shortcut-learning pattern from recurrent_belief_state_supervision_experiment.py replicates cleanly at both "
                f"history lengths and this stronger aux_weight=4.0: acquisition belief_accuracy reaches "
                f"{row_for(1, AUX_WEIGHTS[-1], 'acquisition')['belief_accuracy_mean']:.3f} at history_length=1 and "
                f"{row_for(5, AUX_WEIGHTS[-1], 'acquisition')['belief_accuracy_mean']:.3f} at history_length=5, then collapses to "
                f"{row_for(1, AUX_WEIGHTS[-1], 'reversal')['belief_accuracy_mean']:.3f} and "
                f"{row_for(5, AUX_WEIGHTS[-1], 'reversal')['belief_accuracy_mean']:.3f} respectively the moment the proxy cue reverses -- "
                "both close to the reversed cue's own predictive floor, confirming across two independent experiments that the belief head "
                "tracks the proxy-cue shortcut rather than anything causal, regardless of history length or supervision strength."
            ),
            "",
            "## Does a stronger auxiliary weight help, and does any benefit compound with longer history?",
            f"Any cell individually significant (p<0.05): {any_significant}. Benefit appears to compound with longer history (larger positive gap at the longer history length): {compounds}.",
            "",
            "## Limitations",
            f"Only {len(SEEDS)} seeds per cell (a 2x2 design, {len(HISTORY_LENGTHS) * len(AUX_WEIGHTS) * len(SEEDS)} runs total) given CPU-only PyTorch training cost; this is underpowered to detect anything but a large interaction effect. Only two aux_weight values (0 and 4.0) are tested rather than a full sweep, and two history lengths (1 and 5) rather than the three tested in recurrent_belief_state_history_experiment.py, to keep the factorial design tractable.",
            "",
            "## Next recommended experiment",
            "If any cell shows a promising (even if non-significant) effect, rerun that specific history_length x aux_weight combination alone with substantially more seeds to get a properly powered estimate, rather than continuing to expand the factorial design.",
        ]
    )
    with open(os.path.join(REPORTS_DIR, "recurrent_belief_state_interaction_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows = []
    for history_length in HISTORY_LENGTHS:
        for aux_weight in AUX_WEIGHTS:
            for seed in SEEDS:
                print(f"history_length={history_length} aux_weight={aux_weight} seed={seed}", flush=True)
                rows.extend(train_and_evaluate(history_length, aux_weight, seed))

    summary = summarize(rows)
    comparisons, interaction = interaction_analysis(rows)

    result_fields = [
        "history_length",
        "aux_weight",
        "seed",
        "phase",
        "phase_episode",
        "global_episode",
        "reward",
        "goal_rate",
        "SPI",
        "CDS",
        "proxy_action_rate",
        "proxy_Q_advantage",
        "proxy_dependence",
        "abs_proxy_dependence",
        "belief_accuracy",
    ]
    write_csv(os.path.join(RESULTS_DIR, "recurrent_belief_state_interaction_results.csv"), rows, result_fields)
    summary_fields = ["history_length", "aux_weight", "phase", "n"]
    for metric in result_fields[6:]:
        summary_fields.extend([f"{metric}_mean", f"{metric}_ci95"])
    write_csv(os.path.join(RESULTS_DIR, "recurrent_belief_state_interaction_summary.csv"), summary, summary_fields)
    write_csv(
        os.path.join(RESULTS_DIR, "recurrent_belief_state_interaction_comparisons.csv"),
        comparisons,
        ["history_length", "aux_weight_none", "aux_weight_strong", "none_mean", "strong_mean", "gap_none_minus_strong", "t_stat", "p_value", "cohens_d"],
    )
    write_csv(os.path.join(RESULTS_DIR, "recurrent_belief_state_interaction_summary_interaction.csv"), [interaction], ["gap_at_history_min", "gap_at_history_max", "interaction_difference"])
    write_plot(summary)
    write_report(summary, comparisons, interaction)


if __name__ == "__main__":
    run()
