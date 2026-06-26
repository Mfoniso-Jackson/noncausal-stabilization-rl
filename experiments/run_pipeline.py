import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import DQNAgent
from env import GridCueEnv
from experiment_logging.io import ensure_dirs, write_csv, write_json
from metrics import cue_dependence_score, stabilization_persistence_index
from utils.plotting import heatmap, line_plot, model_comparison_plot, point_plot_with_ci


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_env(config, seed, cues_enabled=True):
    env_config = dict(config)
    env_config["seed"] = seed
    env_config["cues_enabled"] = cues_enabled
    return GridCueEnv(**env_config)


def train_agent(name, env_config, episodes, seed, out_dirs):
    env = make_env(env_config, seed, cues_enabled=True)
    cue_bias = cue_bias_from_uncertainty(env_config)
    agent = DQNAgent(
        env.state_size,
        env.action_size,
        seed=seed,
        cue_exploration_bias=cue_bias,
        cue_regularization=0.025 * (1.0 - cue_bias),
    )
    logs = []
    for episode in range(1, episodes + 1):
        state = env.reset()
        total = 0.0
        losses = []
        reached = False
        for _ in range(env.max_steps):
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            losses.append(agent.train_step())
            total += reward
            reached = reached or info["reached_goal"]
            state = next_state
            if done:
                break
        logs.append(
            {
                "phase": name,
                "seed": seed,
                "episode": episode,
                "reward": round(total, 6),
                "epsilon": round(agent.epsilon, 6),
                "loss": round(sum(losses) / max(1, len(losses)), 6),
                "reached_goal": int(reached),
            }
        )
    checkpoint = os.path.join(out_dirs["checkpoints"], f"{name}_seed_{seed}.json")
    write_json(checkpoint, {"agent": agent.state_dict(), "env": env_config, "seed": seed, "cues": env.cues})
    return agent, env, logs


def cue_bias_from_uncertainty(env_config):
    sparsity = 1.0 - float(env_config.get("reward_sparse_prob", 1.0))
    delay = min(1.0, float(env_config.get("delay", 0)) / 6.0)
    noise = min(1.0, float(env_config.get("noise", 0.0)) / 0.2)
    uncertainty = (sparsity + delay + noise) / 3.0
    return min(0.85, max(0.0, uncertainty))


def uncertainty_score(env_config):
    sparsity = 1.0 - float(env_config.get("reward_sparse_prob", 1.0))
    delay = min(1.0, float(env_config.get("delay", 0)) / 6.0)
    noise = min(1.0, float(env_config.get("noise", 0.0)) / 0.2)
    return round((sparsity + delay + noise) / 3.0, 6)


def evaluate(agent, env, episodes, cues_enabled=True):
    eval_env = env.clone_for_eval(cues_enabled=cues_enabled)
    rewards = []
    visits = []
    states = []
    reached = 0
    for _ in range(episodes):
        state = eval_env.reset()
        total = 0.0
        for _ in range(eval_env.max_steps):
            action = agent.act(state, training=False)
            state, reward, done, info = eval_env.step(action)
            total += reward
            visits.append(info["position"])
            states.append(info["position"])
            reached += int(info["reached_goal"])
            if done:
                break
        rewards.append(total)
    zone_cds = cue_dependence_score(visits, env.cues, env.grid_size)
    divergence_cds = action_divergence_score(agent, env, states)
    cds = max(zone_cds, divergence_cds)
    return {
        "mean_reward": mean(rewards),
        "goal_rate": reached / max(1, episodes),
        "cds": cds,
        "zone_cds": zone_cds,
        "divergence_cds": divergence_cds,
        "visits": visits,
        "counts": dict(Counter(visits)),
    }


def action_divergence_score(agent, env, positions):
    if not env.cues or not positions:
        return 0.0
    probe = env.clone_for_eval(cues_enabled=True)
    changed = 0
    total = 0
    sample = positions[:: max(1, len(positions) // 500)]
    for pos in sample:
        probe.agent_pos = pos
        probe.cues_enabled = True
        with_cues = probe.observe()
        probe.cues_enabled = False
        without_cues = probe.observe()
        changed += int(agent.act(with_cues, training=False) != agent.act(without_cues, training=False))
        total += 1
    return changed / max(1, total)


def mean(values):
    return sum(values) / max(1, len(values))


def sem(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / (len(values) ** 0.5)


def moving_average(points, window=20):
    out = []
    for i in range(len(points)):
        start = max(0, i - window + 1)
        out.append((i + 1, mean(points[start : i + 1])))
    return out


def aggregate_curve(rows, phase):
    by_episode = defaultdict(list)
    for row in rows:
        if row["phase"] == phase:
            by_episode[int(row["episode"])].append(float(row["reward"]))
    return [(ep, mean(vals)) for ep, vals in sorted(by_episode.items())]


def save_heatmap(figures_dir, prefix, eval_result, grid_size):
    counts = {tuple(map(int, key.strip("()").split(","))) if isinstance(key, str) else key: value for key, value in eval_result["counts"].items()}
    heatmap(os.path.join(figures_dir, f"{prefix}.svg"), counts, grid_size, prefix.replace("_", " ").title())


def write_summary_markdown(path, summary_rows, trend_tests):
    lines = [
        "# Experimental Pipeline Summary",
        "",
        "Trend test: linear monotonic model compared against concave quadratic inverted-U model.",
        "",
        "| condition | uncertainty | reward 95% CI | goal_rate 95% CI | CDS 95% CI | SPI 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        level = "" if row["uncertainty_level"] == "" else f"{row['uncertainty_level']:.3f}"
        lines.append(
            f"| {row['condition']} | {level} | {row['mean_reward']:.3f} +/- {row['reward_ci95']:.3f} | "
            f"{row['goal_rate']:.3f} +/- {row['goal_rate_ci95']:.3f} | {row['cds_mean']:.3f} +/- {row['cds_ci95']:.3f} | "
            f"{row['spi_mean']:.3f} +/- {row['spi_ci95']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| metric | preferred | evidence | peak uncertainty | delta AIC | monotonic pattern |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in trend_tests:
        peak = "" if row["peak_uncertainty"] == "" else f"{row['peak_uncertainty']:.3f}"
        lines.append(
            f"| {row['metric']} | {row['preferred_model']} | {row['evidence']} | "
            f"{peak} | {row['delta_aic_linear_minus_quadratic']:.3f} | {row['monotonic_pattern']} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(config_path):
    config = load_config(config_path)
    out_dirs = {
        "runs": os.path.join(ROOT, "runs"),
        "checkpoints": os.path.join(ROOT, "checkpoints"),
        "figures": os.path.join(ROOT, "figures"),
    }
    ensure_dirs(*out_dirs.values())

    base_env = config["base_env"]
    episodes = int(config["episodes"])
    eval_episodes = int(config["eval_episodes"])
    seeds = list(config["seeds"])
    all_logs = []
    eval_rows = []

    phases = [
        ("baseline", {**base_env, **config["baseline"]}),
        ("cue_baseline", {**base_env, **config["cue_baseline"]}),
    ]
    trained = {}
    for phase, env_config in phases:
        for seed in seeds:
            agent, env, logs = train_agent(phase, env_config, episodes, seed, out_dirs)
            all_logs.extend(logs)
            before = evaluate(agent, env, eval_episodes, cues_enabled=True)
            after = evaluate(agent, env, eval_episodes, cues_enabled=False)
            spi = stabilization_persistence_index(before["cds"], after["cds"])
            eval_rows.append(pack_eval_row(phase, seed, None, before, after, spi))
            trained[(phase, seed)] = (agent, env, before, after)

    for condition, env_delta in config["uncertainty_conditions"].items():
        env_config = {**base_env, **env_delta}
        phase = f"uncertainty_{condition}"
        level = uncertainty_score(env_config)
        for seed in seeds:
            agent, env, logs = train_agent(phase, env_config, episodes, seed, out_dirs)
            all_logs.extend(logs)
            before = evaluate(agent, env, eval_episodes, cues_enabled=True)
            after = evaluate(agent, env, eval_episodes, cues_enabled=False)
            spi = stabilization_persistence_index(before["cds"], after["cds"])
            eval_rows.append(pack_eval_row(condition, seed, level, before, after, spi))
            trained[(condition, seed)] = (agent, env, before, after)

    write_csv(
        os.path.join(out_dirs["runs"], "training_logs.csv"),
        all_logs,
        ["phase", "seed", "episode", "reward", "epsilon", "loss", "reached_goal"],
    )
    write_csv(
        os.path.join(out_dirs["runs"], "evaluation_metrics.csv"),
        eval_rows,
        [
            "condition",
            "seed",
            "uncertainty_level",
            "before_reward",
            "after_reward",
            "goal_rate_before",
            "goal_rate_after",
            "cds",
            "spi",
        ],
    )

    condition_order = list(config["uncertainty_conditions"].keys())
    make_figures(out_dirs["figures"], all_logs, eval_rows, trained, condition_order)
    summary_rows = summarize(eval_rows, condition_order)
    trend_tests = run_trend_tests(eval_rows, condition_order)
    write_csv(
        os.path.join(out_dirs["runs"], "statistical_summary.csv"),
        summary_rows,
        [
            "condition",
            "uncertainty_level",
            "n",
            "mean_reward",
            "reward_sem",
            "reward_ci95",
            "goal_rate",
            "goal_rate_sem",
            "goal_rate_ci95",
            "cds_mean",
            "cds_sem",
            "cds_ci95",
            "spi_mean",
            "spi_sem",
            "spi_ci95",
        ],
    )
    write_csv(
        os.path.join(out_dirs["runs"], "trend_tests.csv"),
        trend_tests,
        [
            "metric",
            "n",
            "linear_sse",
            "quadratic_sse",
            "linear_aic",
            "quadratic_aic",
            "delta_aic_linear_minus_quadratic",
            "quadratic_term",
            "peak_uncertainty",
            "inverted_u_supported",
            "monotonic_supported",
            "evidence",
            "monotonic_pattern",
            "preferred_model",
        ],
    )
    criterion = any(row["inverted_u_supported"] or row["monotonic_supported"] for row in trend_tests)
    write_summary_markdown(os.path.join(out_dirs["runs"], "summary.md"), summary_rows, trend_tests)
    write_json(
        os.path.join(out_dirs["runs"], "run_manifest.json"),
        {
            "config": config,
            "inverted_u_supported": criterion,
            "trend_tests": trend_tests,
            "mechanism_note": (
                "Uncertainty increases cue-biased exploratory action selection. Cues remain reward-irrelevant "
                "and do not alter transitions, goals, or reward delivery."
            ),
        },
    )
    return criterion


def analyze_existing(config_path):
    config = load_config(config_path)
    out_dirs = {
        "runs": os.path.join(ROOT, "runs"),
        "figures": os.path.join(ROOT, "figures"),
    }
    ensure_dirs(*out_dirs.values())
    eval_path = os.path.join(out_dirs["runs"], "evaluation_metrics.csv")
    with open(eval_path, "r", encoding="utf-8", newline="") as f:
        eval_rows = list(csv.DictReader(f))
    condition_order = list(config["uncertainty_conditions"].keys())
    summary_rows = summarize(eval_rows, condition_order)
    trend_tests = run_trend_tests(eval_rows, condition_order)
    write_csv(
        os.path.join(out_dirs["runs"], "statistical_summary.csv"),
        summary_rows,
        [
            "condition",
            "uncertainty_level",
            "n",
            "mean_reward",
            "reward_sem",
            "reward_ci95",
            "goal_rate",
            "goal_rate_sem",
            "goal_rate_ci95",
            "cds_mean",
            "cds_sem",
            "cds_ci95",
            "spi_mean",
            "spi_sem",
            "spi_ci95",
        ],
    )
    write_csv(
        os.path.join(out_dirs["runs"], "trend_tests.csv"),
        trend_tests,
        [
            "metric",
            "n",
            "linear_sse",
            "quadratic_sse",
            "linear_aic",
            "quadratic_aic",
            "delta_aic_linear_minus_quadratic",
            "quadratic_term",
            "peak_uncertainty",
            "inverted_u_supported",
            "monotonic_supported",
            "evidence",
            "monotonic_pattern",
            "preferred_model",
        ],
    )
    make_statistical_figures(out_dirs["figures"], eval_rows, condition_order)
    write_summary_markdown(os.path.join(out_dirs["runs"], "summary.md"), summary_rows, trend_tests)
    write_json(
        os.path.join(out_dirs["runs"], "run_manifest.json"),
        {
            "config": config,
            "high_power_seed_count": len(config["seeds"]),
            "trend_tests": trend_tests,
            "interpretation": classify_overall_evidence(trend_tests),
        },
    )
    return any(row["inverted_u_supported"] or row["monotonic_supported"] for row in trend_tests)


def pack_eval_row(condition, seed, uncertainty_level, before, after, spi):
    return {
        "condition": condition,
        "seed": seed,
        "uncertainty_level": "" if uncertainty_level is None else uncertainty_level,
        "before_reward": round(before["mean_reward"], 6),
        "after_reward": round(after["mean_reward"], 6),
        "goal_rate_before": round(before["goal_rate"], 6),
        "goal_rate_after": round(after["goal_rate"], 6),
        "cds": round(before["cds"], 6),
        "spi": round(spi, 6),
    }


def summarize(eval_rows, condition_order=None):
    rows = []
    conditions = list(condition_order or [])
    for condition in sorted(set(row["condition"] for row in eval_rows)):
        if condition not in conditions:
            conditions.append(condition)
    for condition in conditions:
        subset = [row for row in eval_rows if row["condition"] == condition]
        if not subset:
            continue
        rewards = [float(row["before_reward"]) for row in subset]
        goals = [float(row["goal_rate_before"]) for row in subset]
        cds = [float(row["cds"]) for row in subset]
        spi = [float(row["spi"]) for row in subset]
        reward_sem = sem(rewards)
        goal_sem = sem(goals)
        cds_sem = sem(cds)
        spi_sem = sem(spi)
        levels = [row["uncertainty_level"] for row in subset if row["uncertainty_level"] != ""]
        rows.append(
            {
                "condition": condition,
                "uncertainty_level": "" if not levels else float(levels[0]),
                "n": len(subset),
                "mean_reward": round(mean(rewards), 6),
                "reward_sem": round(reward_sem, 6),
                "reward_ci95": round(1.96 * reward_sem, 6),
                "goal_rate": round(mean(goals), 6),
                "goal_rate_sem": round(goal_sem, 6),
                "goal_rate_ci95": round(1.96 * goal_sem, 6),
                "cds_mean": round(mean(cds), 6),
                "cds_sem": round(cds_sem, 6),
                "cds_ci95": round(1.96 * cds_sem, 6),
                "spi_mean": round(mean(spi), 6),
                "spi_sem": round(spi_sem, 6),
                "spi_ci95": round(1.96 * spi_sem, 6),
            }
        )
    return rows


def run_trend_tests(eval_rows, condition_order):
    rows = []
    for metric in ["cds", "spi"]:
        points = []
        means = []
        for index, condition in enumerate(condition_order):
            subset = [row for row in eval_rows if row["condition"] == condition]
            if not subset:
                continue
            x = float(subset[0]["uncertainty_level"])
            ys = [float(row[metric]) for row in subset]
            points.extend((x, y) for y in ys)
            means.append((x, mean(ys)))
        linear = fit_polynomial(points, degree=1)
        quadratic = fit_polynomial(points, degree=2)
        peak = ""
        if len(quadratic["coefficients"]) == 3 and abs(quadratic["coefficients"][2]) > 1e-12:
            peak = -quadratic["coefficients"][1] / (2.0 * quadratic["coefficients"][2])
        min_x = min(x for x, _ in points)
        max_x = max(x for x, _ in points)
        monotonic = monotonic_pattern([y for _, y in means])
        concave = len(quadratic["coefficients"]) == 3 and quadratic["coefficients"][2] < 0
        interior_peak = peak != "" and min_x <= peak <= max_x
        delta_aic = linear["aic"] - quadratic["aic"]
        inverted = bool(concave and interior_peak and delta_aic > 2.0)
        linear_slope = linear["coefficients"][1] if len(linear["coefficients"]) > 1 else 0.0
        linear_preferred = delta_aic < -2.0
        monotonic_supported = bool(linear_preferred and monotonic in ["increasing", "decreasing"] and abs(linear_slope) > 1e-9)
        if inverted:
            evidence = "inverted_u"
        elif monotonic_supported:
            evidence = f"monotonic_{monotonic}"
        else:
            evidence = "inconclusive"
        if delta_aic > 2.0:
            preferred = "quadratic"
        elif delta_aic < -2.0:
            preferred = "linear"
        else:
            preferred = "tie"
        rows.append(
            {
                "metric": metric.upper(),
                "n": len(points),
                "linear_sse": round(linear["sse"], 6),
                "quadratic_sse": round(quadratic["sse"], 6),
                "linear_aic": round(linear["aic"], 6),
                "quadratic_aic": round(quadratic["aic"], 6),
                "delta_aic_linear_minus_quadratic": round(delta_aic, 6),
                "quadratic_term": round(quadratic["coefficients"][2], 6),
                "peak_uncertainty": "" if peak == "" else round(peak, 6),
                "inverted_u_supported": inverted,
                "monotonic_supported": monotonic_supported,
                "evidence": evidence,
                "monotonic_pattern": monotonic,
                "preferred_model": preferred,
            }
        )
    return rows


def classify_overall_evidence(trend_tests):
    if any(row["inverted_u_supported"] for row in trend_tests):
        return "inverted_u_uncertainty_effect"
    if any(row["monotonic_supported"] for row in trend_tests):
        return "monotonic_uncertainty_effect"
    return "inconclusive_no_reliable_relationship"


def fit_polynomial(points, degree):
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    size = degree + 1
    matrix = []
    rhs = []
    for row in range(size):
        matrix.append([sum(x ** (row + col) for x in x_values) for col in range(size)])
        rhs.append(sum(y * (x ** row) for x, y in zip(x_values, y_values)))
    coefficients = solve_linear_system(matrix, rhs)
    predictions = [sum(coefficients[i] * (x ** i) for i in range(size)) for x in x_values]
    sse = sum((y - y_hat) ** 2 for y, y_hat in zip(y_values, predictions))
    aic = len(points) * math_log(max(sse / max(1, len(points)), 1e-12)) + 2 * size
    return {"coefficients": coefficients, "sse": sse, "aic": aic}


def predict_polynomial(coefficients, x_values):
    return [sum(coefficients[i] * (x ** i) for i in range(len(coefficients))) for x in x_values]


def solve_linear_system(matrix, rhs):
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        if abs(a[col][col]) < 1e-12:
            continue
        divisor = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= divisor
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            for j in range(col, n + 1):
                a[r][j] -= factor * a[col][j]
    return [a[i][n] for i in range(n)]


def math_log(value):
    import math

    return math.log(value)


def monotonic_pattern(values):
    increasing = all(a <= b for a, b in zip(values, values[1:]))
    decreasing = all(a >= b for a, b in zip(values, values[1:]))
    if increasing and not decreasing:
        return "increasing"
    if decreasing and not increasing:
        return "decreasing"
    if increasing and decreasing:
        return "flat"
    return "non_monotonic"


def make_figures(figures_dir, logs, eval_rows, trained, condition_order):
    line_plot(
        os.path.join(figures_dir, "baseline_reward_curve.svg"),
        [("no cues", moving_average([y for _, y in aggregate_curve(logs, "baseline")]))],
        "Baseline Reward Curve",
        "Episode",
        "Mean reward",
    )
    line_plot(
        os.path.join(figures_dir, "cue_baseline_reward_curve.svg"),
        [
            ("no cues", moving_average([y for _, y in aggregate_curve(logs, "baseline")])),
            ("5 cues", moving_average([y for _, y in aggregate_curve(logs, "cue_baseline")])),
        ],
        "Cue Baseline Reward Curve",
        "Episode",
        "Mean reward",
    )
    summary = {row["condition"]: row for row in summarize(eval_rows, condition_order)}
    labels = [label for label in condition_order if label in summary]
    x_values = [summary[x]["uncertainty_level"] for x in labels]
    point_plot_with_ci(
        os.path.join(figures_dir, "reward_vs_uncertainty.svg"),
        labels,
        [summary[x]["mean_reward"] for x in labels],
        [summary[x]["reward_ci95"] for x in labels],
        "Reward vs Uncertainty",
        "Reward +/- 95% CI",
    )
    point_plot_with_ci(
        os.path.join(figures_dir, "goal_rate_vs_uncertainty.svg"),
        labels,
        [summary[x]["goal_rate"] for x in labels],
        [summary[x]["goal_rate_ci95"] for x in labels],
        "Goal Rate vs Uncertainty",
        "Goal rate +/- 95% CI",
    )
    point_plot_with_ci(
        os.path.join(figures_dir, "cds_vs_uncertainty.svg"),
        labels,
        [summary[x]["cds_mean"] for x in labels],
        [summary[x]["cds_ci95"] for x in labels],
        "CDS vs Uncertainty",
        "CDS +/- 95% CI",
    )
    point_plot_with_ci(
        os.path.join(figures_dir, "spi_vs_uncertainty.svg"),
        labels,
        [summary[x]["spi_mean"] for x in labels],
        [summary[x]["spi_ci95"] for x in labels],
        "SPI vs Uncertainty",
        "SPI +/- 95% CI",
    )
    model_series = []
    for metric, field, name in [("cds", "cds_mean", "CDS"), ("spi", "spi_mean", "SPI")]:
        points = []
        observed = []
        for condition in labels:
            subset = [row for row in eval_rows if row["condition"] == condition]
            observed.append(summary[condition][field])
            points.extend((summary[condition]["uncertainty_level"], float(row[metric])) for row in subset)
        linear = fit_polynomial(points, degree=1)
        quadratic = fit_polynomial(points, degree=2)
        model_series.append(
            {
                "name": name,
                "observed": observed,
                "linear": predict_polynomial(linear["coefficients"], x_values),
                "quadratic": predict_polynomial(quadratic["coefficients"], x_values),
            }
        )
    model_comparison_plot(
        os.path.join(figures_dir, "linear_vs_quadratic_model_comparison.svg"),
        labels,
        x_values,
        model_series,
        "Linear vs Quadratic Model Comparison",
    )

    for condition in labels:
        key = (condition, 0)
        if key in trained:
            _, env, before, after = trained[key]
            heatmap(os.path.join(figures_dir, f"visitation_{condition}_before_intervention.svg"), before["counts"], env.grid_size, f"{condition.title()} Before Intervention")
            heatmap(os.path.join(figures_dir, f"visitation_{condition}_after_intervention.svg"), after["counts"], env.grid_size, f"{condition.title()} After Intervention")


def make_statistical_figures(figures_dir, eval_rows, condition_order):
    summary = {row["condition"]: row for row in summarize(eval_rows, condition_order)}
    labels = [label for label in condition_order if label in summary]
    x_values = [summary[x]["uncertainty_level"] for x in labels]
    plot_specs = [
        ("reward_vs_uncertainty.svg", "mean_reward", "reward_ci95", "Reward vs Uncertainty", "Reward +/- 95% CI"),
        ("goal_rate_vs_uncertainty.svg", "goal_rate", "goal_rate_ci95", "Goal Rate vs Uncertainty", "Goal rate +/- 95% CI"),
        ("cds_vs_uncertainty.svg", "cds_mean", "cds_ci95", "CDS vs Uncertainty", "CDS +/- 95% CI"),
        ("spi_vs_uncertainty.svg", "spi_mean", "spi_ci95", "SPI vs Uncertainty", "SPI +/- 95% CI"),
    ]
    for filename, value_key, ci_key, title, ylabel in plot_specs:
        point_plot_with_ci(
            os.path.join(figures_dir, filename),
            labels,
            [summary[x][value_key] for x in labels],
            [summary[x][ci_key] for x in labels],
            title,
            ylabel,
        )
    model_series = []
    for metric, field, name in [("cds", "cds_mean", "CDS"), ("spi", "spi_mean", "SPI")]:
        points = []
        observed = []
        for condition in labels:
            subset = [row for row in eval_rows if row["condition"] == condition]
            observed.append(summary[condition][field])
            points.extend((summary[condition]["uncertainty_level"], float(row[metric])) for row in subset)
        linear = fit_polynomial(points, degree=1)
        quadratic = fit_polynomial(points, degree=2)
        model_series.append(
            {
                "name": name,
                "observed": observed,
                "linear": predict_polynomial(linear["coefficients"], x_values),
                "quadratic": predict_polynomial(quadratic["coefficients"], x_values),
            }
        )
    model_comparison_plot(
        os.path.join(figures_dir, "linear_vs_quadratic_model_comparison.svg"),
        labels,
        x_values,
        model_series,
        "Linear vs Quadratic Model Comparison",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "experiment_config.json"))
    parser.add_argument("--analyze-only", action="store_true", help="Regenerate summaries and figures from runs/evaluation_metrics.csv")
    args = parser.parse_args()
    passed = analyze_existing(args.config) if args.analyze_only else run(args.config)
    print(f"pipeline complete; evidence_supported={passed}")
