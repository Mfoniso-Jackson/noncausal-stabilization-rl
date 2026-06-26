import argparse
import csv
import hashlib
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import DQNAgent
from experiment_logging.io import ensure_dirs, write_csv, write_json
from experiments.run_pipeline import (
    ROOT,
    cue_bias_from_uncertainty,
    evaluate,
    fit_polynomial,
    load_config,
    make_env,
    mean,
    pack_eval_row,
    predict_polynomial,
    run_trend_tests,
    sem,
    stabilization_persistence_index,
    summarize,
    train_agent,
    uncertainty_score,
)
from utils.plotting import model_comparison_plot, point_plot_with_ci


SEEDS = list(range(30))
EPISODES = 300
EVAL_EPISODES = 30


def mechanism_specs():
    base = {"grid_size": 10, "max_steps": 200, "num_cues": 5}
    return {
        "sparsity": [
            condition("sparsity_1.0", 0.0, "reward_sparse_prob=1.0", {**base, "reward_sparse_prob": 1.0, "delay": 0, "noise": 0.0}),
            condition("sparsity_0.7", 0.3, "reward_sparse_prob=0.7", {**base, "reward_sparse_prob": 0.7, "delay": 0, "noise": 0.0}),
            condition("sparsity_0.5", 0.5, "reward_sparse_prob=0.5", {**base, "reward_sparse_prob": 0.5, "delay": 0, "noise": 0.0}),
            condition("sparsity_0.3", 0.7, "reward_sparse_prob=0.3", {**base, "reward_sparse_prob": 0.3, "delay": 0, "noise": 0.0}),
            condition("sparsity_0.1", 0.9, "reward_sparse_prob=0.1", {**base, "reward_sparse_prob": 0.1, "delay": 0, "noise": 0.0}),
        ],
        "delay": [
            condition("delay_0", 0.0, "delay=0", {**base, "reward_sparse_prob": 1.0, "delay": 0, "noise": 0.0}),
            condition("delay_1", 1.0, "delay=1", {**base, "reward_sparse_prob": 1.0, "delay": 1, "noise": 0.0}),
            condition("delay_3", 3.0, "delay=3", {**base, "reward_sparse_prob": 1.0, "delay": 3, "noise": 0.0}),
            condition("delay_6", 6.0, "delay=6", {**base, "reward_sparse_prob": 1.0, "delay": 6, "noise": 0.0}),
            condition("delay_10", 10.0, "delay=10", {**base, "reward_sparse_prob": 1.0, "delay": 10, "noise": 0.0}),
        ],
        "noise": [
            condition("noise_0.0", 0.0, "noise=0.0", {**base, "reward_sparse_prob": 1.0, "delay": 0, "noise": 0.0}),
            condition("noise_0.05", 0.05, "noise=0.05", {**base, "reward_sparse_prob": 1.0, "delay": 0, "noise": 0.05}),
            condition("noise_0.1", 0.1, "noise=0.1", {**base, "reward_sparse_prob": 1.0, "delay": 0, "noise": 0.1}),
            condition("noise_0.2", 0.2, "noise=0.2", {**base, "reward_sparse_prob": 1.0, "delay": 0, "noise": 0.2}),
            condition("noise_0.3", 0.3, "noise=0.3", {**base, "reward_sparse_prob": 1.0, "delay": 0, "noise": 0.3}),
        ],
        "combined": [
            condition("very_low", 0.0, "very_low", {**base, "reward_sparse_prob": 1.0, "delay": 0, "noise": 0.0}),
            condition("low", 0.255556, "low", {**base, "reward_sparse_prob": 0.65, "delay": 1, "noise": 0.05}),
            condition("medium", 0.566667, "medium", {**base, "reward_sparse_prob": 0.3, "delay": 3, "noise": 0.1}),
            condition("high", 0.817778, "high", {**base, "reward_sparse_prob": 0.18, "delay": 5, "noise": 0.16}),
            condition("very_high", 0.966667, "very_high", {**base, "reward_sparse_prob": 0.1, "delay": 6, "noise": 0.2}),
        ],
    }


def condition(name, x_value, label, env):
    return {"name": name, "x_value": x_value, "label": label, "env": env}


def run_mechanism_isolation():
    out_dirs = {
        "runs": os.path.join(ROOT, "runs"),
        "checkpoints": os.path.join(ROOT, "checkpoints"),
        "figures": os.path.join(ROOT, "figures"),
        "eval_cache": os.path.join(ROOT, "runs", "mechanism_eval_cache"),
    }
    ensure_dirs(*out_dirs.values())
    specs = mechanism_specs()
    all_logs = []
    eval_rows = []
    cache = {}
    total_jobs = sum(len(items) * len(SEEDS) for items in specs.values())
    completed = 0

    for group, items in specs.items():
        for item in items:
            for seed in SEEDS:
                completed += 1
                phase = f"mechanism_{group}_{item['name']}"
                print(f"[{completed}/{total_jobs}] {group}/{item['name']} seed={seed}", flush=True)
                cached = load_cached_eval(item["env"], seed, out_dirs["eval_cache"])
                if cached:
                    eval_rows.append(row_from_cached_eval(group, item, seed, cached))
                    continue
                agent, env, logs = train_or_reuse(phase, item["env"], seed, out_dirs, cache)
                all_logs.extend(logs)
                before = evaluate(agent, env, EVAL_EPISODES, cues_enabled=True)
                after = evaluate(agent, env, EVAL_EPISODES, cues_enabled=False)
                spi = stabilization_persistence_index(before["cds"], after["cds"])
                save_cached_eval(item["env"], seed, before, after, spi, out_dirs["eval_cache"])
                eval_rows.append(
                    {
                        **pack_eval_row(item["name"], seed, item["x_value"], before, after, spi),
                        "group": group,
                        "x_value": item["x_value"],
                        "x_label": item["label"],
                    }
                )

    write_outputs(out_dirs, specs, all_logs, eval_rows)
    return eval_rows


def train_or_reuse(phase, env_config, seed, out_dirs, cache):
    key = config_key(env_config, seed)
    if key in cache:
        agent, env = cache[key]
        return agent, env, []
    loaded = load_existing_checkpoint(env_config, seed, phase)
    if loaded:
        cache[key] = loaded
        return loaded[0], loaded[1], []
    agent, env, logs = train_agent(phase, env_config, EPISODES, seed, out_dirs)
    cache[key] = (agent, env)
    return agent, env, logs


def config_key(env_config, seed):
    return (
        int(env_config["grid_size"]),
        int(env_config["num_cues"]),
        float(env_config["reward_sparse_prob"]),
        int(env_config["delay"]),
        float(env_config["noise"]),
        int(env_config["max_steps"]),
        int(seed),
    )


def load_existing_checkpoint(env_config, seed, phase=None):
    candidates = checkpoint_candidates(env_config, seed, phase)
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved_env = data["env"]
            if not same_env(saved_env, env_config):
                continue
            env = make_env(saved_env, seed, cues_enabled=True)
            env.cues = [tuple(cue) for cue in data.get("cues", env.cues)]
            cue_bias = cue_bias_from_uncertainty(saved_env)
            agent = DQNAgent(
                env.state_size,
                env.action_size,
                seed=seed,
                cue_exploration_bias=cue_bias,
                cue_regularization=0.025 * (1.0 - cue_bias),
            )
            agent.load_state_dict(data["agent"])
            return agent, env
    return None


def checkpoint_candidates(env_config, seed, phase=None):
    checkpoints = os.path.join(ROOT, "checkpoints")
    names = []
    if phase:
        names.append(f"{phase}_seed_{seed}.json")
    if same_env(env_config, {"grid_size": 10, "max_steps": 200, "num_cues": 5, "reward_sparse_prob": 1.0, "delay": 0, "noise": 0.0}):
        names.extend([f"cue_baseline_seed_{seed}.json", f"uncertainty_very_low_seed_{seed}.json"])
    combined_names = {
        (0.65, 1, 0.05): "uncertainty_low",
        (0.3, 3, 0.1): "uncertainty_medium",
        (0.18, 5, 0.16): "uncertainty_high",
        (0.1, 6, 0.2): "uncertainty_very_high",
    }
    key = (float(env_config["reward_sparse_prob"]), int(env_config["delay"]), float(env_config["noise"]))
    if key in combined_names:
        names.append(f"{combined_names[key]}_seed_{seed}.json")
    return [os.path.join(checkpoints, name) for name in names]


def eval_cache_key(env_config, seed):
    payload = {
        "env": {
            "grid_size": int(env_config["grid_size"]),
            "num_cues": int(env_config["num_cues"]),
            "reward_sparse_prob": float(env_config["reward_sparse_prob"]),
            "delay": int(env_config["delay"]),
            "noise": float(env_config["noise"]),
            "max_steps": int(env_config["max_steps"]),
        },
        "seed": int(seed),
        "eval_episodes": EVAL_EPISODES,
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def eval_cache_path(env_config, seed, cache_dir):
    return os.path.join(cache_dir, f"{eval_cache_key(env_config, seed)}.json")


def load_cached_eval(env_config, seed, cache_dir):
    path = eval_cache_path(env_config, seed, cache_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cached_eval(env_config, seed, before, after, spi, cache_dir):
    path = eval_cache_path(env_config, seed, cache_dir)
    write_json(
        path,
        {
            "before_reward": round(before["mean_reward"], 6),
            "after_reward": round(after["mean_reward"], 6),
            "goal_rate_before": round(before["goal_rate"], 6),
            "goal_rate_after": round(after["goal_rate"], 6),
            "cds": round(before["cds"], 6),
            "spi": round(spi, 6),
        },
    )


def row_from_cached_eval(group, item, seed, cached):
    return {
        "group": group,
        "condition": item["name"],
        "seed": seed,
        "x_value": item["x_value"],
        "x_label": item["label"],
        "uncertainty_level": item["x_value"],
        "before_reward": cached["before_reward"],
        "after_reward": cached["after_reward"],
        "goal_rate_before": cached["goal_rate_before"],
        "goal_rate_after": cached["goal_rate_after"],
        "cds": cached["cds"],
        "spi": cached["spi"],
    }


def same_env(a, b):
    return (
        int(a["grid_size"]) == int(b["grid_size"])
        and int(a["max_steps"]) == int(b["max_steps"])
        and int(a["num_cues"]) == int(b["num_cues"])
        and abs(float(a["reward_sparse_prob"]) - float(b["reward_sparse_prob"])) < 1e-9
        and int(a["delay"]) == int(b["delay"])
        and abs(float(a["noise"]) - float(b["noise"])) < 1e-9
    )


def write_outputs(out_dirs, specs, all_logs, eval_rows):
    write_csv(
        os.path.join(out_dirs["runs"], "mechanism_training_logs.csv"),
        all_logs,
        ["phase", "seed", "episode", "reward", "epsilon", "loss", "reached_goal"],
    )
    write_csv(
        os.path.join(out_dirs["runs"], "mechanism_evaluation_metrics.csv"),
        eval_rows,
        [
            "group",
            "condition",
            "seed",
            "x_value",
            "x_label",
            "uncertainty_level",
            "before_reward",
            "after_reward",
            "goal_rate_before",
            "goal_rate_after",
            "cds",
            "spi",
        ],
    )
    summary_rows = mechanism_summary(eval_rows, specs)
    write_csv(
        os.path.join(out_dirs["runs"], "mechanism_summary.csv"),
        summary_rows,
        [
            "group",
            "condition",
            "x_value",
            "x_label",
            "n",
            "mean_reward",
            "reward_ci95",
            "goal_rate",
            "goal_rate_ci95",
            "cds_mean",
            "cds_ci95",
            "spi_mean",
            "spi_ci95",
        ],
    )
    mechanism_rows = mechanism_strength_summary(summary_rows)
    write_csv(
        os.path.join(out_dirs["runs"], "mechanism_strength_summary.csv"),
        mechanism_rows,
        [
            "group",
            "best_condition",
            "best_x_value",
            "best_goal_rate",
            "best_cds_mean",
            "best_spi_mean",
            "cds_gain_from_control",
            "spi_gain_from_control",
            "stabilization_score",
            "goal_rate_preserved",
        ],
    )
    trends = mechanism_trends(eval_rows, specs)
    write_csv(
        os.path.join(out_dirs["runs"], "mechanism_trend_tests.csv"),
        trends,
        [
            "group",
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
    make_mechanism_figures(out_dirs["figures"], summary_rows, mechanism_rows, specs)
    write_mechanism_markdown(os.path.join(out_dirs["runs"], "mechanism_summary.md"), summary_rows, mechanism_rows, trends)
    write_json(
        os.path.join(out_dirs["runs"], "mechanism_manifest.json"),
        {
            "seeds": SEEDS,
            "episodes": EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "groups": specs,
            "mechanism_strength_summary": mechanism_rows,
        },
    )


def mechanism_summary(eval_rows, specs):
    rows = []
    order = [(group, item) for group, items in specs.items() for item in items]
    for group, item in order:
        subset = [row for row in eval_rows if row["group"] == group and row["condition"] == item["name"]]
        rewards = [float(row["before_reward"]) for row in subset]
        goals = [float(row["goal_rate_before"]) for row in subset]
        cds = [float(row["cds"]) for row in subset]
        spi = [float(row["spi"]) for row in subset]
        rows.append(
            {
                "group": group,
                "condition": item["name"],
                "x_value": item["x_value"],
                "x_label": item["label"],
                "n": len(subset),
                "mean_reward": round(mean(rewards), 6),
                "reward_ci95": round(1.96 * sem(rewards), 6),
                "goal_rate": round(mean(goals), 6),
                "goal_rate_ci95": round(1.96 * sem(goals), 6),
                "cds_mean": round(mean(cds), 6),
                "cds_ci95": round(1.96 * sem(cds), 6),
                "spi_mean": round(mean(spi), 6),
                "spi_ci95": round(1.96 * sem(spi), 6),
            }
        )
    return rows


def mechanism_strength_summary(summary_rows, min_goal_rate=0.75):
    rows = []
    for group in ["sparsity", "delay", "noise", "combined"]:
        subset = [row for row in summary_rows if row["group"] == group]
        control = subset[0]
        viable = [row for row in subset if row["goal_rate"] >= min_goal_rate]
        if not viable:
            viable = subset
        best = max(viable, key=lambda row: (row["cds_mean"] - control["cds_mean"]) + (row["spi_mean"] - control["spi_mean"]))
        cds_gain = best["cds_mean"] - control["cds_mean"]
        spi_gain = best["spi_mean"] - control["spi_mean"]
        rows.append(
            {
                "group": group,
                "best_condition": best["condition"],
                "best_x_value": best["x_value"],
                "best_goal_rate": best["goal_rate"],
                "best_cds_mean": best["cds_mean"],
                "best_spi_mean": best["spi_mean"],
                "cds_gain_from_control": round(cds_gain, 6),
                "spi_gain_from_control": round(spi_gain, 6),
                "stabilization_score": round(cds_gain + spi_gain, 6),
                "goal_rate_preserved": best["goal_rate"] >= min_goal_rate,
            }
        )
    rows.sort(key=lambda row: row["stabilization_score"], reverse=True)
    return rows


def mechanism_trends(eval_rows, specs):
    rows = []
    for group, items in specs.items():
        group_rows = []
        for row in eval_rows:
            if row["group"] == group:
                group_rows.append({**row, "uncertainty_level": row["x_value"]})
        condition_order = [item["name"] for item in items]
        for trend in run_trend_tests(group_rows, condition_order):
            rows.append({"group": group, **trend})
    return rows


def make_mechanism_figures(figures_dir, summary_rows, mechanism_rows, specs):
    labels_by_group = {
        "sparsity": "Reward sparsity",
        "delay": "Reward delay",
        "noise": "Transition noise",
        "combined": "Combined uncertainty",
    }
    for group, items in specs.items():
        subset = [row for row in summary_rows if row["group"] == group]
        labels = [row["x_label"].split("=")[-1] for row in subset]
        xlabel = labels_by_group[group]
        for metric, field, ci, title_name in [
            ("cds", "cds_mean", "cds_ci95", "CDS"),
            ("spi", "spi_mean", "spi_ci95", "SPI"),
        ]:
            point_plot_with_ci(
                os.path.join(figures_dir, f"{metric}_by_{group}.svg"),
                labels,
                [row[field] for row in subset],
                [row[ci] for row in subset],
                f"{title_name} by {group.title()}",
                f"{title_name} +/- 95% CI",
                xlabel=xlabel,
            )
    point_plot_with_ci(
        os.path.join(figures_dir, "mechanism_comparison_summary.svg"),
        [row["group"] for row in mechanism_rows],
        [row["stabilization_score"] for row in mechanism_rows],
        [0.0 for _ in mechanism_rows],
        "Mechanism Comparison Summary",
        "CDS gain + SPI gain",
        xlabel="Mechanism",
    )
    make_combined_model_plot(figures_dir, summary_rows, specs)


def make_combined_model_plot(figures_dir, summary_rows, specs):
    labels = []
    x_values = []
    series = []
    for group in ["sparsity", "delay", "noise", "combined"]:
        subset = [row for row in summary_rows if row["group"] == group]
        labels = [row["x_label"].split("=")[-1] for row in subset]
        x_values = [float(row["x_value"]) for row in subset]
        for metric, field, name in [("cds", "cds_mean", "CDS"), ("spi", "spi_mean", "SPI")]:
            points = []
            for row in subset:
                points.extend([(float(row["x_value"]), row[field])] * int(row["n"]))
            linear = fit_polynomial(points, degree=1)
            quadratic = fit_polynomial(points, degree=2)
            series.append(
                {
                    "name": f"{group} {name}",
                    "observed": [row[field] for row in subset],
                    "linear": predict_polynomial(linear["coefficients"], x_values),
                    "quadratic": predict_polynomial(quadratic["coefficients"], x_values),
                }
            )
    # Keep this plot readable by showing the combined mechanism models only.
    combined = [row for row in summary_rows if row["group"] == "combined"]
    labels = [row["x_label"] for row in combined]
    x_values = [float(row["x_value"]) for row in combined]
    combined_series = []
    for metric, field, name in [("cds", "cds_mean", "CDS"), ("spi", "spi_mean", "SPI")]:
        points = []
        for row in combined:
            points.extend([(float(row["x_value"]), row[field])] * int(row["n"]))
        linear = fit_polynomial(points, degree=1)
        quadratic = fit_polynomial(points, degree=2)
        combined_series.append(
            {
                "name": name,
                "observed": [row[field] for row in combined],
                "linear": predict_polynomial(linear["coefficients"], x_values),
                "quadratic": predict_polynomial(quadratic["coefficients"], x_values),
            }
        )
    model_comparison_plot(
        os.path.join(figures_dir, "mechanism_combined_linear_vs_quadratic.svg"),
        labels,
        x_values,
        combined_series,
        "Combined Mechanism Linear vs Quadratic",
    )


def write_mechanism_markdown(path, summary_rows, mechanism_rows, trends):
    winner = mechanism_rows[0]
    lines = [
        "# Mechanism Isolation Summary",
        "",
        f"Best mechanism preserving goal_rate: {winner['group']} ({winner['best_condition']}).",
        "",
        "| group | condition | x | reward 95% CI | goal_rate 95% CI | CDS 95% CI | SPI 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['group']} | {row['condition']} | {row['x_value']:.3f} | "
            f"{row['mean_reward']:.3f} +/- {row['reward_ci95']:.3f} | "
            f"{row['goal_rate']:.3f} +/- {row['goal_rate_ci95']:.3f} | "
            f"{row['cds_mean']:.3f} +/- {row['cds_ci95']:.3f} | "
            f"{row['spi_mean']:.3f} +/- {row['spi_ci95']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| group | best condition | goal_rate | CDS gain | SPI gain | score | goal preserved |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mechanism_rows:
        lines.append(
            f"| {row['group']} | {row['best_condition']} | {row['best_goal_rate']:.3f} | "
            f"{row['cds_gain_from_control']:.3f} | {row['spi_gain_from_control']:.3f} | "
            f"{row['stabilization_score']:.3f} | {row['goal_rate_preserved']} |"
        )
    lines.extend(["", "| group | metric | preferred | evidence | delta AIC | peak |", "|---|---|---|---|---:|---:|"])
    for row in trends:
        peak = "" if row["peak_uncertainty"] == "" else f"{row['peak_uncertainty']:.3f}"
        lines.append(
            f"| {row['group']} | {row['metric']} | {row['preferred_model']} | {row['evidence']} | "
            f"{row['delta_aic_linear_minus_quadratic']:.3f} | {peak} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def analyze_existing():
    specs = mechanism_specs()
    runs = os.path.join(ROOT, "runs")
    figures = os.path.join(ROOT, "figures")
    with open(os.path.join(runs, "mechanism_evaluation_metrics.csv"), "r", encoding="utf-8", newline="") as f:
        eval_rows = list(csv.DictReader(f))
    summary_rows = mechanism_summary(eval_rows, specs)
    mechanism_rows = mechanism_strength_summary(summary_rows)
    trends = mechanism_trends(eval_rows, specs)
    write_csv(
        os.path.join(runs, "mechanism_summary.csv"),
        summary_rows,
        [
            "group",
            "condition",
            "x_value",
            "x_label",
            "n",
            "mean_reward",
            "reward_ci95",
            "goal_rate",
            "goal_rate_ci95",
            "cds_mean",
            "cds_ci95",
            "spi_mean",
            "spi_ci95",
        ],
    )
    write_csv(
        os.path.join(runs, "mechanism_strength_summary.csv"),
        mechanism_rows,
        [
            "group",
            "best_condition",
            "best_x_value",
            "best_goal_rate",
            "best_cds_mean",
            "best_spi_mean",
            "cds_gain_from_control",
            "spi_gain_from_control",
            "stabilization_score",
            "goal_rate_preserved",
        ],
    )
    write_csv(
        os.path.join(runs, "mechanism_trend_tests.csv"),
        trends,
        [
            "group",
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
    make_mechanism_figures(figures, summary_rows, mechanism_rows, specs)
    write_mechanism_markdown(os.path.join(runs, "mechanism_summary.md"), summary_rows, mechanism_rows, trends)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        analyze_existing()
        print("mechanism analysis regenerated")
    else:
        run_mechanism_isolation()
        print("mechanism isolation complete")
