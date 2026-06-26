import argparse
import csv
import json
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_logging.io import ensure_dirs, write_csv, write_json
from experiments.run_mechanism_isolation import (
    EVAL_EPISODES,
    EPISODES,
    SEEDS,
    load_cached_eval,
    row_from_cached_eval,
    save_cached_eval,
    train_or_reuse,
)
from experiments.run_pipeline import ROOT, evaluate, mean, sem, stabilization_persistence_index


BASE_ENV = {"grid_size": 10, "max_steps": 200, "num_cues": 5, "noise": 0.0}


def factorial_conditions():
    return [
        {
            "condition": "low_delay_low_sparsity",
            "delay_level": "low",
            "sparsity_level": "low",
            "delay_high": 0,
            "sparsity_high": 0,
            "env": {**BASE_ENV, "delay": 0, "reward_sparse_prob": 1.0},
        },
        {
            "condition": "high_delay_low_sparsity",
            "delay_level": "high",
            "sparsity_level": "low",
            "delay_high": 1,
            "sparsity_high": 0,
            "env": {**BASE_ENV, "delay": 6, "reward_sparse_prob": 1.0},
        },
        {
            "condition": "low_delay_high_sparsity",
            "delay_level": "low",
            "sparsity_level": "high",
            "delay_high": 0,
            "sparsity_high": 1,
            "env": {**BASE_ENV, "delay": 0, "reward_sparse_prob": 0.1},
        },
        {
            "condition": "high_delay_high_sparsity",
            "delay_level": "high",
            "sparsity_level": "high",
            "delay_high": 1,
            "sparsity_high": 1,
            "env": {**BASE_ENV, "delay": 6, "reward_sparse_prob": 0.1},
        },
    ]


def run_factorial():
    out_dirs = {
        "runs": os.path.join(ROOT, "runs"),
        "checkpoints": os.path.join(ROOT, "checkpoints"),
        "figures": os.path.join(ROOT, "figures"),
        "eval_cache": os.path.join(ROOT, "runs", "mechanism_eval_cache"),
    }
    ensure_dirs(*out_dirs.values())
    rows = []
    logs = []
    cache = {}
    conditions = factorial_conditions()
    total = len(conditions) * len(SEEDS)
    completed = 0
    for condition in conditions:
        for seed in SEEDS:
            completed += 1
            print(f"[{completed}/{total}] {condition['condition']} seed={seed}", flush=True)
            cached = load_cached_eval(condition["env"], seed, out_dirs["eval_cache"])
            if cached:
                rows.append(row_from_factorial_cache(condition, seed, cached))
                continue
            phase = f"factorial_{condition['condition']}"
            agent, env, new_logs = train_or_reuse(phase, condition["env"], seed, out_dirs, cache)
            logs.extend(new_logs)
            before = evaluate(agent, env, EVAL_EPISODES, cues_enabled=True)
            after = evaluate(agent, env, EVAL_EPISODES, cues_enabled=False)
            spi = stabilization_persistence_index(before["cds"], after["cds"])
            save_cached_eval(condition["env"], seed, before, after, spi, out_dirs["eval_cache"])
            rows.append(row_from_eval(condition, seed, before, after, spi))

    write_factorial_outputs(out_dirs, rows, logs)
    return rows


def row_from_factorial_cache(condition, seed, cached):
    row = row_base(condition, seed)
    row.update(
        {
            "before_reward": cached["before_reward"],
            "after_reward": cached["after_reward"],
            "goal_rate_before": cached["goal_rate_before"],
            "goal_rate_after": cached["goal_rate_after"],
            "cds": cached["cds"],
            "spi": cached["spi"],
        }
    )
    return row


def row_from_eval(condition, seed, before, after, spi):
    row = row_base(condition, seed)
    row.update(
        {
            "before_reward": round(before["mean_reward"], 6),
            "after_reward": round(after["mean_reward"], 6),
            "goal_rate_before": round(before["goal_rate"], 6),
            "goal_rate_after": round(after["goal_rate"], 6),
            "cds": round(before["cds"], 6),
            "spi": round(spi, 6),
        }
    )
    return row


def row_base(condition, seed):
    return {
        "condition": condition["condition"],
        "seed": seed,
        "delay_level": condition["delay_level"],
        "sparsity_level": condition["sparsity_level"],
        "delay_high": condition["delay_high"],
        "sparsity_high": condition["sparsity_high"],
        "delay": condition["env"]["delay"],
        "reward_sparse_prob": condition["env"]["reward_sparse_prob"],
        "noise": condition["env"]["noise"],
    }


def write_factorial_outputs(out_dirs, rows, logs):
    write_csv(
        os.path.join(out_dirs["runs"], "factorial_evaluation_metrics.csv"),
        rows,
        [
            "condition",
            "seed",
            "delay_level",
            "sparsity_level",
            "delay_high",
            "sparsity_high",
            "delay",
            "reward_sparse_prob",
            "noise",
            "before_reward",
            "after_reward",
            "goal_rate_before",
            "goal_rate_after",
            "cds",
            "spi",
        ],
    )
    write_csv(
        os.path.join(out_dirs["runs"], "factorial_training_logs.csv"),
        logs,
        ["phase", "seed", "episode", "reward", "epsilon", "loss", "reached_goal"],
    )
    summary = factorial_summary(rows)
    effects = factorial_effects(rows)
    write_csv(
        os.path.join(out_dirs["runs"], "factorial_summary.csv"),
        summary,
        [
            "condition",
            "delay_level",
            "sparsity_level",
            "delay",
            "reward_sparse_prob",
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
        os.path.join(out_dirs["runs"], "factorial_effects.csv"),
        effects,
        [
            "metric",
            "effect",
            "estimate",
            "ci95",
            "standardized_effect",
            "t_stat",
            "p_value_normal",
            "significant_approx",
        ],
    )
    make_factorial_figures(out_dirs["figures"], summary)
    write_factorial_report(os.path.join(out_dirs["runs"], "factorial_report.md"), summary, effects)
    write_json(
        os.path.join(out_dirs["runs"], "factorial_manifest.json"),
        {
            "seeds": SEEDS,
            "episodes": EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "conditions": factorial_conditions(),
            "interpretation": interpret_factorial(summary, effects),
        },
    )


def factorial_summary(rows):
    summary = []
    for condition in [c["condition"] for c in factorial_conditions()]:
        subset = [row for row in rows if row["condition"] == condition]
        rewards = [float(row["before_reward"]) for row in subset]
        goals = [float(row["goal_rate_before"]) for row in subset]
        cds = [float(row["cds"]) for row in subset]
        spi = [float(row["spi"]) for row in subset]
        first = subset[0]
        summary.append(
            {
                "condition": condition,
                "delay_level": first["delay_level"],
                "sparsity_level": first["sparsity_level"],
                "delay": int(first["delay"]),
                "reward_sparse_prob": float(first["reward_sparse_prob"]),
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
    return summary


def factorial_effects(rows):
    output = []
    for metric, source in [
        ("reward", "before_reward"),
        ("goal_rate", "goal_rate_before"),
        ("CDS", "cds"),
        ("SPI", "spi"),
    ]:
        y = [float(row[source]) for row in rows]
        x = [[1.0, float(row["delay_high"]), float(row["sparsity_high"]), float(row["delay_high"]) * float(row["sparsity_high"])] for row in rows]
        fit = ols(x, y)
        for idx, effect in [(1, "delay_main"), (2, "sparsity_main"), (3, "delay_x_sparsity")]:
            estimate = fit["beta"][idx]
            se = fit["se"][idx]
            t_stat = estimate / se if se > 0 else 0.0
            p = math.erfc(abs(t_stat) / math.sqrt(2.0))
            output.append(
                {
                    "metric": metric,
                    "effect": effect,
                    "estimate": round(estimate, 6),
                    "ci95": round(1.96 * se, 6),
                    "standardized_effect": round(estimate / fit["residual_sd"], 6) if fit["residual_sd"] > 0 else 0.0,
                    "t_stat": round(t_stat, 6),
                    "p_value_normal": round(p, 6),
                    "significant_approx": p < 0.05,
                }
            )
    return output


def ols(x, y):
    xtx = [[sum(row[i] * row[j] for row in x) for j in range(len(x[0]))] for i in range(len(x[0]))]
    xty = [sum(row[i] * value for row, value in zip(x, y)) for i in range(len(x[0]))]
    beta = solve(xtx, xty)
    fitted = [sum(b * value for b, value in zip(beta, row)) for row in x]
    residuals = [value - pred for value, pred in zip(y, fitted)]
    df = max(1, len(y) - len(beta))
    sigma2 = sum(r * r for r in residuals) / df
    inv = inverse(xtx)
    se = [math.sqrt(max(0.0, sigma2 * inv[i][i])) for i in range(len(beta))]
    return {"beta": beta, "se": se, "residual_sd": math.sqrt(sigma2)}


def solve(matrix, rhs):
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
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


def inverse(matrix):
    n = len(matrix)
    inv = []
    for col in range(n):
        rhs = [1.0 if i == col else 0.0 for i in range(n)]
        inv.append(solve(matrix, rhs))
    return [[inv[col][row] for col in range(n)] for row in range(n)]


def make_factorial_figures(figures_dir, summary):
    for metric, field, ci, title in [
        ("cds", "cds_mean", "cds_ci95", "CDS Factorial Plot"),
        ("spi", "spi_mean", "spi_ci95", "SPI Factorial Plot"),
        ("reward", "mean_reward", "reward_ci95", "Reward Factorial Plot"),
        ("goal_rate", "goal_rate", "goal_rate_ci95", "Goal-Rate Factorial Plot"),
    ]:
        write_interaction_svg(
            os.path.join(figures_dir, f"{metric}_factorial_plot.svg"),
            summary,
            field,
            ci,
            title,
            metric.upper() if metric in ["cds", "spi"] else metric.replace("_", " ").title(),
        )


def write_interaction_svg(path, summary, field, ci_field, title, ylabel):
    width, height = 720, 450
    margin = 68
    values = [row[field] for row in summary]
    cis = [row[ci_field] for row in summary]
    ymin = min([v - c for v, c in zip(values, cis)] + [0.0])
    ymax = max([v + c for v, c in zip(values, cis)] + [1.0])
    pad = (ymax - ymin) * 0.1
    ymin -= pad
    ymax += pad
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    x_pos = {"low": margin + plot_w * 0.25, "high": margin + plot_w * 0.75}
    colors = {"low": "#245c9c", "high": "#d66b2a"}

    def sy(value):
        return margin + (ymax - value) / (ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-14}" text-anchor="middle" font-family="Arial" font-size="13">Reward delay</text>',
        f'<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{ylabel} +/- 95% CI</text>',
        f'<text x="{x_pos["low"]}" y="{height-margin+24}" text-anchor="middle" font-family="Arial" font-size="12">low</text>',
        f'<text x="{x_pos["high"]}" y="{height-margin+24}" text-anchor="middle" font-family="Arial" font-size="12">high</text>',
    ]
    for sparsity in ["low", "high"]:
        rows = [row for row in summary if row["sparsity_level"] == sparsity]
        rows.sort(key=lambda row: row["delay"])
        coords = []
        for row in rows:
            x = x_pos[row["delay_level"]]
            y = sy(row[field])
            coords.append(f"{x:.2f},{y:.2f}")
        color = colors[sparsity]
        parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for row in rows:
            x = x_pos[row["delay_level"]]
            y = sy(row[field])
            y_low = sy(row[field] - row[ci_field])
            y_high = sy(row[field] + row[ci_field])
            parts.append(f'<line x1="{x:.2f}" y1="{y_low:.2f}" x2="{x:.2f}" y2="{y_high:.2f}" stroke="#333" stroke-width="1.4"/>')
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{width-margin-145}" y="{58 if sparsity == "low" else 78}" font-family="Arial" font-size="12" fill="{color}">{sparsity} sparsity</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def interpret_factorial(summary, effects):
    by = {row["condition"]: row for row in summary}
    target = by["high_delay_high_sparsity"]
    cds_largest = target["cds_mean"] == max(row["cds_mean"] for row in summary)
    spi_largest = target["spi_mean"] == max(row["spi_mean"] for row in summary)
    learnable = target["goal_rate"] >= 0.75
    effect_map = {(row["metric"], row["effect"]): row for row in effects}
    return {
        "high_delay_high_sparsity_cds_largest": cds_largest,
        "high_delay_high_sparsity_spi_largest": spi_largest,
        "high_delay_high_sparsity_learnable": learnable,
        "cds_interaction": effect_map[("CDS", "delay_x_sparsity")],
        "spi_interaction": effect_map[("SPI", "delay_x_sparsity")],
        "supports_credit_assignment_ambiguity": bool(cds_largest and spi_largest and learnable),
    }


def write_factorial_report(path, summary, effects):
    interpretation = interpret_factorial(summary, effects)
    lines = [
        "# Factorial Credit-Assignment Ambiguity Report",
        "",
        "2x2 design: reward delay low/high crossed with reward sparsity low/high.",
        "",
        "| condition | reward | goal_rate | CDS | SPI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['condition']} | {row['mean_reward']:.3f} +/- {row['reward_ci95']:.3f} | "
            f"{row['goal_rate']:.3f} +/- {row['goal_rate_ci95']:.3f} | "
            f"{row['cds_mean']:.3f} +/- {row['cds_ci95']:.3f} | "
            f"{row['spi_mean']:.3f} +/- {row['spi_ci95']:.3f} |"
        )
    lines.extend(["", "| metric | effect | estimate | 95% CI | std. effect | p approx |", "|---|---|---:|---:|---:|---:|"])
    for row in effects:
        if row["metric"] in ["CDS", "SPI"]:
            lines.append(
                f"| {row['metric']} | {row['effect']} | {row['estimate']:.3f} | +/- {row['ci95']:.3f} | "
                f"{row['standardized_effect']:.3f} | {row['p_value_normal']:.4f} |"
            )
    lines.extend(
        [
            "",
            f"High delay + high sparsity has largest CDS: {interpretation['high_delay_high_sparsity_cds_largest']}.",
            f"High delay + high sparsity has largest SPI: {interpretation['high_delay_high_sparsity_spi_largest']}.",
            f"High delay + high sparsity remains learnable: {interpretation['high_delay_high_sparsity_learnable']}.",
            f"Supports credit-assignment ambiguity: {interpretation['supports_credit_assignment_ambiguity']}.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def analyze_existing():
    runs = os.path.join(ROOT, "runs")
    figures = os.path.join(ROOT, "figures")
    with open(os.path.join(runs, "factorial_evaluation_metrics.csv"), "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out_dirs = {"runs": runs, "figures": figures}
    write_factorial_outputs(out_dirs, rows, [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        analyze_existing()
        print("factorial analysis regenerated")
    else:
        run_factorial()
        print("factorial experiment complete")
