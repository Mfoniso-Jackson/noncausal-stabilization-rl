"""Multiple-comparison correction across the cue-validity lineage.

Policy and results: docs/multiple_comparison_correction.md. Reads every
experiment's own already-committed criteria CSV, applies Holm-Bonferroni
within each experiment's pre-registered comparison family, and reports
which cells survive. No new training is run here -- this is a
retrospective statistical-policy application over already-collected
results.
"""

import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
REPORTS = os.path.join(ROOT, "reports")


def read(path):
    with open(os.path.join(RESULTS, path), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    return [(i, pvals[i], adjusted[i], adjusted[i] < 0.05) for i in range(n)]


def build_families():
    families = {}

    rows = read("cue_validity_mve_criteria.csv")
    families["MVE (criterion 1, decorrelate-vs-naive)"] = [(f"v{r['validity']}", float(r["p_value"])) for r in rows]

    rows = read("cue_validity_proxy_dependence_criteria.csv")
    families["Proxy-dependence (criterion 1)"] = [(f"{r['agent_type']}_v{r['validity']}", float(r["p_value"])) for r in rows]

    rows = read("cue_validity_reversal_criteria.csv")
    families["Reversal (pre-registered raw-gap)"] = [(f"{r['agent_type']}_v{r['validity']}", float(r["p_value"])) for r in rows]

    rows = read("cue_validity_nonlinear_coordinate_test.csv")
    families["Nonlinear (coordinate-dependence test)"] = [(f"{r['architecture']}_v{r['validity']}", float(r["p_value"])) for r in rows]

    rows = read("cue_validity_neural_reversal_normalized.csv")
    entries = [(f"{r['condition']}_v{r['validity']}", float(r["p_value"])) for r in rows if r["condition"] != "neural_raw"]
    topup_rows = read("cue_validity_neural_raw_topup_normalized.csv")
    entries += [(f"neural_raw_topup_v{r['validity']}", float(r["p_value"])) for r in topup_rows]
    families["Neural reversal (pre-registered normalized, neural_raw via topup)"] = entries

    rows = read("cue_validity_optimizer_ablation_primary.csv")
    families["Optimizer ablation (primary)"] = [(f"{r['condition']}_v{r['validity']}", float(r["p_value"])) for r in rows]

    rows = read("cue_validity_replay_buffer_sweep_criteria.csv")
    families["Replay buffer sweep"] = [(f"buf{r['buffer_size']}_v{r['validity']}", float(r["p_value"])) for r in rows]

    capacity_path = os.path.join(RESULTS, "cue_validity_capacity_sweep_criteria.csv")
    if os.path.exists(capacity_path):
        rows = read("cue_validity_capacity_sweep_criteria.csv")
        families["Capacity sweep"] = [(f"w{r['width']}_v{r['validity']}", float(r["p_value"])) for r in rows]

    return families


def run():
    families = build_families()
    shuffle_rows = read("cue_validity_shuffle_placebo_criteria.csv")

    lines = [
        "# Multiple-Comparison Correction: Analysis Output",
        "",
        "Policy and interpretation: docs/multiple_comparison_correction.md.",
        "",
    ]
    for family_name, entries in families.items():
        labels = [e[0] for e in entries]
        pvals = [e[1] for e in entries]
        results = holm_bonferroni(pvals)
        lines.append(f"## {family_name} (n={len(pvals)})")
        lines.append("")
        lines.append("| cell | raw p | holm-adjusted p | survives (alpha=0.05) |")
        lines.append("|---|---:|---:|---|")
        for (idx, p, adj, reject), label in zip(results, labels):
            lines.append(f"| {label} | {p:.6f} | {adj:.6f} | {reject} |")
        lines.append("")

    lines.append("## Shuffle/placebo control (excluded from discovery family; success = non-significance)")
    lines.append("")
    lines.append("| validity | raw p (uncorrected; success criterion is p>=0.05) |")
    lines.append("|---:|---:|")
    for r in shuffle_rows:
        lines.append(f"| {r['validity']} | {float(r['p_value']):.6f} |")

    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "multiple_comparison_correction_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
