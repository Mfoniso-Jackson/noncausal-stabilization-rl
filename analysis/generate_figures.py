"""Generates the manuscript's figures directly from already-committed,
verified results CSVs -- no new data collection, no numbers not already
in a committed report. See docs/research_report.md Section 7 for the
source of every number plotted here.

Palette and mark specs follow the project's dataviz skill: categorical
slots 1 (blue, #2a78d6) and 2 (orange, #eb6834) for <=2-series charts;
slot 3 (aqua, #1baf7a) added only where a third series is needed (the
first three slots validate all-pairs under CVD checks). Light surface
#fcfcfb, primary ink #0b0b0b, muted ink #898781, hairline gridlines
#e1e0d9. 2px lines, >=8px markers with a surface-color ring, thin bars,
legend for >=2 series, sparing direct labels, single axis always.
"""
import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 1.0,
        "grid.linestyle": "-",
        "font.size": 11,
    }
)


def read_csv(name):
    with open(os.path.join(RESULTS_DIR, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def strip_spines(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
        if side in keep:
            spine.set_color(BASELINE)
    ax.tick_params(length=0)


def savefig(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Figure 1: capacity dose-response (DQN neural_raw), Cohen's d vs. width
# Source: results/cue_validity_capacity_sweep_criteria.csv
# ---------------------------------------------------------------------------

def figure_capacity_dose_response():
    rows = read_csv("cue_validity_capacity_sweep_criteria.csv")
    widths = sorted({int(r["width"]) for r in rows})
    series = {"0.5": (BLUE, "cue validity 0.5"), "0.95": (ORANGE, "cue validity 0.95")}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = list(range(len(widths)))
    for validity, (color, label) in series.items():
        ys = [float(next(r["cohens_d"] for r in rows if r["width"] == str(w) and r["validity"] == validity)) for w in widths]
        significant = [next(r["has_effect"] for r in rows if r["width"] == str(w) and r["validity"] == validity) == "True" for w in widths]
        ax.plot(x, ys, color=color, linewidth=2, zorder=2, label=label)
        for xi, yi, sig in zip(x, ys, significant):
            ax.plot(
                xi, yi, marker="o", markersize=8,
                markeredgecolor=color, markeredgewidth=1.5,
                markerfacecolor=color if sig else SURFACE, zorder=3,
            )

    ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("hidden-layer width")
    ax.set_ylabel("Cohen's d (decorrelate vs. naive)")
    ax.set_title("neural_raw: effect size shrinks with capacity", loc="left", fontsize=12, color=INK_PRIMARY, pad=12)
    strip_spines(ax)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK_SECONDARY)
    fig.text(0.02, -0.02, "Filled marker = survives Holm-Bonferroni correction; open = does not.", fontsize=9, color=INK_MUTED, ha="left")
    savefig(fig, "capacity_dose_response.png")


# ---------------------------------------------------------------------------
# Figure 2: reversal vs. decorrelation fraction-adapted (H3)
# Source: results/cue_validity_reversal_normalized.csv
# ---------------------------------------------------------------------------

def figure_reversal_vs_decorrelation():
    rows = read_csv("cue_validity_reversal_normalized.csv")
    conditions = [("linear_raw", "0.5"), ("linear_raw", "0.95"), ("linear_orthogonal", "0.5"), ("linear_orthogonal", "0.95")]
    labels = ["linear_raw\nv=0.5", "linear_raw\nv=0.95", "linear_orthogonal\nv=0.5", "linear_orthogonal\nv=0.95"]

    reversal = []
    decorrelation = []
    for cond, v in conditions:
        row = next(r for r in rows if r["agent_type"] == cond and r["validity"] == v)
        reversal.append(float(row["reversal_fraction_adapted"]))
        decorrelation.append(float(row["decorrelation_fraction_adapted"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = list(range(len(labels)))
    width = 0.32
    bars_r = ax.bar([xi - width / 2 - 0.02 for xi in x], reversal, width=width, color=BLUE, label="reversal", zorder=2)
    bars_d = ax.bar([xi + width / 2 + 0.02 for xi in x], decorrelation, width=width, color=ORANGE, label="decorrelation", zorder=2)

    for b in list(bars_r) + list(bars_d):
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}",
            ha="center", va="bottom", fontsize=9, color=INK_SECONDARY,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(reversal + decorrelation) * 1.18)
    ax.set_ylabel("fraction of required adaptation reached\n(30-episode post-intervention window)")
    ax.set_title("An active correction adapts faster than a passive one", loc="left", fontsize=12, color=INK_PRIMARY, pad=38)
    strip_spines(ax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, labelcolor=INK_SECONDARY)
    savefig(fig, "reversal_vs_decorrelation.png")


# ---------------------------------------------------------------------------
# Figure 3: representation gradient at cue validity 0.95 (headline figure)
# Sources: results/cue_validity_proxy_dependence_criteria.csv,
# reports/cue_validity_nonlinear_report.md, results/cue_validity_ppo_seed_topup_criteria.csv
# ---------------------------------------------------------------------------

def figure_representation_gradient():
    # (label, cohens_d, survives_correction) at validity 0.95, most-to-least dependence
    data = [
        ("linear_raw", 23.589, True),
        ("linear_orthogonal", 17.826, True),
        ("linear_interaction_only", 5.377, True),
        ("neural_raw", 1.623, True),
        ("neural_interaction_only", 0.733, False),
        ("PPO (total-matched, n=30)", 0.305, False),
        ("tabular", -0.048, False),
    ]
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    survives = [d[2] for d in data]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = list(range(len(labels)))[::-1]
    for yi, val, sig, label in zip(y, values, survives, labels):
        ax.barh(
            yi, val, height=0.6, color=BLUE,
            hatch=None if sig else "//", edgecolor=BLUE if sig else INK_MUTED,
            alpha=1.0 if sig else 0.55, zorder=2,
        )
        # Small-magnitude bars are visually indistinguishable from zero, so
        # always label them to the right of the axis to avoid colliding
        # with the y-tick labels sitting just left of x=0.
        if abs(val) < 1.0:
            label_x, ha = 0.6, "left"
        else:
            label_x, ha = (val + 0.6, "left") if val >= 0 else (val - 0.6, "right")
        ax.text(label_x, yi, f"{val:.2f}", va="center", ha=ha, fontsize=9, color=INK_SECONDARY)

    ax.axvline(0, color=BASELINE, linewidth=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d (decorrelate vs. naive, cue validity 0.95)")
    ax.set_title("Persistence is graded by representation, not binary", loc="left", fontsize=12, color=INK_PRIMARY, pad=12)
    strip_spines(ax, keep=("bottom",))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    fig.text(0.02, -0.02, "Solid = survives Holm-Bonferroni correction within its own family; hatched = does not.", fontsize=9, color=INK_MUTED, ha="left")
    savefig(fig, "representation_gradient.png")


if __name__ == "__main__":
    figure_capacity_dose_response()
    figure_reversal_vs_decorrelation()
    figure_representation_gradient()
