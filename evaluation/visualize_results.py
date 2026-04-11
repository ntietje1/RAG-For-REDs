"""
visualize_results.py
--------------------
Generates seven charts from evaluation/results.json:

  1. Grouped bar chart    – Per-category metric breakdown
  2. Diverging delta chart– Strategy improvement / regression vs. baseline
  3. Scatter plot         – Context Precision vs. Recall trade-off
  4. Per-question heatmap – Individual question scores across strategies × metrics
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
RESULTS_FILE = SCRIPT_DIR / "results.json"
OUTPUT_DIR = SCRIPT_DIR / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── style ───────────────────────────────────────────────────────────────────
STRATEGIES = ["baseline", "temporal_only", "authority_only", "full"]
STRATEGY_LABELS = {
    "baseline": "Baseline",
    "temporal_only": "Temporal Only",
    "authority_only": "Authority Only",
    "full": "Temporal + Authority",
}
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Relevancy",
    "context_precision": "Context\nPrecision",
    "context_recall": "Context\nRecall",
}
# Difficulty order for line chart
CATEGORIES = ["Evergreen", "Mixed", "Version-sensitive"]
CATEGORY_LABELS = {
    "Evergreen": "Evergreen\n(17 Qs)",
    "Mixed": "Mixed\n(13 Qs)",
    "Version-sensitive": "Version-sensitive\n(17 Qs)",
}

COLORS = {
    "baseline":       "#4C72B0",
    "full":           "#DD8452",
    "temporal_only":  "#55A868",
    "authority_only": "#C44E52",
}
MARKERS = {
    "baseline":       "o",
    "full":           "s",
    "temporal_only":  "^",
    "authority_only": "D",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})


# ── helpers ─────────────────────────────────────────────────────────────────

def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def aggregate(entries: list, group_by: str | None = None) -> dict:
    """
    Returns {group_key: {metric: mean}} if group_by is given,
    otherwise {metric: mean} for the whole list.
    """
    if group_by is None:
        agg = defaultdict(list)
        for e in entries:
            for m in METRICS:
                v = e["metrics"].get(m)
                if v is not None:
                    agg[m].append(v)
        return {m: (sum(vs) / len(vs) if vs else 0) for m, vs in agg.items()}

    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        key = e.get(group_by, "Unknown")
        for m in METRICS:
            v = e["metrics"].get(m)
            if v is not None:
                groups[key][m].append(v)
    return {
        grp: {m: (sum(vs) / len(vs) if vs else 0) for m, vs in metrics.items()}
        for grp, metrics in groups.items()
    }


# ── chart 1: grouped bar ──────────────────────────────────────────────────────

def plot_grouped_bar(data: dict):
    """Grouped bar chart – per-category metric breakdown (one subplot per category + overall)."""
    all_cats = CATEGORIES + ["Overall"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes = axes.flatten()

    x = np.arange(len(METRICS))
    n = len(STRATEGIES)
    width = 0.18
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for ax, cat in zip(axes, all_cats):
        for i, strategy in enumerate(STRATEGIES):
            if cat == "Overall":
                scores = aggregate(data[strategy])
            else:
                by_cat = aggregate(data[strategy], group_by="temporality")
                scores = by_cat.get(cat, {m: 0 for m in METRICS})

            vals = [scores.get(m, 0) for m in METRICS]
            bars = ax.bar(x + offsets[i], vals, width,
                          label=STRATEGY_LABELS[strategy],
                          color=COLORS[strategy], alpha=0.88, edgecolor="white")

        ax.set_title(cat if cat == "Overall" else f"{cat}", fontsize=18, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS[m].replace("\n", " ") for m in METRICS], fontsize=13)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", fontsize=14)
        ax.tick_params(axis="y", labelsize=12)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    # shared legend below
    handles = [mpatches.Patch(color=COLORS[s], label=STRATEGY_LABELS[s])
               for s in STRATEGIES]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.06), fontsize=13)

    fig.suptitle("Metric Breakdown by Category", fontsize=22, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUTPUT_DIR / "1_grouped_bar_by_category.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── chart 2: diverging delta ──────────────────────────────────────────────────

def plot_diverging_delta(data: dict):
    """Diverging bar chart – Δ vs baseline per strategy × metric, split by category."""
    compare_strategies = ["full", "temporal_only", "authority_only"]
    all_cats = CATEGORIES + ["Overall"]

    fig, axes = plt.subplots(1, len(all_cats), figsize=(16, 5), sharey=True)

    y = np.arange(len(METRICS))
    n = len(compare_strategies)
    height = 0.22
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * height

    for ax, cat in zip(axes, all_cats):
        if cat == "Overall":
            baseline_scores = aggregate(data["baseline"])
        else:
            baseline_scores = aggregate(data["baseline"], group_by="temporality").get(
                cat, {m: 0 for m in METRICS})

        for i, strategy in enumerate(compare_strategies):
            if cat == "Overall":
                strat_scores = aggregate(data[strategy])
            else:
                strat_scores = aggregate(data[strategy], group_by="temporality").get(
                    cat, {m: 0 for m in METRICS})

            deltas = [strat_scores.get(m, 0) - baseline_scores.get(m, 0) for m in METRICS]
            colors_bar = COLORS[strategy]

            ax.barh(y + offsets[i], deltas, height,
                    color=colors_bar, alpha=0.88, edgecolor="white",
                    label=STRATEGY_LABELS[strategy])

        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(cat, fontsize=11)
        ax.set_yticks(y)
        if ax == axes[0]:
            ax.set_yticklabels([METRIC_LABELS[m].replace("\n", " ") for m in METRICS])
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_xlabel("Δ vs Baseline")

    handles = [mpatches.Patch(color=COLORS[s], label=STRATEGY_LABELS[s])
               for s in compare_strategies]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Score Delta vs Baseline (positive = improvement)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = OUTPUT_DIR / "2_diverging_delta_vs_baseline.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# ── chart 3: scatter precision vs recall ─────────────────────────────────────

def plot_scatter_pr(data: dict):
    """Scatter – context_precision vs context_recall, coloured by strategy, shaped by category."""
    SHAPE_MAP = {"Evergreen": "o", "Version-sensitive": "s", "Mixed": "^"}
    SHAPE_SIZE = 90

    fig, ax = plt.subplots(figsize=(7.5, 6))

    for strategy in STRATEGIES:
        by_cat = aggregate(data[strategy], group_by="temporality")
        # also plot Overall
        overall = aggregate(data[strategy])
        by_cat["Overall"] = overall

        for cat, scores in by_cat.items():
            prec = scores.get("context_precision", 0)
            rec = scores.get("context_recall", 0)
            marker = SHAPE_MAP.get(cat, "P")
            size = SHAPE_SIZE * 1.6 if cat == "Overall" else SHAPE_SIZE
            edgecolor = "black" if cat == "Overall" else "white"
            lw = 1.5 if cat == "Overall" else 0.6
            ax.scatter(prec, rec, color=COLORS[strategy],
                       marker=marker, s=size,
                       edgecolors=edgecolor, linewidths=lw, zorder=3,
                       alpha=0.9)
            label = f"{STRATEGY_LABELS[strategy]}\n({cat})"
            ax.annotate("", (prec, rec),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7, color=COLORS[strategy])

    # diagonal reference line (precision = recall)
    lims = [0.2, 1.0]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.4, label="Precision = Recall")

    # legend patches: strategy colours
    strat_handles = [mpatches.Patch(color=COLORS[s], label=STRATEGY_LABELS[s])
                     for s in STRATEGIES]
    # legend markers: category shapes
    cat_handles = [plt.Line2D([0], [0], marker=m, color="grey", linestyle="None",
                              markersize=8, label=c)
                   for c, m in SHAPE_MAP.items()]
    cat_handles.append(plt.Line2D([0], [0], marker="P", color="grey", linestyle="None",
                                  markersize=8, label="Overall"))
    first_legend = ax.legend(handles=strat_handles, title="Strategy",
                             loc="upper left", fontsize=9, framealpha=0.9)
    ax.add_artist(first_legend)
    ax.legend(handles=cat_handles, title="Category",
              loc="lower right", fontsize=9, framealpha=0.9)

    ax.set_xlabel("Context Precision")
    ax.set_ylabel("Context Recall")
    ax.set_xlim(0.55, 1.02)
    ax.set_ylim(0.20, 0.90)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title("Context Precision vs. Recall Trade-off", fontweight="bold")

    fig.tight_layout()
    path = OUTPUT_DIR / "3_scatter_precision_vs_recall.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── chart 4: per-question heatmap ─────────────────────────────────────────────

def plot_question_heatmap(data: dict):
    """
    Heatmap grid – one subplot per metric (2 × 2).
    Rows  = individual questions, sorted by temporality group then ID.
    Cols  = strategies.
    Color = score (red → yellow → green via RdYlGn).

    Horizontal dividers separate temporality groups; group labels appear on the
    right side of the last column.  A single shared colour-bar sits at the bottom.
    """
    TEMP_ORDER = ["Evergreen", "Mixed", "Version-sensitive"]
    TEMP_BAND_COLORS = {
        "Evergreen":        "#d5f5e3",
        "Mixed":            "#fef9e7",
        "Version-sensitive": "#fdedec",
    }

    # ── build a stable, sorted question list ──────────────────────────────
    def sort_key(e):
        order = {t: i for i, t in enumerate(TEMP_ORDER)}
        return (order.get(e["temporality"], 99), e["id"])

    sorted_entries = sorted(data["baseline"], key=sort_key)
    q_ids   = [e["id"]          for e in sorted_entries]
    q_temps = [e["temporality"] for e in sorted_entries]
    id_to_row = {qid: i for i, qid in enumerate(q_ids)}

    n_q = len(q_ids)
    n_s = len(STRATEGIES)

    # ── find group boundary rows ───────────────────────────────────────────
    boundaries = []          # row indices where a new group starts (excl. row 0)
    group_spans = []         # (start_row, end_row, label) for each temporality group
    prev = None
    grp_start = 0
    for i, t in enumerate(q_temps):
        if t != prev:
            if prev is not None:
                boundaries.append(i)
                group_spans.append((grp_start, i - 1, prev))
            grp_start = i
            prev = t
    group_spans.append((grp_start, n_q - 1, prev))   # last group

    # ── build score matrices ───────────────────────────────────────────────
    matrices = {}
    for metric in METRICS:
        mat = np.full((n_q, n_s), np.nan)
        for j, strategy in enumerate(STRATEGIES):
            for e in data[strategy]:
                row = id_to_row.get(e["id"])
                if row is not None:
                    v = e["metrics"].get(metric)
                    if v is not None:
                        mat[row, j] = v
        matrices[metric] = mat

    # ── figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 2,
        figsize=(15, 22),
        gridspec_kw={"hspace": 0.08, "wspace": 0.35},
    )
    axes = axes.flatten()

    cmap = plt.get_cmap("RdYlGn")
    im_ref = None                          # keep one imshow handle for the colorbar

    for ax, metric in zip(axes, METRICS):
        mat = matrices[metric]

        # ── background bands per temporality group ─────────────────────────
        for start, end, label in group_spans:
            ax.axhspan(
                start - 0.5, end + 0.5,
                color=TEMP_BAND_COLORS.get(label, "#ffffff"),
                alpha=0.35, zorder=0,
            )

        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1, zorder=1)
        if im_ref is None:
            im_ref = im

        # ── value annotations inside each cell ────────────────────────────
        for row in range(n_q):
            for col in range(n_s):
                v = mat[row, col]
                if not np.isnan(v):
                    txt_color = "black" if 0.35 < v < 0.85 else "white"
                    ax.text(
                        col, row, f"{v:.2f}",
                        ha="center", va="center",
                        fontsize=5.5, color=txt_color, fontweight="bold",
                    )

        # ── horizontal dividers between temporality groups ─────────────────
        for b in boundaries:
            ax.axhline(b - 0.5, color="black", linewidth=1.8, zorder=2)

        # ── group labels on the right ──────────────────────────────────────
        for start, end, label in group_spans:
            mid = (start + end) / 2
            ax.annotate(
                label,
                xy=(1.01, 1 - (mid / n_q)),
                xycoords="axes fraction",
                fontsize=7.5, rotation=270,
                va="center", ha="left",
                color="#444444",
            )

        # ── axes labels ───────────────────────────────────────────────────
        ax.set_xticks(range(n_s))
        ax.set_xticklabels(
            [STRATEGY_LABELS[s] for s in STRATEGIES],
            rotation=35, ha="right", fontsize=8.5,
        )
        ax.set_yticks(range(n_q))
        ax.set_yticklabels(q_ids, fontsize=6.5)
        ax.set_title(
            METRIC_LABELS[metric].replace("\n", " "),
            fontsize=11, pad=6,
        )
        ax.tick_params(axis="x", which="both", length=0)
        ax.tick_params(axis="y", which="both", length=0)

    # ── shared colour-bar ──────────────────────────────────────────────────
    cbar = fig.colorbar(
        im_ref, ax=axes, orientation="horizontal",
        fraction=0.02, pad=0.03, shrink=0.6,
    )
    cbar.set_label("Score  (0 = worst · 1 = best)", fontsize=10)

    fig.suptitle(
        "Per-Question Scores — All Strategies × All Metrics\n"
        "(rows grouped by temporality; horizontal lines = group boundaries)",
        fontsize=13, fontweight="bold", y=1.005,
    )
    path = OUTPUT_DIR / "4_question_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {RESULTS_FILE} …")
    data = load_results(RESULTS_FILE)

    print("Generating charts …")
    plot_grouped_bar(data)
    plot_diverging_delta(data)
    plot_scatter_pr(data)
    plot_question_heatmap(data)

    print(f"\nAll charts saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
