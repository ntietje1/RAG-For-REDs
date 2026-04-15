from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

# ── Strategy metadata ─────────────────────────────────────────────────────────

#: Canonical order used for tables, charts, and iteration.
STRATEGY_ORDER: list[str] = ["baseline", "temporal_only", "authority_only", "full"]

#: Human-readable labels for each strategy key.
STRATEGY_LABELS: dict[str, str] = {
    "baseline":       "Baseline",
    "temporal_only":  "Temporal Only",
    "authority_only": "Authority Only",
    "full":           "Temporal + Authority",
}

#: Hex colours — consistent across every chart in both matplotlib and Plotly.
STRATEGY_COLORS: dict[str, str] = {
    "baseline":       "#4C72B0",
    "temporal_only":  "#55A868",
    "authority_only": "#C44E52",
    "full":           "#DD8452",
}

# ── Metric metadata ───────────────────────────────────────────────────────────

#: Canonical order used for tables, charts, and iteration.
METRICS: list[str] = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

#: Human-readable labels — plain text, no newlines.
#: matplotlib callers that previously used short/wrapped labels can either use
#: these as-is or apply their own local overrides for axis tick formatting.
METRIC_LABELS: dict[str, str] = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
}

# ── Temporality metadata ──────────────────────────────────────────────────────

#: Three temporality buckets in difficulty / recency-sensitivity order.
TEMPORALITIES: list[str] = ["Evergreen", "Mixed", "Version-sensitive"]

#: Maps each temporality name to a sort index (used for stable question ordering).
TEMP_ORDER_IDX: dict[str, int] = {t: i for i, t in enumerate(TEMPORALITIES)}

# ── Data helpers ──────────────────────────────────────────────────────────────

#: Default path to the evaluation results file.
RESULTS_PATH: Path = Path(__file__).parent / "results.json"


def load_results(path: Path | None = None) -> dict:
    """Load evaluation/results.json (or a custom path) and return the raw dict."""
    with open(path or RESULTS_PATH) as f:
        return json.load(f)


def active_strategies(data: dict) -> list[str]:
    """Return strategy keys that are present in *data*, in canonical order."""
    return [s for s in STRATEGY_ORDER if s in data]


def sort_questions(entries: list[dict]) -> list[dict]:
    """Sort question entries by temporality group (Evergreen → Mixed → Version-sensitive)
    then by question ID within each group."""
    return sorted(
        entries,
        key=lambda e: (TEMP_ORDER_IDX.get(e["temporality"], 99), e["id"]),
    )


# ── Core aggregation ──────────────────────────────────────────────────────────

def aggregate(entries: list[dict], group_by: str | None = None) -> dict:
    """Compute per-metric mean scores from a list of evaluation entries.

    Parameters
    ----------
    entries:
        List of result dicts, each containing a ``"metrics"`` sub-dict.
    group_by:
        If ``None``, returns ``{metric: mean_float}`` across all entries.
        If a field name (e.g. ``"temporality"``), returns
        ``{group_value: {metric: mean_float}}``.
    """
    if group_by is None:
        agg: dict[str, list] = defaultdict(list)
        for e in entries:
            for m in METRICS:
                v = e["metrics"].get(m)
                if v is not None:
                    agg[m].append(v)
        return {m: (sum(vs) / len(vs) if vs else 0.0) for m, vs in agg.items()}

    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        key = e.get(group_by, "Unknown")
        for m in METRICS:
            v = e["metrics"].get(m)
            if v is not None:
                groups[key][m].append(v)
    return {
        grp: {m: (sum(vs) / len(vs) if vs else 0.0) for m, vs in metrics.items()}
        for grp, metrics in groups.items()
    }
