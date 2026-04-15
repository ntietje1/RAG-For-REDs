"""
RAG-For-REDs Streamlit UI

Run with:
    streamlit run app.py

Requires:
    - Docker running Qdrant: docker compose up -d
    - OPENROUTER_API_KEY set in .env
"""

from __future__ import annotations

import json
from pathlib import Path

from collections import defaultdict
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from evaluation.utils import (
    STRATEGY_ORDER,
    STRATEGY_LABELS,
    STRATEGY_COLORS,
    METRICS,
    METRIC_LABELS,
    TEMPORALITIES,
    TEMP_ORDER_IDX,
    aggregate,
    active_strategies,
    sort_questions,
    load_results,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG-For-REDs",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── App-level constants (not shared with visualize_results.py) ────────────────

PRESETS: dict[str, dict] = {
    "Baseline": {
        "temporal": False, "authority": False, "cross_encoder": True, "expansion": True,
    },
    "Temporal Only": {
        "temporal": True, "authority": False, "cross_encoder": True, "expansion": True,
    },
    "Authority Only": {
        "temporal": False, "authority": True, "cross_encoder": True, "expansion": True,
    },
    "Full (Temporal + Authority)": {
        "temporal": True, "authority": True, "cross_encoder": True, "expansion": True,
    },
}

SOURCE_DISPLAY = {
    "riot_patch_notes": "Riot Patch Notes",
    "lolalytics": "Lolalytics",
    "wiki": "Wiki",
    "reddit": "Reddit",
}
SOURCE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

SCOPE_COLORS = {
    "evergreen": "green",
    "version-sensitive": "red",
    "mixed": "orange",
}

# ── Cached resources ──────────────────────────────────────────────────────────


@st.cache_resource
def get_vector_store():
    """Connect to Qdrant once per process. Returns None on failure."""
    try:
        from indexing.store import VectorStore
        return VectorStore()
    except Exception:
        return None


@st.cache_resource
def get_pipeline(
    use_temporal: bool,
    use_authority: bool,
    use_cross_encoder: bool,
    use_expansion: bool,
):
    """Build EnhancedRAG keyed by feature flags. Compiles LangGraph once per config."""
    from retrieval.enhanced import EnhancedRAG
    store = get_vector_store()
    if store is None:
        return None
    return EnhancedRAG(
        store=store,
        use_temporal=use_temporal,
        use_authority=use_authority,
        use_cross_encoder=use_cross_encoder,
        use_expansion=use_expansion,
        final_k=5,
    )


@st.cache_data
def load_questions() -> list[dict]:
    path = Path(__file__).parent / "evaluation" / "questions.json"
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_eval_results() -> dict | None:
    """Load results.json; returns None if the file does not exist yet."""
    from evaluation.utils import RESULTS_PATH
    if not RESULTS_PATH.exists():
        return None
    return load_results()


# ── Result renderer ───────────────────────────────────────────────────────────


def render_result(result: dict, question: str) -> None:
    st.markdown("---")

    # Answer
    st.info(f"**Answer**\n\n{result['answer']}")

    # Classification
    clf = result.get("classification") or {}
    if clf:
        with st.expander("Classification Details", expanded=True):
            col1, col2, col3 = st.columns([1, 2, 2])

            with col1:
                scope = clf.get("temporal_scope", "unknown")
                color = SCOPE_COLORS.get(scope, "gray")
                st.markdown("**Temporal Scope**")
                st.markdown(f":{color}[**{scope.upper()}**]")

                sensitivity = clf.get("temporal_sensitivity", 0.0)
                st.markdown("**Temporal Sensitivity**")
                st.progress(float(sensitivity), text=f"{sensitivity:.2f}")

                target = clf.get("target_patch")
                if target:
                    st.markdown(f"**Target Patch:** `{target}`")

                reasoning = clf.get("reasoning", "")
                if reasoning:
                    st.caption(reasoning)

            with col2:
                st.markdown("**Authority Weights**")
                weights: dict = clf.get("authority_weights") or {}
                if weights:
                    labels = [SOURCE_DISPLAY.get(k, k) for k in weights]
                    values = list(weights.values())
                    fig = go.Figure(go.Bar(
                        x=labels,
                        y=values,
                        marker_color=SOURCE_COLORS[: len(labels)],
                        text=[f"{v:.2f}" for v in values],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        height=220,
                        margin=dict(l=10, r=10, t=10, b=40),
                        yaxis=dict(range=[0, 1.25], title="Weight"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col3:
                alts = clf.get("alternate_queries") or []
                if alts:
                    st.markdown("**Alternate Queries Used**")
                    for i, alt in enumerate(alts, 1):
                        st.markdown(f"{i}. _{alt}_")

    # Sources table
    sources = result.get("sources") or []
    if sources:
        st.markdown("#### Retrieved Sources")
        df = pd.DataFrame(sources)
        display_cols = ["source"]
        if "score" in df.columns:
            display_cols.append("score")
            df["score"] = df["score"].round(4)
        if "adjusted_score" in df.columns and df["adjusted_score"].notna().any():
            display_cols.append("adjusted_score")
            df["adjusted_score"] = df["adjusted_score"].round(4)
        if "url" in df.columns:
            display_cols.append("url")

        col_cfg: dict = {
            "score": st.column_config.NumberColumn("Cosine Score", format="%.4f"),
            "adjusted_score": st.column_config.NumberColumn("Adjusted Score", format="%.4f"),
        }
        if "url" in display_cols:
            col_cfg["url"] = st.column_config.LinkColumn("URL")

        st.dataframe(df[display_cols], use_container_width=True, hide_index=True, column_config=col_cfg)

    # Retrieved chunks
    chunks = result.get("retrieved_chunks") or []
    if chunks:
        st.markdown("#### Retrieved Chunks")
        for i, chunk in enumerate(chunks, 1):
            source_label = chunk.get("source", "unknown")
            patch = chunk.get("patch_version", "")
            score = chunk.get("adjusted_score") or chunk.get("score", 0)
            header = f"Chunk {i} — {source_label}"
            if patch:
                header += f" (patch {patch})"
            if score:
                header += f" | score: {float(score):.4f}"

            with st.expander(header, expanded=(i == 1)):
                st.markdown(chunk.get("text", ""))
                meta1, meta2, meta3 = st.columns(3)
                with meta1:
                    url = chunk.get("url")
                    if url:
                        st.markdown(f"[Source URL]({url})")
                with meta2:
                    date = chunk.get("date")
                    if date:
                        st.caption(f"Date: {date}")
                with meta3:
                    ctype = chunk.get("content_type")
                    if ctype:
                        st.caption(f"Type: {ctype}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("RAG-For-REDs")
    st.caption("League of Legends RAG Assistant")
    st.markdown("---")
    st.header("Pipeline Configuration")

    preset_names = list(PRESETS.keys())
    preset_choice = st.selectbox(
        "Configuration Preset",
        options=["Custom"] + preset_names,
        key="preset",
        help="Quickly apply one of the four ablation configurations. Sets all toggles below.",
    )
    st.caption("Selecting a preset updates all feature toggles automatically.")

    defaults = PRESETS.get(preset_choice, {
        "temporal": True, "authority": True, "cross_encoder": False, "expansion": True,
    })

    # When preset changes, push new values directly into session_state BEFORE
    # rendering the toggles. Writing to session_state before widget render is the
    # only reliable way to programmatically set widget values in Streamlit.
    if st.session_state.get("_last_preset") != preset_choice:
        st.session_state["_last_preset"] = preset_choice
        st.session_state["tog_temporal"] = defaults["temporal"]
        st.session_state["tog_authority"] = defaults["authority"]
        st.session_state["tog_cross_encoder"] = defaults["cross_encoder"]
        st.session_state["tog_expansion"] = defaults["expansion"]

    st.markdown("---")
    st.subheader("Feature Toggles")
    st.caption("Override individual features, or use the preset above for quick selection.")

    use_temporal = st.toggle(
        "Temporal Decay", value=defaults["temporal"], key="tog_temporal",
        help="Apply exponential decay to chunks from older patches.",
    )
    use_authority = st.toggle(
        "Authority Weighting", value=defaults["authority"], key="tog_authority",
        help="Boost/demote chunks by source trustworthiness for this query type.",
    )
    use_cross_encoder = st.toggle(
        "Cross-Encoder Reranking", value=defaults["cross_encoder"], key="tog_cross_encoder",
        help="Use cross-encoder/ms-marco-MiniLM-L-6-v2 to rerank candidates (slow on first use).",
    )
    use_expansion = st.toggle(
        "Query Expansion", value=defaults["expansion"], key="tog_expansion",
        help="Generate 2-3 alternate phrasings and retrieve for each.",
    )

    st.markdown("---")
    st.caption("Generator: Gemini Flash via OpenRouter")
    st.caption("Retrieval: text-embedding-3-small + Qdrant")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_query, tab_eval = st.tabs(["Query", "Evaluation Results"])

# ==============================================================================
# QUERY TAB
# ==============================================================================

with tab_query:
    questions = load_questions()

    # Group by category
    grouped: dict[str, list] = defaultdict(list)
    for q in questions:
        grouped[q["category"]].append(q)

    # Build selectbox options
    q_options: dict[str, dict | None] = {"— Type your own question below —": None}
    for category in sorted(grouped.keys()):
        for q in grouped[category]:
            label = f"[{q['id']}] {q['question'][:80]}{'...' if len(q['question']) > 80 else ''}"
            q_options[label] = q

    col_input, col_meta = st.columns([3, 1])

    with col_input:
        selected_label = st.selectbox(
            "Load a sample question",
            options=list(q_options.keys()),
            key="q_select",
            help=(
                "Pick one of the 40 evaluation questions to auto-fill the text box below. "
                "You can edit it freely before running."
            ),
        )
        selected_q = q_options[selected_label]

        # Push the selected question into session_state BEFORE rendering the
        # text_area. The value= parameter is only applied on the very first
        # render; after that, session_state controls the widget's value.
        if st.session_state.get("_last_q_select") != selected_label:
            st.session_state["_last_q_select"] = selected_label
            st.session_state["q_text"] = selected_q["question"] if selected_q else ""

        user_question = st.text_area(
            "Your question",
            height=100,
            placeholder="Ask anything about League of Legends...",
            key="q_text",
        )

        run_btn = st.button("Run Query", type="primary")

    with col_meta:
        if selected_q:
            st.markdown("**Question info**")
            st.markdown(f"**ID:** `{selected_q['id']}`")
            st.markdown(f"**Temporality:** {selected_q['temporality']}")
            st.markdown(f"**Category:** {selected_q['category']}")
            if selected_q.get("expected_answer"):
                with st.expander("Expected answer"):
                    st.write(selected_q["expected_answer"])

    # ── Run query ──────────────────────────────────────────────────────────────

    if run_btn:
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            store = get_vector_store()
            if store is None:
                st.error(
                    "Cannot connect to Qdrant at localhost:6333. "
                    "Start the Docker container with `docker compose up -d` and refresh."
                )
                st.stop()
            else:
                pipeline = get_pipeline(use_temporal, use_authority, use_cross_encoder, use_expansion)
                if pipeline is None:
                    st.error("Failed to initialize pipeline. Check Qdrant connection and API keys.")
                else:
                    spinner_msg = "Running RAG pipeline"
                    if use_cross_encoder:
                        spinner_msg += " (cross-encoder may be slow on first use)"
                    spinner_msg += "..."
                    with st.spinner(spinner_msg):
                        try:
                            result = pipeline.query(user_question.strip())
                            st.session_state["last_result"] = result
                            st.session_state["last_question"] = user_question.strip()
                        except Exception as e:
                            st.error(f"Pipeline error: {e}")
                            st.session_state.pop("last_result", None)

    # ── Display persisted result ───────────────────────────────────────────────

    if st.session_state.get("last_result"):
        render_result(
            st.session_state["last_result"],
            st.session_state.get("last_question", ""),
        )

# ==============================================================================
# EVALUATION RESULTS TAB
# ==============================================================================

with tab_eval:
    data = load_eval_results()

    if data is None:
        st.error(
            "`evaluation/results.json` not found. "
            "Run `python -m evaluation.evaluate` to generate it."
        )
        st.stop()

    # Determine which strategies are present
    strategies = active_strategies(data)
    if not strategies:
        st.error("No recognized strategies found in results.json.")
        st.stop()

    # ── Results summary table  ─────────────────────────────────────────────────

    st.subheader("Evaluation Results Summary")
    st.caption(
        "Replicates the `_print_table` output from `evaluation/evaluate.py`. "
        "Rows are grouped by temporality; columns are LLM-as-judge metrics. "
        "Green gradient highlights the best-performing config per metric within each group."
    )

    _PRINT_TABLE_ORDER = ["Evergreen", "Version-sensitive", "Mixed", "Overall"]

    # Count questions per group from the reference strategy's entries
    ref_entries = data[strategies[0]]
    q_counts: dict[str, int] = {
        g: sum(1 for e in ref_entries if e["temporality"] == g)
        for g in ["Evergreen", "Version-sensitive", "Mixed"]
    }
    q_counts["Overall"] = len(ref_entries)

    # Build MultiIndex DataFrame: rows = (group_label, config), cols = metrics
    table_rows = []
    index_tuples = []

    for group in _PRINT_TABLE_ORDER:
        group_label = f"{group} ({q_counts[group]} Qs)"
        for s in strategies:
            if group == "Overall":
                agg = aggregate(data[s])
            else:
                agg = aggregate(data[s], group_by="temporality").get(
                    group, {m: 0.0 for m in METRICS}
                )
            table_rows.append({METRIC_LABELS[m]: round(agg.get(m, 0.0), 3) for m in METRICS})
            index_tuples.append((group_label, STRATEGY_LABELS[s]))

    df_table = pd.DataFrame(
        table_rows,
        index=pd.MultiIndex.from_tuples(index_tuples, names=["Group", "Configuration"]),
    )

    # Apply green gradient per metric column across all rows so the best
    # config is clearly visible regardless of which group it belongs to.
    styled = (
        df_table.style
        .background_gradient(cmap="YlGn", axis=0)
        .format("{:.3f}")
        .set_table_styles([
            # Bold top border at every group boundary (every N_strategies rows)
            {"selector": f"tr:nth-child({len(strategies)}n+1) td, tr:nth-child({len(strategies)}n+1) th",
             "props": [("border-top", "2px solid #555")]},
        ])
    )
    st.dataframe(styled, use_container_width=True)

    # ── Chart 1: Grouped bar ───────────────────────────────────────────────────

    st.subheader("Metric Breakdown by Category")

    all_cats = TEMPORALITIES + ["Overall"]
    fig_bar = make_subplots(
        rows=2, cols=2,
        subplot_titles=all_cats,
        shared_yaxes=True,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (row, col), cat in zip(positions, all_cats):
        for s in strategies:
            if cat == "Overall":
                scores = aggregate(data[s])
            else:
                scores = aggregate(data[s], group_by="temporality").get(cat, {m: 0.0 for m in METRICS})
            fig_bar.add_trace(
                go.Bar(
                    name=STRATEGY_LABELS[s],
                    x=[METRIC_LABELS[m] for m in METRICS],
                    y=[scores.get(m, 0.0) for m in METRICS],
                    marker_color=STRATEGY_COLORS[s],
                    opacity=0.88,
                    showlegend=(row == 1 and col == 1),
                    legendgroup=s,
                ),
                row=row, col=col,
            )

    fig_bar.update_layout(
        barmode="group",
        height=550,
        legend=dict(orientation="h", y=-0.12),
        title_text="Metric Breakdown by Category",
        yaxis=dict(range=[0, 1.15]),
        yaxis2=dict(range=[0, 1.15]),
        yaxis3=dict(range=[0, 1.15]),
        yaxis4=dict(range=[0, 1.15]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Chart 2: Scatter precision vs recall ──────────────────────────────────

    st.subheader("Context Precision vs. Recall Trade-off")

    SHAPE_MAP = {
        "Evergreen": "circle",
        "Version-sensitive": "square",
        "Mixed": "triangle-up",
    }

    fig_scatter = go.Figure()
    for s in strategies:
        by_temp = aggregate(data[s], group_by="temporality")
        by_temp["Overall"] = aggregate(data[s])

        for cat, scores in by_temp.items():
            prec = scores.get("context_precision", 0.0)
            rec = scores.get("context_recall", 0.0)
            is_overall = cat == "Overall"
            fig_scatter.add_trace(go.Scatter(
                x=[prec], y=[rec],
                mode="markers+text",
                name=f"{STRATEGY_LABELS[s]} ({cat})",
                marker=dict(
                    symbol=SHAPE_MAP.get(cat, "star"),
                    size=18 if is_overall else 12,
                    color=STRATEGY_COLORS[s],
                    line=dict(color="black", width=2) if is_overall else dict(color="white", width=1),
                    opacity=0.9,
                ),
                text=[f"{STRATEGY_LABELS[s][:3]}·{cat[:3]}"],
                textposition="top right",
                textfont=dict(size=8),
            ))

    fig_scatter.add_shape(
        type="line", x0=0.2, y0=0.2, x1=1.0, y1=1.0,
        line=dict(color="gray", dash="dash", width=1),
    )
    fig_scatter.update_layout(
        xaxis_title="Context Precision",
        yaxis_title="Context Recall",
        xaxis=dict(range=[0.4, 1.05]),
        yaxis=dict(range=[0.0, 1.05]),
        height=500,
        legend=dict(font=dict(size=9)),
        title="Context Precision vs. Recall",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Chart 3: Per-question heatmap ─────────────────────────────────────────

    st.subheader("Per-Question Heatmap")

    import numpy as np

    # Build question order from first available strategy
    ref_strategy = strategies[0]
    sorted_entries = sort_questions(data[ref_strategy])
    q_ids = [e["id"] for e in sorted_entries]
    q_temps = [e["temporality"] for e in sorted_entries]
    id_to_row = {qid: i for i, qid in enumerate(q_ids)}

    metric_tabs = st.tabs([METRIC_LABELS[m] for m in METRICS])
    for tab, metric in zip(metric_tabs, METRICS):
        with tab:
            mat = np.full((len(q_ids), len(strategies)), np.nan)
            for col_i, s in enumerate(strategies):
                for e in data[s]:
                    row_i = id_to_row.get(e["id"])
                    if row_i is not None:
                        v = e["metrics"].get(metric)
                        if v is not None:
                            mat[row_i, col_i] = v

            text_vals = [
                [f"{mat[r, c]:.2f}" if not np.isnan(mat[r, c]) else ""
                 for c in range(len(strategies))]
                for r in range(len(q_ids))
            ]

            fig_hm = go.Figure(go.Heatmap(
                z=mat,
                x=[STRATEGY_LABELS[s] for s in strategies],
                y=q_ids,
                text=text_vals,
                texttemplate="%{text}",
                textfont=dict(size=9),
                colorscale="RdYlGn",
                zmin=0, zmax=1,
                colorbar=dict(title="Score"),
            ))

            # Horizontal lines at temporality group boundaries
            prev_temp = None
            for i, t in enumerate(q_temps):
                if t != prev_temp and i > 0:
                    fig_hm.add_hline(y=i - 0.5, line_color="black", line_width=2)
                prev_temp = t

            fig_hm.update_layout(
                height=max(450, len(q_ids) * 18),
                yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
                xaxis=dict(tickfont=dict(size=10)),
                title=f"Per-Question: {METRIC_LABELS[metric]}",
                margin=dict(l=60, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

    # ── Per-question drill-down ────────────────────────────────────────────────

    st.subheader("Per-Question Drill-Down")

    all_q_ids = [e["id"] for e in sorted_entries]
    selected_qid = st.selectbox("Select question", all_q_ids, key="eval_q_select")

    if selected_qid:
        base_entry = next((e for e in data[ref_strategy] if e["id"] == selected_qid), None)
        if base_entry:
            st.markdown(f"**Question:** {base_entry['question']}")
            st.markdown(f"**Expected:** {base_entry.get('expected_answer', 'N/A')}")
            st.markdown(
                f"**Temporality:** {base_entry['temporality']} | "
                f"**Category:** {base_entry['category']}"
            )

        st.markdown("---")
        cols = st.columns(len(strategies))
        for col, s in zip(cols, strategies):
            entry = next((e for e in data[s] if e["id"] == selected_qid), None)
            with col:
                st.markdown(f"**{STRATEGY_LABELS[s]}**")
                if entry:
                    for m in METRICS:
                        v = entry["metrics"].get(m, 0.0)
                        badge_color = "green" if v >= 0.8 else ("orange" if v >= 0.5 else "red")
                        st.markdown(f":{badge_color}[{METRIC_LABELS[m]}: **{v:.3f}**]")
                    with st.expander("Generated answer"):
                        st.write(entry.get("generated_answer", "N/A"))
                    clf = entry.get("classification") or {}
                    if clf:
                        with st.expander("Classification"):
                            st.json(clf)
                else:
                    st.caption("No data for this strategy.")
