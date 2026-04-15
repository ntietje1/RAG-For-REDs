"""Ablation evaluation orchestrator for the EnhancedRAG pipeline.

Runs a set of evaluation questions through multiple pipeline configurations
and computes LLM-as-judge metrics, producing per-question results and
aggregate comparison tables broken down by temporality category.

Usage:
    python -m evaluation.evaluate                         # full run
    python -m evaluation.evaluate --dry-run               # 1 question per config
    python -m evaluation.evaluate --configs baseline full  # subset of configs
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from evaluation.metrics import score_all
from evaluation.questions import load_questions
from indexing.store import VectorStore
from retrieval.enhanced import EnhancedRAG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ablation configurations — the 4 experimental conditions
# ---------------------------------------------------------------------------
# use_expansion=True and use_cross_encoder=True are held constant across all
# configs so the ablation isolates the two variables under study (temporal
# scope and source authority).  The cross-encoder is always enabled because
# it is part of the intended production pipeline — evaluating without it
# would measure temporal/authority interaction with raw cosine scores, which
# is not representative of real usage.

CONFIGS: dict[str, dict] = {
    "baseline": {
        "use_temporal": False,
        "use_authority": False,
        "use_expansion": True,
        "use_cross_encoder": True,
        "discrete_weights": False,
    },
    "temporal_only": {
        "use_temporal": True,
        "use_authority": False,
        "use_expansion": True,
        "use_cross_encoder": True,
        "discrete_weights": True,
    },
    "authority_only": {
        "use_temporal": False,
        "use_authority": True,
        "use_expansion": True,
        "use_cross_encoder": True,
        "discrete_weights": True,
    },
    "full": {
        "use_temporal": True,
        "use_authority": True,
        "use_expansion": True,
        "use_cross_encoder": True,
        "discrete_weights": True,
    },
}

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    questions: list[dict],
    store: VectorStore,
    config_names: list[str],
    dry_run: bool = False,
) -> dict:
    """Run all questions through each config and compute metrics.

    Returns a nested dict: results[config_name] = list of per-question dicts.
    """
    results: dict[str, list[dict]] = {}

    for config_name in config_names:
        cfg = CONFIGS[config_name]
        logger.info("=== Config: %s ===", config_name)

        pipeline = EnhancedRAG(store=store, **cfg)

        question_subset = questions[:1] if dry_run else questions
        config_results: list[dict] = []

        for idx, q in enumerate(question_subset, 1):
            qid = q["id"]
            question_text = q["question"]
            expected = q["expected_answer"]
            temporality = q["temporality"]

            logger.info(
                "  Config %s: %d/%d — [%s] %s",
                config_name, idx, len(question_subset), qid, question_text[:60],
            )

            # Run the RAG pipeline
            pipeline_result = pipeline.query(question_text)
            answer = pipeline_result["answer"]
            chunks = [r["text"] for r in pipeline_result["retrieved_chunks"]]

            # Compute metrics using GPT-4o-mini judge
            metrics = score_all(
                question=question_text,
                answer=answer,
                expected_answer=expected,
                context_chunks=chunks,
            )

            config_results.append({
                "id": qid,
                "question": question_text,
                "temporality": temporality,
                "category": q.get("category", ""),
                "expected_answer": expected,
                "generated_answer": answer,
                "classification": pipeline_result.get("classification"),
                "num_chunks": len(chunks),
                "metrics": metrics,
            })

        results[config_name] = config_results
        logger.info("  Config %s complete: %d questions evaluated", config_name, len(config_results))

    return results


# ---------------------------------------------------------------------------
# Reporting11           
# ---------------------------------------------------------------------------

def _print_table(results: dict, questions: list[dict]) -> None:
    """Print comparison tables: overall + broken down by temporality."""
    temporalities = ["Evergreen", "Version-sensitive", "Mixed"]

    def _avg(scores: list[float]) -> float:
        return sum(scores) / len(scores) if scores else 0.0

    # Build aggregated scores: agg[config][temporality][metric] = [scores]
    agg: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for config_name, config_results in results.items():
        for qr in config_results:
            temp = qr["temporality"]
            for metric in METRICS:
                agg[config_name][temp][metric].append(qr["metrics"][metric])
                agg[config_name]["OVERALL"][metric].append(qr["metrics"][metric])

    # Print
    col_width = 16
    header = f"{'':20s}" + "".join(f"{m:>{col_width}s}" for m in METRICS)
    separator = "-" * len(header)

    for group in temporalities + ["OVERALL"]:
        # Count questions in this group
        if group == "OVERALL":
            n = sum(len(cr) for cr in results.values()) // max(len(results), 1)
        else:
            n = sum(1 for q in questions if q["temporality"] == group)
            # Skip if no questions in this group after dry-run filtering
            if not any(agg[c][group][METRICS[0]] for c in results):
                continue

        print(f"\n{group} ({n} Qs)")
        print(header)
        print(separator)
        for config_name in results:
            scores_str = ""
            for metric in METRICS:
                vals = agg[config_name][group][metric]
                scores_str += f"{_avg(vals):>{col_width}.3f}"
            print(f"{config_name:20s}{scores_str}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    # Suppress HTTPX noisy logs
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Run ablation evaluation of the EnhancedRAG pipeline"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("evaluation/questions.json"),
        help="Path to evaluation questions JSON (default: evaluation/questions.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results.json"),
        help="Path to write raw results JSON (default: evaluation/results.json)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS.keys()),
        default=list(CONFIGS.keys()),
        help="Subset of configs to evaluate (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only 1 question per config to verify pipeline works",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    store = VectorStore()

    results = run_evaluation(
        questions=questions,
        store=store,
        config_names=args.configs,
        dry_run=args.dry_run,
    )

    # Save raw results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Raw results saved to %s", args.output)

    # Print comparison tables
    _print_table(results, questions)


if __name__ == "__main__":
    main()
