"""Query a RAG pipeline interactively or in batch mode."""

import argparse

from config.pipeline_config import TOP_K
from indexing.store import VectorStore
from retrieval.enhanced import EnhancedRAG


def _print_result(result: dict) -> None:
    cl = result.get("classification") or {}
    header_parts = []
    if "temporal_scope" in cl:
        header_parts.append(f"Temporal scope: {cl['temporal_scope']}")
    if "authority_weights" in cl:
        header_parts.append(f"Authority weights: {cl['authority_weights']}")
    if "target_patch" in cl and cl["target_patch"]:
        header_parts.append(f"Target patch: {cl['target_patch']}")
    if "alternate_queries" in cl:
        header_parts.append(f"Alternate queries: {cl['alternate_queries']}")
    if "reasoning" in cl and cl["reasoning"]:
        header_parts.append(f"Reasoning: {cl['reasoning']}")
    if header_parts:
        print()
        print("\n".join(header_parts))

    print(f"\nAnswer:\n{result['answer']}\n")
    print("Sources:")
    for i, src in enumerate(result["sources"], 1):
        score = src.get("score", 0.0)
        adj = src.get("adjusted_score")
        score_str = f"cosine={score:.3f}"
        if adj is not None and adj != score:
            score_str += f"  adjusted={adj:.3f}"
        print(f"  [{i}] {src.get('source', '')} — {src.get('url', '')} ({score_str})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to run (omit for interactive mode)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K})",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Enable temporal decay (down-weight older patches)",
    )
    parser.add_argument(
        "--authority",
        action="store_true",
        help="Enable source-authority weighting",
    )
    parser.add_argument(
        "--cross-encoder",
        action="store_true",
        help="Enable cross-encoder re-ranking",
    )
    parser.add_argument(
        "--no-expansion",
        action="store_true",
        help="Disable query expansion (alternate phrasings)",
    )
    parser.add_argument(
        "--discrete-weights",
        action="store_true",
        help="Use discrete authority levels (low/medium/high) instead of continuous floats",
    )
    parser.add_argument(
        "--continuous-temporal",
        action="store_true",
        help="Pass LLM-derived temporal_sensitivity float directly as λ (default: discrete scope lookup)",
    )
    args = parser.parse_args()

    store = VectorStore()

    pipeline = EnhancedRAG(
        store=store,
        use_temporal=args.temporal,
        use_authority=args.authority,
        use_cross_encoder=args.cross_encoder,
        use_expansion=not args.no_expansion,
        discrete_weights=args.discrete_weights,
        continuous_temporal=args.continuous_temporal,
        final_k=args.top_k,
    )

    if args.query:
        _print_result(pipeline.query(args.query))
    else:
        print("Interactive mode — enter a question or type 'quit' to exit.\n")
        while True:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question or question.lower() in {"quit", "exit", "q"}:
                break
            _print_result(pipeline.query(question))


if __name__ == "__main__":
    main()
