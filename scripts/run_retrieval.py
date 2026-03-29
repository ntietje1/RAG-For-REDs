"""Query a RAG pipeline interactively or in batch mode."""

import argparse

from config.pipeline_config import TOP_K
from indexing.store import VectorStore
from retrieval.baseline import BaselineRAG
from retrieval.enhanced import EnhancedRAG

PIPELINE_MODES = {
    "baseline":  {"use_temporal": False, "use_authority": False},
    "temporal":  {"use_temporal": True,  "use_authority": False},
    "authority": {"use_temporal": False, "use_authority": True},
    "full":      {"use_temporal": True,  "use_authority": True},
}


def _print_result(result: dict) -> None:
    classification = result.get("classification")
    if classification:
        print(f"\nTemporal scope: {classification['temporal_scope']}")
        print(f"Authority weights: {classification['authority_weights']}")
        alt = classification.get("alternate_queries", [])
        if alt:
            print(f"Alternate queries: {alt}")

    print(f"\nAnswer:\n{result['answer']}\n")
    print("Sources:")
    for i, src in enumerate(result["sources"], 1):
        score = src.get("score", 0.0)
        adj = src.get("adjusted_score")
        score_str = f"cosine: {score:.3f}"
        if adj is not None:
            score_str += f", adjusted: {adj:.3f}"
        print(f"  [{i}] {src.get('source', '')} — {src.get('url', '')} ({score_str})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument(
        "--pipeline",
        choices=list(PIPELINE_MODES),
        required=True,
        help="Which retrieval pipeline to use",
    )
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
    args = parser.parse_args()

    store = VectorStore()
    mode = PIPELINE_MODES[args.pipeline]

    if args.pipeline == "baseline":
        pipeline = BaselineRAG(store=store, top_k=args.top_k)
    else:
        pipeline = EnhancedRAG(
            store=store,
            use_temporal=mode["use_temporal"],
            use_authority=mode["use_authority"],
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
