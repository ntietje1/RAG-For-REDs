"""Query a RAG pipeline interactively or in batch mode."""

import argparse
import sys
from pathlib import Path

from config.pipeline_config import TOP_K, VECTOR_STORE_DIR
from indexing.store import VectorStore
from retrieval.baseline import BaselineRAG


def _print_result(result: dict) -> None:
    print(f"\nAnswer:\n{result['answer']}\n")
    print("Sources:")
    for i, src in enumerate(result["sources"], 1):
        score = src.get("score", 0.0)
        print(f"  [{i}] {src.get('source', '')} — {src.get('url', '')} (score: {score:.3f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument(
        "--pipeline",
        choices=["baseline", "temporal"],
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
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=VECTOR_STORE_DIR,
        help="Qdrant on-disk store directory (default: data/indices)",
    )
    args = parser.parse_args()

    if args.pipeline == "temporal":
        print("Error: temporal pipeline is not yet implemented.")
        sys.exit(1)

    store = VectorStore(args.store_dir)
    pipeline = BaselineRAG(store=store, top_k=args.top_k)

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
