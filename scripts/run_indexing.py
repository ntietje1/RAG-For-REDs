"""Build or update vector store indices from processed documents."""

import argparse
import json
import sys
from collections.abc import Generator
from pathlib import Path

from config.settings import DATA_DIR
from indexing.embedder import embed_documents
from indexing.store import VectorStore
from processing.loader import Document


def _stream_chunk_batches(
    path: Path, batch_size: int
) -> Generator[list[Document], None, None]:
    """Yield successive batches of Documents from a JSONL file.

    Reading and deserializing one line at a time means memory usage is
    proportional to batch_size, not the total corpus size.
    """
    batch: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(Document(**json.loads(line)))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a file without loading it into memory."""
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Build/update vector store indices")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index from scratch (default: incremental upsert)",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DATA_DIR / "processed" / "chunks.jsonl",
        help="Input JSONL file of processed chunks (default: data/processed/chunks.jsonl)",
    )
    parser.add_argument(
        "--index-batch-size",
        type=int,
        default=500,
        help=(
            "Number of chunks loaded into memory per iteration (default: 500). "
            "Controls peak memory usage: each iteration embeds this many chunks "
            "then immediately upserts them to Qdrant before moving on."
        ),
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=100,
        help=(
            "Number of chunks per embedding API request (default: 100). "
            "Each index-batch is split into ceil(index/embed) API calls. "
            "The OpenRouter text-embedding-3-small endpoint accepts up to 2048 "
            "inputs per request, but 100 is a safe default."
        ),
    )
    args = parser.parse_args()

    if not args.chunks.exists():
        print(f"Error: chunks file not found: {args.chunks}", file=sys.stderr)
        print("Run scripts/run_processing.py first.", file=sys.stderr)
        sys.exit(1)

    store = VectorStore()

    if args.rebuild:
        print("Clearing existing index...")
        store.clear()

    total = _count_lines(args.chunks)
    print(
        f"Indexing {total} chunks from {args.chunks} "
        f"(index-batch-size={args.index_batch_size}, "
        f"embed-batch-size={args.embed_batch_size})"
    )

    indexed = 0
    for batch in _stream_chunk_batches(args.chunks, args.index_batch_size):
        embeddings = embed_documents(batch, batch_size=args.embed_batch_size)
        store.add_documents(batch, embeddings)
        indexed += len(batch)
        print(f"  {indexed}/{total} chunks indexed", end="\r", flush=True)

    print(f"\nDone. Indexed {indexed} chunks into Qdrant.")


if __name__ == "__main__":
    main()
