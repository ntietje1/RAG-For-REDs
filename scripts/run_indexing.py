"""Build or update vector store indices from processed documents."""

import argparse
import json
import sys
from pathlib import Path

from config.pipeline_config import VECTOR_STORE_DIR
from config.settings import DATA_DIR
from indexing.embedder import embed_documents
from indexing.store import VectorStore
from processing.loader import Document


def _load_chunks(path: Path) -> list[Document]:
    """Deserialize Documents from the JSONL file produced by run_processing.py."""
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(Document(**json.loads(line)))
    return docs


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
        "--store-dir",
        type=Path,
        default=VECTOR_STORE_DIR,
        help="Qdrant on-disk store directory (default: data/indices)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Embedding API batch size (default: 100)",
    )
    args = parser.parse_args()

    if not args.chunks.exists():
        print(f"Error: chunks file not found: {args.chunks}", file=sys.stderr)
        print("Run scripts/run_processing.py first.", file=sys.stderr)
        sys.exit(1)

    store = VectorStore(args.store_dir)

    if args.rebuild:
        print("Clearing existing index...")
        store.clear()

    print(f"Loading chunks from {args.chunks}")
    docs = _load_chunks(args.chunks)
    print(f"Loaded {len(docs)} chunks")

    print(f"Embedding {len(docs)} chunks (batch_size={args.batch_size})...")
    embeddings = embed_documents(docs, batch_size=args.batch_size)

    print("Upserting into vector store...")
    store.add_documents(docs, embeddings)
    print(f"Indexed {len(docs)} chunks to {args.store_dir}")


if __name__ == "__main__":
    main()
