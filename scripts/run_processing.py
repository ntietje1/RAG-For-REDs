"""Process raw scraped data into chunked documents with metadata."""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from config.pipeline_config import CHUNK_OVERLAP, CHUNK_SIZE
from config.settings import DATA_DIR
from processing.chunker import chunk_fixed_size
from processing.loader import Document, load_all_sources, load_source

# Sources that need fixed-size splitting. Stats (lolalytics) are one short
# sentence per champion and never exceed the chunk limit.
_SOURCES_NEEDING_CHUNKING = {"riot_patch_notes", "wiki", "reddit"}


def _chunk_documents(docs: list[Document]) -> list[Document]:
    """Apply per-source chunking strategies and assign stable chunk doc_ids."""
    chunks: list[Document] = []
    for doc in docs:
        if doc.source in _SOURCES_NEEDING_CHUNKING:
            sub = chunk_fixed_size(doc, CHUNK_SIZE, CHUNK_OVERLAP)
            for i, chunk in enumerate(sub):
                chunk.doc_id = f"{doc.doc_id}_c{i}"
            chunks.extend(sub)
        else:
            # lolalytics stats: one doc per champion.
            chunks.append(doc)
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Process and chunk raw scraped data")
    parser.add_argument(
        "--source",
        choices=["patch_notes", "wiki", "reddit", "stats", "all"],
        default="all",
        help="Which data source to process (default: all)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DATA_DIR / "raw",
        help="Directory containing raw scraped data (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "processed" / "chunks.jsonl",
        help="Output JSONL file (default: data/processed/chunks.jsonl)",
    )
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    if not raw_dir.exists():
        print(f"Error: raw data directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading source: {args.source}")
    if args.source == "all":
        docs = load_all_sources(raw_dir)
    else:
        source_dir = raw_dir / args.source
        docs = load_source(source_dir)

    print(f"Loaded {len(docs)} documents")
    chunks = _chunk_documents(docs)
    print(f"Produced {len(chunks)} chunks")

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(dataclasses.asdict(chunk)) + "\n")

    print(f"Saved {len(chunks)} chunks → {output}")


if __name__ == "__main__":
    main()
