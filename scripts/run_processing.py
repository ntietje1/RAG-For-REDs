"""Process raw scraped data into chunked documents with metadata."""

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

from config.pipeline_config import CHUNK_OVERLAP, CHUNK_SIZE
from config.settings import DATA_DIR
from processing.chunker import chunk_sentences
from processing.loader import Document, stream_all_sources, stream_source


def _context_prefix(doc: Document) -> str:
    """Return a short context string to prepend to each chunk, based on source."""
    if doc.source == "riot_patch_notes":
        pv = doc.patch_version
        heading = doc.metadata.get("heading", "")
        return f"Patch {pv} — {heading}" if pv and heading else heading or ""
    if doc.source == "wiki":
        title = doc.metadata.get("title", "")
        return re.sub(r"<[^>]+>", "", title)
    if doc.source == "reddit":
        return doc.metadata.get("title", "")
    return ""


# Sources that need splitting. Stats (lolalytics) are one short
# sentence per champion and never exceed the chunk limit.
_SOURCES_NEEDING_CHUNKING = {"riot_patch_notes", "wiki", "reddit"}


def _chunk_and_write(docs, output: Path) -> tuple[int, int]:
    """Stream docs through chunking and write directly to disk.

    Returns (doc_count, chunk_count).
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    doc_count = 0
    chunk_count = 0

    with output.open("w", encoding="utf-8") as f:
        for doc in docs:
            doc_count += 1
            if doc_count % 500 == 0:
                print(f"processed {doc_count} documents")

            if doc.source in _SOURCES_NEEDING_CHUNKING:
                prefix = _context_prefix(doc)
                sub = chunk_sentences(doc, CHUNK_SIZE, CHUNK_OVERLAP, prefix)
                for i, chunk in enumerate(sub):
                    chunk.doc_id = f"{doc.doc_id}_c{i}"
                    f.write(json.dumps(dataclasses.asdict(chunk)) + "\n")
                chunk_count += len(sub)
            else:
                f.write(json.dumps(dataclasses.asdict(doc)) + "\n")
                chunk_count += 1

    return doc_count, chunk_count


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
        docs = stream_all_sources(raw_dir)
    else:
        source_dir = raw_dir / args.source
        docs = stream_source(source_dir)

    doc_count, chunk_count = _chunk_and_write(docs, args.output)
    print(f"Processed {doc_count} documents: {chunk_count} chunks")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
