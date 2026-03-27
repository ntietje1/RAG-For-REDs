"""Chunking strategies for splitting documents into retrieval units."""

import copy

from processing.loader import Document


def chunk_fixed_size(doc: Document, chunk_size: int = 512, overlap: int = 64) -> list[Document]:
    """Split a document into fixed-size character chunks with overlap.

    Each chunk inherits all metadata from the parent document and gets an
    updated doc_id suffixed with its chunk index for stable deduplication.
    Text shorter than chunk_size is returned as a single chunk unchanged.
    """
    text = doc.text
    if len(text) <= chunk_size:
        return [doc]

    base_id: str = doc.metadata.get("doc_id", "")
    # Strip any trailing chunk index from a prior splitting pass so IDs don't
    # accumulate nested suffixes when the function is called more than once.
    if base_id and base_id.rsplit("_", 1)[-1].isdigit():
        base_id = base_id.rsplit("_", 1)[0]

    chunks: list[Document] = []
    start = 0
    chunk_idx = 0
    step = max(chunk_size - overlap, 1)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        chunk_meta = copy.deepcopy(doc.metadata)
        chunk_meta["doc_id"] = f"{base_id}_{chunk_idx}" if base_id else str(chunk_idx)
        chunk_meta["chunk_index"] = chunk_idx
        chunk_meta["chunk_start_char"] = start

        chunks.append(
            Document(
                text=chunk_text,
                source=doc.source,
                url=doc.url,
                date=doc.date,
                patch_version=doc.patch_version,
                content_type=doc.content_type,
                metadata=chunk_meta,
            )
        )

        if end == len(text):
            break
        start += step
        chunk_idx += 1

    return chunks

