"""Chunking strategies for splitting documents into retrieval units."""

import copy
import re
from processing.loader import Document

# split after any ".!?" plus " " (attempt to split after sentences)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_sentences(
    doc: Document,
    chunk_size: int = 1000,
    overlap: int = 128,
    context_prefix: str = "",
) -> list[Document]:
    text = doc.text.strip()
    if not text:
        return [doc]

    prefix = f"{context_prefix.strip()}\n" if context_prefix.strip() else ""
    effective_size = chunk_size - len(prefix)
    if effective_size <= 0:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be greater than prefix length ({len(prefix)})"
        )

    if len(text) <= effective_size:
        doc = copy.copy(doc)
        doc.text = prefix + text if prefix else text
        doc.metadata = copy.deepcopy(doc.metadata)
        return [doc]

    # split into sentences, then break any large sentences (over chunk size)
    # into hard character-limit pieces on newlines.
    raw_sentences = _SENTENCE_RE.split(text)
    sentences: list[str] = []
    for s in raw_sentences:
        if len(s) <= effective_size:
            sentences.append(s)
        else:
            # break on newlines first, then hard-cut as a last resort.
            for part in s.split("\n"):
                while len(part) > effective_size:
                    sentences.append(part[:effective_size])
                    part = part[effective_size:]
                if part:
                    sentences.append(part)

    chunks: list[Document] = []
    chunk_idx = 0
    i = 0

    while i < len(sentences):
        current: list[str] = []
        current_len = 0

        while i < len(sentences):
            s = sentences[i]
            added_len = len(s) + (1 if current else 0)  # space between sentences
            if current and current_len + added_len > effective_size:
                break
            current.append(s)
            current_len += added_len
            i += 1

        chunk_text = prefix + " ".join(current)
        char_offset = text.find(current[0])

        chunk_meta = copy.deepcopy(doc.metadata)
        chunk_meta["chunk_index"] = chunk_idx
        chunk_meta["chunk_start_char"] = char_offset

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
        chunk_idx += 1

        # back up for overlap: rewind sentences that fit within the overlap budget.
        if i < len(sentences) and len(current) > 1:
            overlap_len = 0
            rewind = 0
            for j in range(len(current) - 1, 0, -1):
                overlap_len += len(current[j]) + 1
                if overlap_len > overlap:
                    break
                rewind += 1
            if rewind:
                i -= rewind

    return chunks


def chunk_fixed_size(doc: Document, chunk_size: int = 512, overlap: int = 64) -> list[Document]:
    """Split a document into fixed-size character chunks with overlap.

    Each chunk inherits all metadata from the parent document. The caller is
    responsible for assigning unique doc_ids to the returned chunks.
    Text shorter than chunk_size is returned as a single chunk unchanged.
    """
    text = doc.text
    if len(text) <= chunk_size:
        return [doc]

    chunks: list[Document] = []
    start = 0
    chunk_idx = 0
    step = max(chunk_size - overlap, 1)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        chunk_meta = copy.deepcopy(doc.metadata)
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
