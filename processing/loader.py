"""Load raw scraped JSON files into a unified Document format."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from processing.cleaner import (
    clean_reddit,
    clean_wiki,
    is_redirect,
    serialize_stats_champion,
)


@dataclass
class Document:
    """Unified document representation across all data sources."""

    text: str
    source: str
    doc_id: str = ""
    url: str = ""
    date: str = ""
    patch_version: str = ""
    content_type: str = ""
    metadata: dict = field(default_factory=dict)


# per-source handlers ────────────────────────────────────────────────────────

def _url_hash(url: str) -> str:
    """Generate a short stable hash from a URL to use in unique IDs."""
    return hashlib.md5(url.encode()).hexdigest()[:8]


def _load_patch_notes(raw: dict) -> list[Document]:
    """Parse Riot patch notes into documents, splitting by section.

    Each section receives a unique doc_id for stable deduplication in the vector store.
    """
    meta = raw.get("metadata", {})
    content = raw.get("content", {})
    url = meta.get("url", "")
    date = meta.get("date") or ""
    patch_version = meta.get("patch_version") or ""

    sections: list[dict] = content.get("sections") or []
    if not sections:
        return []

    docs: list[Document] = []
    for i, section in enumerate(sections):
        heading = section.get("heading", "").strip()
        body = section.get("content", "").strip()
        if not body:
            continue

        text = f"{heading}\n{body}" if heading else body
        docs.append(
            Document(
                text=text,
                source="riot_patch_notes",
                doc_id=f"riot_patch_notes_{_url_hash(url)}_{i}",
                url=url,
                date=date,
                patch_version=patch_version,
                content_type="patch_notes",
                metadata={
                    "heading": heading,
                    "section_index": i,
                },
            )
        )
    return docs


def _load_wiki(raw: dict) -> list[Document]:
    meta = raw.get("metadata", {})
    content = raw.get("content", {})
    url = meta.get("url", "")
    raw_text: str = content.get("raw_text") or ""

    if is_redirect(raw_text):
        return []

    text = clean_wiki(raw_text)
    if len(text) < 50:
        return []

    return [
        Document(
            text=text,
            source="wiki",
            doc_id=f"wiki_{_url_hash(url)}_0",
            url=url,
            date=meta.get("date") or "",
            patch_version=meta.get("patch_version") or "",
            content_type="wiki_page",
            metadata={
                "title": content.get("title", ""),
                "categories": content.get("categories", []),
            },
        )
    ]


def _load_reddit(raw: dict) -> list[Document]:
    meta = raw.get("metadata", {})
    content = raw.get("content", {})
    url = meta.get("url", "")

    text = clean_reddit(content)
    if not text.strip():
        return []

    return [
        Document(
            text=text,
            source="reddit",
            doc_id=f"reddit_{_url_hash(url)}_0",
            url=url,
            date=meta.get("date") or "",
            patch_version=meta.get("patch_version") or "",
            content_type="reddit_post",
            metadata={
                "title": content.get("title", ""),
                "score": content.get("score", 0),
                "num_comments": content.get("num_comments", 0),
                "link_flair_text": content.get("link_flair_text", ""),
            },
        )
    ]


def _load_stats(raw: dict) -> list[Document]:
    meta = raw.get("metadata", {})
    content = raw.get("content", {})
    url = meta.get("url", "")
    patch_version: str = content.get("patch_version") or meta.get("patch_version") or ""
    champions: list[dict] = content.get("champions") or []

    docs: list[Document] = []
    for i, champ in enumerate(champions):
        text = serialize_stats_champion(champ, patch_version)
        docs.append(
            Document(
                text=text,
                source="lolalytics",
                doc_id=f"lolalytics_{_url_hash(url)}_{i}",
                url=url,
                date=meta.get("date") or "",
                patch_version=patch_version,
                content_type="champion_stats",
                metadata={
                    "champion_name": champ.get("name", ""),
                    "tier": champ.get("tier", ""),
                    "lane": champ.get("lane", ""),
                    "rank": champ.get("rank"),
                },
            )
        )
    return docs


_SOURCE_HANDLERS = {
    "patch_notes": _load_patch_notes,
    "wiki": _load_wiki,
    "reddit": _load_reddit,
    "stats": _load_stats,
}

# public API ─────────────────────────────────────────────────────────────────

def load_source(source_dir: Path) -> list[Document]:
    """Load all JSON files from a source directory into Documents.

    The directory name determines which per-source handler is used.
    Files that are empty, malformed, or produce no usable content
    (e.g. wiki redirect pages) are skipped.
    """
    handler = _SOURCE_HANDLERS.get(source_dir.name)
    if handler is None:
        raise ValueError(
            f"Unknown source directory '{source_dir.name}'. "
            f"Expected one of: {list(_SOURCE_HANDLERS)}"
        )

    docs: list[Document] = []
    for json_file in sorted(source_dir.glob("*.json")):
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
            docs.extend(handler(raw))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return docs


def load_all_sources(raw_dir: Path) -> list[Document]:
    """Load documents from all known source directories under *raw_dir*."""
    docs: list[Document] = []
    for name in _SOURCE_HANDLERS:
        source_dir = raw_dir / name
        if source_dir.is_dir():
            docs.extend(load_source(source_dir))
    return docs
