"""Vector store backed by Qdrant in local on-disk mode."""

import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from processing.loader import Document

_COLLECTION = "chunks"
_VECTOR_SIZE = 1536  # text-embedding-3-small output dimension



class VectorStore:
    """Qdrant-backed vector store using local on-disk persistence."""

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self.store_dir))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if _COLLECTION not in existing:
            self._client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
            )

    @staticmethod
    def _doc_to_payload(doc: Document) -> dict:
        return {
            "doc_id": doc.doc_id,
            "text": doc.text,
            "source": doc.source,
            "url": doc.url,
            "date": doc.date,
            "patch_version": doc.patch_version,
            "content_type": doc.content_type,
            **{f"meta_{k}": v for k, v in (doc.metadata or {}).items()},
        }

    # public API ───────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        """Upsert documents and their embeddings into the store."""
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")

        points: list[PointStruct] = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.doc_id)),     # Stable, collision-resistant IDs.
                vector=vector,
                payload=self._doc_to_payload(doc),
            )
            for doc, vector in zip(documents, embeddings)
        ]

        self._client.upsert(collection_name=_COLLECTION, points=points)

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        """Return the top-k most similar documents (no filter)."""
        return self._search(embedding, top_k, query_filter=None)

    def query_with_filter(
        self,
        embedding: list[float],
        filters: dict[str, str],
        top_k: int = 5,
    ) -> list[dict]:
        """Return the top-k most similar documents matching all payload filters.

        ``filters`` is a flat dict of ``{payload_field: exact_value}`` pairs.
        All conditions are combined with AND logic.
        """
        conditions = [
            FieldCondition(key=field, match=MatchValue(value=value))
            for field, value in filters.items()
        ]
        query_filter = Filter(must=conditions) if conditions else None
        return self._search(embedding, top_k, query_filter=query_filter)

    def _search(
        self,
        embedding: list[float],
        top_k: int,
        query_filter: Filter | None,
    ) -> list[dict]:
        response = self._client.query_points(
            collection_name=_COLLECTION,
            query=embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            {
                "score": hit.score,
                **hit.payload,
            }
            for hit in response.points
        ]

    def clear(self) -> None:
        """Delete and recreate the collection, removing all documents."""
        self._client.delete_collection(_COLLECTION)
        self._ensure_collection()
