"""Enhanced RAG pipeline with temporal scope classification and source authority re-ranking."""

import logging

from config.pipeline_config import RERANK_CANDIDATE_K, TOP_K
from indexing.embedder import embed_query
from indexing.store import VectorStore
from retrieval.classifier import classify_query
from retrieval.generator import generate_answer
from retrieval.reranker import build_patch_index, rerank

logger = logging.getLogger(__name__)


class EnhancedRAG:
    """RAG pipeline with optional temporal decay and/or authority re-ranking."""

    def __init__(
        self,
        store: VectorStore,
        use_temporal: bool = True,
        use_authority: bool = True,
        candidate_k: int = RERANK_CANDIDATE_K,
        final_k: int = TOP_K,
    ):
        self.store = store
        self.use_temporal = use_temporal
        self.use_authority = use_authority
        self.candidate_k = candidate_k
        self.final_k = final_k

        versions = store.get_patch_versions()
        self.patch_index = build_patch_index(versions)
        self.current_patch = max(self.patch_index, key=self.patch_index.get)
        logger.info("Resolved current patch: %s", self.current_patch)

    def query(self, question: str) -> dict:
        """Run the enhanced RAG pipeline and return answer, sources, and classification."""
        classification = classify_query(question, self.current_patch)
        logger.info("Classification: %s", classification)

        # embed the original query + alternate phrasings, retrieve for each and merge
        all_queries = [question] + classification.get("alternate_queries", [])
        seen: dict[str, dict] = {}  # doc_id -> best so far
        for q in all_queries:
            embedding = embed_query(q)
            for hit in self.store.query(embedding, top_k=self.candidate_k):
                doc_id = hit.get("doc_id", id(hit))
                if doc_id not in seen or hit["score"] > seen[doc_id]["score"]:
                    seen[doc_id] = hit
        candidates = list(seen.values())

        results = rerank(
            candidates=candidates,
            patch_index=self.patch_index,
            current_patch=self.current_patch,
            temporal_scope=classification["temporal_scope"] if self.use_temporal else None,
            authority_weights=classification["authority_weights"] if self.use_authority else None,
            final_k=self.final_k,
        )

        context_chunks = [r["text"] for r in results]
        answer = generate_answer(question, context_chunks, self.current_patch)

        sources = [
            {
                "url": r.get("url", ""),
                "source": r.get("source", ""),
                "score": r.get("score", 0.0),
                "adjusted_score": r.get("adjusted_score"),
            }
            for r in results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": results,
            "classification": classification,
        }
