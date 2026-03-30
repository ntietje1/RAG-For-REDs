"""RAG pipeline with optional temporal decay, source authority, and cross-encoder re-ranking."""

import logging

from config.pipeline_config import RERANK_CANDIDATE_K, TOP_K
from indexing.embedder import embed_query
from indexing.store import VectorStore
from retrieval.generator import generate_answer
from retrieval.reranker import build_patch_index, rerank

logger = logging.getLogger(__name__)


class EnhancedRAG:
    """RAG pipeline with optional query expansion, temporal, authority, and cross-encoder features.

    When none of the classifier-dependent features (temporal, authority,
    expansion) are enabled, the LLM classifier call is skipped entirely.
    """

    def __init__(
        self,
        store: VectorStore,
        use_temporal: bool = False,
        use_authority: bool = False,
        use_cross_encoder: bool = False,
        use_expansion: bool = True,
        candidate_k: int = RERANK_CANDIDATE_K,
        final_k: int = TOP_K,
    ):
        self.store = store
        self.use_temporal = use_temporal
        self.use_authority = use_authority
        self.use_cross_encoder = use_cross_encoder
        self.use_expansion = use_expansion
        self.candidate_k = candidate_k
        self.final_k = final_k

        versions = store.get_patch_versions()
        self.patch_index = build_patch_index(versions)
        self.current_patch = max(self.patch_index, key=self.patch_index.get)
        logger.info("Resolved current patch: %s", self.current_patch)

    def query(self, question: str) -> dict:
        """Run the RAG pipeline and return answer, sources, and classification."""
        # only call the LLM classifier when at least one feature needs it
        if (self.use_temporal or self.use_authority or self.use_expansion):
            from retrieval.classifier import classify_query

            classification = classify_query(question, self.current_patch)
            logger.info("Classification: %s", classification)
        else:
            classification = {}

        # embed the original query + alternate phrasings, retrieve for each and merge
        alt = classification.get("alternate_queries", []) if self.use_expansion else []
        all_queries = [question] + alt
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
            temporal_scope=classification.get("temporal_scope") if self.use_temporal else None,
            authority_weights=classification.get("authority_weights") if self.use_authority else None,
            query=question,
            use_cross_encoder=self.use_cross_encoder,
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
            "classification": classification or None,
        }
