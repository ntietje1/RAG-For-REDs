"""LangGraph-based RAG pipeline with optional temporal decay, source authority, and cross-encoder re-ranking.

Replaces the procedural EnhancedRAG class with a state graph whose nodes
correspond to pipeline stages: classify → expand+embed → retrieve → rerank → generate.
"""

import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from config.pipeline_config import RERANK_CANDIDATE_K, TOP_K
from indexing.embedder import embed_query
from indexing.store import VectorStore
from retrieval.generator import generate_answer
from retrieval.reranker import build_patch_index, rerank

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    """State flowing through the LangGraph RAG pipeline."""
    question: str
    classification: dict
    candidates: list[dict]
    reranked: list[dict]
    context_chunks: list[str]
    answer: str
    sources: list[dict]


# ---------------------------------------------------------------------------
# EnhancedRAG — configurable graph builder
# ---------------------------------------------------------------------------

class EnhancedRAG:
    """RAG pipeline built as a LangGraph state graph.

    When none of the classifier-dependent features (temporal, authority,
    expansion) are enabled, the classifier node is skipped via a conditional edge.
    """

    def __init__(
        self,
        store: VectorStore,
        use_temporal: bool = False,
        use_authority: bool = False,
        use_cross_encoder: bool = False,
        use_expansion: bool = True,
        discrete_weights: bool = False,
        candidate_k: int = RERANK_CANDIDATE_K,
        final_k: int = TOP_K,
    ):
        self.store = store
        self.use_temporal = use_temporal
        self.use_authority = use_authority
        self.use_cross_encoder = use_cross_encoder
        self.use_expansion = use_expansion
        self.discrete_weights = discrete_weights
        self.candidate_k = candidate_k
        self.final_k = final_k

        versions = store.get_patch_versions()
        self.patch_index = build_patch_index(versions)
        self.current_patch = max(self.patch_index, key=self.patch_index.get)
        logger.info("Resolved current patch: %s", self.current_patch)

        self._graph = self._build_graph()

    # -- Node functions (closures over self) --------------------------------

    def _classify_node(self, state: PipelineState) -> dict:
        """LLM-based query classification for temporal scope and authority weights."""
        from retrieval.classifier import classify_query

        classification = classify_query(
            state["question"],
            self.current_patch,
            discrete_weights=self.discrete_weights,
        )
        logger.info("Classification: %s", classification)
        return {"classification": classification}

    def _skip_classify_node(self, state: PipelineState) -> dict:
        """No-op when classifier features are disabled."""
        return {"classification": {}}

    def _retrieve_node(self, state: PipelineState) -> dict:
        """Embed query (+ expansions) and retrieve candidates from vector store."""
        classification = state.get("classification", {})
        question = state["question"]

        alt = classification.get("alternate_queries", []) if self.use_expansion else []
        all_queries = [question] + alt

        seen: dict[str, dict] = {}
        for q in all_queries:
            embedding = embed_query(q)
            for hit in self.store.query(embedding, top_k=self.candidate_k):
                doc_id = hit.get("doc_id", id(hit))
                if doc_id not in seen or hit["score"] > seen[doc_id]["score"]:
                    seen[doc_id] = hit
        return {"candidates": list(seen.values())}

    def _rerank_node(self, state: PipelineState) -> dict:
        """Re-rank candidates using temporal decay, authority weights, and cross-encoder."""
        classification = state.get("classification", {})

        results = rerank(
            candidates=state["candidates"],
            patch_index=self.patch_index,
            current_patch=self.current_patch,
            temporal_scope=classification.get("temporal_scope") if self.use_temporal else None,
            target_patch=classification.get("target_patch") if self.use_temporal else None,
            authority_weights=classification.get("authority_weights") if self.use_authority else None,
            query=state["question"],
            use_cross_encoder=self.use_cross_encoder,
            final_k=self.final_k,
        )

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
            "reranked": results,
            "context_chunks": [r["text"] for r in results],
            "sources": sources,
        }

    def _generate_node(self, state: PipelineState) -> dict:
        """Generate the final answer from context chunks."""
        answer = generate_answer(
            state["question"],
            state["context_chunks"],
            self.current_patch,
        )
        return {"answer": answer}

    # -- Graph construction -------------------------------------------------

    def _needs_classifier(self) -> bool:
        return self.use_temporal or self.use_authority or self.use_expansion

    def _build_graph(self):
        """Construct and compile the LangGraph state graph."""
        graph = StateGraph(PipelineState)

        # Add nodes
        graph.add_node("classify", self._classify_node)
        graph.add_node("skip_classify", self._skip_classify_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("rerank", self._rerank_node)
        graph.add_node("generate", self._generate_node)

        # Entry: conditional edge to classify or skip
        if self._needs_classifier():
            graph.set_entry_point("classify")
            graph.add_edge("classify", "retrieve")
        else:
            graph.set_entry_point("skip_classify")
            graph.add_edge("skip_classify", "retrieve")

        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    # -- Public API ---------------------------------------------------------

    def query(self, question: str) -> dict:
        """Run the RAG pipeline and return answer, sources, and classification."""
        initial_state: PipelineState = {"question": question}
        final_state = self._graph.invoke(initial_state)

        return {
            "answer": final_state["answer"],
            "sources": final_state.get("sources", []),
            "retrieved_chunks": final_state.get("reranked", []),
            "classification": final_state.get("classification") or None,
        }
