"""Baseline RAG pipeline: embed query → retrieve top-k → generate."""

from indexing.embedder import embed_query
from indexing.store import VectorStore
from retrieval.generator import generate_answer


class BaselineRAG:
    """Standard RAG pipeline without temporal or authority awareness."""

    def __init__(self, store: VectorStore, top_k: int = 5):
        self.store = store
        self.top_k = top_k

    def query(self, question: str) -> dict:
        """Run the full RAG pipeline for a question.

        Returns dict with 'answer', 'sources', and 'retrieved_chunks'.
        """
        embedding = embed_query(question)
        results = self.store.query(embedding, top_k=self.top_k)

        context_chunks = [r["text"] for r in results]
        answer = generate_answer(question, context_chunks)

        sources = [
            {
                "url": r.get("url", ""),
                "source": r.get("source", ""),
                "score": r.get("score", 0.0),
            }
            for r in results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": results,
        }
