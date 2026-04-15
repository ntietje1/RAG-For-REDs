"""Embedding generation via OpenRouter."""

from config.client import get_embeddings


def embed_documents(documents, batch_size: int = 100) -> list[list[float]]:
    """Generate embeddings for a list of documents in batches.

    Accepts either Document dataclass instances (with .text) or raw strings.
    Returns embeddings in the same order as the input documents.
    """
    embeddings_model = get_embeddings()
    texts = [doc.text if hasattr(doc, "text") else str(doc) for doc in documents]

    # Batching the embeddings to improve performance.
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = embeddings_model.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Generate an embedding for a single query string."""
    return get_embeddings().embed_query(query)
