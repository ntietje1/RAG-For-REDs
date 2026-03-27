"""Embedding generation via OpenRouter."""

from config.client import get_client
from config.pipeline_config import EMBEDDING_MODEL
from processing.loader import Document


def embed_documents(documents: list[Document], batch_size: int = 100) -> list[list[float]]:
    """Generate embeddings for a list of documents in batches.

    Returns embeddings in the same order as the input documents.
    """
    client = get_client()
    texts = [doc.text for doc in documents]
    embeddings: list[list[float]] = []

    # Batching the embeddings to improve performance.
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        # Order the embeddings to match the input documents.
        batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        embeddings.extend(batch_embeddings)

    return embeddings


def embed_query(query: str) -> list[float]:
    """Generate an embedding for a single query string."""
    client = get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding
