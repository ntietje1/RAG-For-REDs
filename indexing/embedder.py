"""Embedding generation via LangChain OpenAI embeddings (OpenRouter-compatible)."""

import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from config.pipeline_config import EMBEDDING_MODEL

load_dotenv()

_BASE_URL = "https://openrouter.ai/api/v1"

_embeddings: OpenAIEmbeddings | None = None


def get_embeddings() -> OpenAIEmbeddings:
    """Return the shared LangChain embeddings instance, initializing on first call."""
    global _embeddings
    if _embeddings is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set")
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=api_key,
            openai_api_base=_BASE_URL,
        )
    return _embeddings


def embed_documents(documents, batch_size: int = 100) -> list[list[float]]:
    """Generate embeddings for a list of documents in batches.

    Accepts either Document dataclass instances (with .text) or raw strings.
    Returns embeddings in the same order as the input documents.
    """
    embeddings_model = get_embeddings()
    texts = [doc.text if hasattr(doc, "text") else str(doc) for doc in documents]

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = embeddings_model.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Generate an embedding for a single query string."""
    return get_embeddings().embed_query(query)
