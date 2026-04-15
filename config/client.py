"""Shared LangChain clients for all pipeline modules (OpenRouter-compatible)."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.pipeline_config import EMBEDDING_MODEL, EVAL_MODEL, GENERATION_MODEL

load_dotenv()

_BASE_URL = "https://openrouter.ai/api/v1"

_generation_llm: ChatOpenAI | None = None
_eval_llm: ChatOpenAI | None = None
_embeddings: OpenAIEmbeddings | None = None


def _get_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set")
    return api_key


def get_generation_llm() -> ChatOpenAI:
    """Return the shared generation LLM (Gemini Flash via OpenRouter)."""
    global _generation_llm
    if _generation_llm is None:
        _generation_llm = ChatOpenAI(
            model=GENERATION_MODEL,
            openai_api_key=_get_api_key(),
            openai_api_base=_BASE_URL,
            temperature=0.0,
        )
    return _generation_llm


def get_eval_llm() -> ChatOpenAI:
    """Return the shared evaluation judge LLM (GPT-4o-mini via OpenRouter)."""
    global _eval_llm
    if _eval_llm is None:
        _eval_llm = ChatOpenAI(
            model=EVAL_MODEL,
            openai_api_key=_get_api_key(),
            openai_api_base=_BASE_URL,
            temperature=0.0,
        )
    return _eval_llm


def get_embeddings() -> OpenAIEmbeddings:
    """Return the shared embeddings model (text-embedding-3-small via OpenRouter)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=_get_api_key(),
            openai_api_base=_BASE_URL,
        )
    return _embeddings
