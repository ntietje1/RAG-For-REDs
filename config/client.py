"""Shared OpenRouter client for all pipeline modules"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_BASE_URL = "https://openrouter.ai/api/v1"
_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return the shared OpenRouter client, initializing it on first call."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set")
        _client = OpenAI(base_url=_BASE_URL, api_key=api_key)
    return _client
