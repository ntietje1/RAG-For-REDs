"""Shared LLM generation logic for RAG pipelines."""

from config.client import get_client
from config.pipeline_config import GENERATION_MODEL


def generate_answer(
    question: str, context_chunks: list[str], current_patch: str | None = None
) -> str:
    """Generate an answer using an LLM given a question and retrieved context."""
    client = get_client()

    numbered_context = "\n\n".join(
        f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )
    patch_line = f"The current patch is {current_patch}.\n" if current_patch else ""
    prompt = (
        "You are a helpful assistant with expertise in League of Legends.\n"
        f"{patch_line}"
        "Answer the question using ONLY the provided context. "
        "If the context does not contain enough information to answer, say so clearly.\n\n"
        f"Context:\n{numbered_context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()
