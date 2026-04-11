"""Shared LLM generation logic for RAG pipelines using LangChain."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config.pipeline_config import GENERATION_MODEL

load_dotenv()

_BASE_URL = "https://openrouter.ai/api/v1"

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """Return the shared LangChain ChatOpenAI instance for generation."""
    global _llm
    if _llm is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set")
        _llm = ChatOpenAI(
            model=GENERATION_MODEL,
            openai_api_key=api_key,
            openai_api_base=_BASE_URL,
            temperature=0.0,
        )
    return _llm


_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("user",
     "You are a helpful assistant with expertise in League of Legends.\n"
     "{patch_line}"
     "Answer the question using ONLY the provided context. "
     "If the context does not contain enough information to answer, say so clearly.\n\n"
     "Context:\n{numbered_context}\n\n"
     "Question: {question}\n\n"
     "Answer:"),
])


def generate_answer(
    question: str, context_chunks: list[str], current_patch: str | None = None
) -> str:
    """Generate an answer using an LLM given a question and retrieved context."""
    llm = get_llm()

    numbered_context = "\n\n".join(
        f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )
    patch_line = f"The current patch is {current_patch}.\n" if current_patch else ""

    chain = _GENERATION_PROMPT | llm
    result = chain.invoke({
        "question": question,
        "numbered_context": numbered_context,
        "patch_line": patch_line,
    })
    return result.content.strip()
