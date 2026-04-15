"""Shared LLM generation logic for RAG pipelines using LangChain."""

from langchain_core.prompts import ChatPromptTemplate

from config.client import get_generation_llm

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
    numbered_context = "\n\n".join(
        f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )
    patch_line = f"The current patch is {current_patch}.\n" if current_patch else ""

    chain = _GENERATION_PROMPT | get_generation_llm()
    result = chain.invoke({
        "question": question,
        "numbered_context": numbered_context,
        "patch_line": patch_line,
    })
    return result.content.strip()
