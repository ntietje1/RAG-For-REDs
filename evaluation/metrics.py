"""LLM-as-judge evaluation metrics for RAG pipeline ablation study.

Uses GPT-4o-mini (via OpenRouter) as an independent judge — a different model
family from the Gemini Flash generator to avoid self-evaluation bias.
"""

import json
import logging
import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.pipeline_config import EVAL_MODEL

load_dotenv()

logger = logging.getLogger(__name__)

_BASE_URL = "https://openrouter.ai/api/v1"

_judge_llm: ChatOpenAI | None = None


def _get_judge_llm() -> ChatOpenAI:
    """Return the shared LangChain ChatOpenAI instance for evaluation judging."""
    global _judge_llm
    if _judge_llm is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set")
        _judge_llm = ChatOpenAI(
            model=EVAL_MODEL,
            openai_api_key=api_key,
            openai_api_base=_BASE_URL,
            temperature=0.0,
        )
    return _judge_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_parse_score(text: str) -> dict:
    """Parse an LLM judge response into {"score": float, "reasoning": str}."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        score = float(parsed.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reasoning = str(parsed.get("reasoning", ""))
        return {"score": score, "reasoning": reasoning}
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Failed to parse judge response: %s — %s", exc, text[:200])
        return {"score": 0.0, "reasoning": f"parse_error: {exc}"}


def _judge(system_prompt: str, user_prompt: str) -> dict:
    """Call the judge model and return parsed score + reasoning."""
    llm = _get_judge_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    result = llm.invoke(messages)
    content = result.content.strip()
    return _safe_parse_score(content)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def score_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """Score how well the answer is grounded in the retrieved context.

    0.0 = the answer fabricates information not in the context.
    1.0 = every claim in the answer is directly supported by the context.
    """
    system = (
        "You are an evaluation judge. Your task is to assess whether an answer "
        "is faithful to the provided context passages.\n\n"
        "Score 0.0 = the answer fabricates information not present in the context.\n"
        "Score 1.0 = every claim in the answer is directly supported by the context.\n\n"
        "Respond with ONLY a JSON object: {\"score\": <float 0-1>, \"reasoning\": \"<brief explanation>\"}"
    )
    numbered = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    user = f"Context:\n{numbered}\n\nAnswer:\n{answer}"
    return _judge(system, user)["score"]


def score_answer_relevancy(question: str, answer: str) -> float:
    """Score how relevant the answer is to the question.

    0.0 = completely off-topic or fails to address the question.
    1.0 = directly and completely answers the question.
    """
    system = (
        "You are an evaluation judge. Your task is to assess whether an answer "
        "is relevant to and addresses the given question.\n\n"
        "Score 0.0 = the answer is completely off-topic or does not address the question.\n"
        "Score 1.0 = the answer directly and completely addresses the question.\n\n"
        "Respond with ONLY a JSON object: {\"score\": <float 0-1>, \"reasoning\": \"<brief explanation>\"}"
    )
    user = f"Question:\n{question}\n\nAnswer:\n{answer}"
    return _judge(system, user)["score"]


def score_context_precision(question: str, context_chunks: list[str]) -> float:
    """Score what fraction of retrieved chunks are relevant to the question.

    0.0 = none of the retrieved chunks are relevant.
    1.0 = all retrieved chunks are relevant to the question.
    """
    system = (
        "You are an evaluation judge. Your task is to assess what fraction of "
        "the provided context passages are relevant to answering the question.\n\n"
        "Score 0.0 = none of the passages are relevant.\n"
        "Score 1.0 = all passages are relevant to the question.\n\n"
        "Respond with ONLY a JSON object: {\"score\": <float 0-1>, \"reasoning\": \"<brief explanation>\"}"
    )
    numbered = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    user = f"Question:\n{question}\n\nRetrieved passages:\n{numbered}"
    return _judge(system, user)["score"]


def score_context_recall(
    expected_answer: str, context_chunks: list[str]
) -> float:
    """Score how much of the expected answer's key information is in the context.

    0.0 = none of the key facts from the expected answer appear in the context.
    1.0 = all key facts from the expected answer are present in the context.
    """
    system = (
        "You are an evaluation judge. Your task is to assess how much of the "
        "expected answer's key information is present in the retrieved context.\n\n"
        "Score 0.0 = none of the key facts from the expected answer appear in the context.\n"
        "Score 1.0 = all key facts from the expected answer are present in the context.\n\n"
        "Respond with ONLY a JSON object: {\"score\": <float 0-1>, \"reasoning\": \"<brief explanation>\"}"
    )
    numbered = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    user = f"Expected answer:\n{expected_answer}\n\nRetrieved passages:\n{numbered}"
    return _judge(system, user)["score"]


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def score_all(
    question: str,
    answer: str,
    expected_answer: str,
    context_chunks: list[str],
) -> dict[str, float]:
    """Compute all four metrics and return as a dict."""
    return {
        "faithfulness": score_faithfulness(answer, context_chunks),
        "answer_relevancy": score_answer_relevancy(question, answer),
        "context_precision": score_context_precision(question, context_chunks),
        "context_recall": score_context_recall(expected_answer, context_chunks),
    }
