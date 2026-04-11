"""Question set management for evaluation."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = {"id", "question", "expected_answer", "temporality"}


def load_questions(path: Path) -> list[dict]:
    """Load evaluation questions from a JSON file.

    Expected format: list of objects, each containing at least
    ``id``, ``question``, ``expected_answer``, and ``temporality``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If a question is missing required fields.
    """
    with open(path) as f:
        questions = json.load(f)

    for i, q in enumerate(questions):
        missing = _REQUIRED_FIELDS - set(q.keys())
        if missing:
            raise ValueError(
                f"Question at index {i} (id={q.get('id', '?')}) "
                f"is missing required fields: {missing}"
            )

    logger.info("Loaded %d evaluation questions from %s", len(questions), path)
    return questions
