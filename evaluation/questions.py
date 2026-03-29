"""Question set management for evaluation."""

import json
from pathlib import Path


def load_questions(path: Path) -> list[dict]:
    """Load evaluation questions from a JSON file.

    Expected format: list of {"question": str, "expected_answer": str, "category": str}
    """
    raise NotImplementedError


def generate_questions(documents_dir: Path, num_questions: int = 50) -> list[dict]:
    """Auto-generate evaluation questions from the document corpus."""
    raise NotImplementedError
