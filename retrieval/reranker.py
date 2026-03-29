"""Re-rank retrieved chunks using temporal decay and source authority weights."""

import math

from config.pipeline_config import TEMPORAL_LAMBDA


def _patch_sort_key(version: str) -> tuple:
    """Sort key for versions like '14.1', '25.S1.2', '26.6'.

    Numeric parts sort naturally; non-numeric parts (e.g. 'S1') sort
    after all numeric values within the same position.
    """
    parts = []
    for part in version.split("."):
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            digits = "".join(c for c in part if c.isdigit())
            parts.append((1, int(digits) if digits else 0))
    return tuple(parts)


def build_patch_index(versions: set[str]) -> dict[str, int]:
    """Map each patch version to its position in chronological order."""
    return {v: i for i, v in enumerate(sorted(versions, key=_patch_sort_key))}


def rerank(
    candidates: list[dict],
    patch_index: dict[str, int],
    current_patch: str,
    temporal_scope: str,
    authority_weights: dict[str, float],
    final_k: int = 5,
) -> list[dict]:
    """Re-rank candidates by: cosine_score * temporal_decay * authority_weight."""
    lam = TEMPORAL_LAMBDA.get(temporal_scope, 0.0) if temporal_scope else 0.0
    current_idx = patch_index.get(current_patch, 0)

    scored = []
    for chunk in candidates:
        score = chunk.get("score", 0.0)

        if lam > 0.0 and chunk.get("patch_version"):
            age = current_idx - patch_index.get(chunk["patch_version"], current_idx)
            score *= math.exp(-lam * max(0, age))

        if authority_weights:
            score *= authority_weights.get(chunk.get("source", ""), 0.5)

        scored.append({**chunk, "adjusted_score": score})

    scored.sort(key=lambda c: c["adjusted_score"], reverse=True)

    # keep only the best chunk per source document.
    seen_docs: set[str] = set()
    deduped: list[dict] = []
    for chunk in scored:
        doc_id = chunk.get("url") or chunk.get("doc_id", "")
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            deduped.append(chunk)
            if len(deduped) == final_k:
                break
    return deduped
