"""Re-rank retrieved chunks using cross-encoder relevance, temporal decay, and source authority."""

import math

from config.pipeline_config import MAX_PER_SOURCE, TEMPORAL_LAMBDA


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
    temporal_scope: str | None = None,
    target_patch: str | None = None,
    authority_weights: dict[str, float] | None = None,
    query: str | None = None,
    use_cross_encoder: bool = False,
    final_k: int = 5,
    temporal_sensitivity: float | None = None,
) -> list[dict]:
    """Re-rank candidates by: relevance_score * temporal_decay * authority_weight
    OR cosine_score * temporal_decay * authority_weight

    Each factor is optional and applied only when its corresponding argument
    is supplied:
      - use_cross_encoder + query: replace cosine with cross-encoder scores
      - temporal_scope: apply exponential decay based on patch age
      - target_patch: use this patch as the decay reference point instead of
        current_patch (for historical patch queries)
      - authority_weights: multiply by per-source weight
      - temporal_sensitivity: continuous float [0.0-1.0] used directly as λ
        (overrides the discrete TEMPORAL_LAMBDA lookup when provided)
    """
    if use_cross_encoder and query is not None:
        from retrieval.cross_encoder import score_pairs

        texts = [c.get("text", "") for c in candidates]
        ce_scores = score_pairs(query, texts)
    else:
        ce_scores = None

    # Continuous mode: use temporal_sensitivity directly as lambda
    # Discrete mode (fallback): use TEMPORAL_LAMBDA lookup
    if temporal_sensitivity is not None:
        lam = temporal_sensitivity
    else:
        lam = TEMPORAL_LAMBDA.get(temporal_scope, 0.0) if temporal_scope else 0.0

    # Determine the reference point for temporal decay.  When the query targets
    # a specific historical patch, decay relative to that patch so chunks from
    # that patch are NOT penalised.  Otherwise decay from the current patch.
    if target_patch and target_patch in patch_index:
        reference_idx = patch_index[target_patch]
    else:
        reference_idx = patch_index.get(current_patch, 0)

    scored = []
    for i, chunk in enumerate(candidates):
        score = ce_scores[i] if ce_scores is not None else chunk.get("score", 0.0)

        if lam > 0.0 and chunk.get("patch_version"):
            chunk_idx = patch_index.get(chunk["patch_version"], reference_idx)
            age = abs(reference_idx - chunk_idx)
            score *= math.exp(-lam * age)

        if authority_weights:
            score *= authority_weights.get(chunk.get("source", ""), 0.5)

        scored.append({**chunk, "adjusted_score": score})

    scored.sort(key=lambda c: c["adjusted_score"], reverse=True)

    # keep only the best chunk per source document.
    # only allow up to MAX_PER_SOURCE of the same source
    seen_docs: set[str] = set()
    source_counts: dict[str, int] = {}
    deduped: list[dict] = []
    for chunk in scored:
        doc_id = chunk.get("url") or chunk.get("doc_id", "")
        src = chunk.get("source", "")
        if doc_id in seen_docs:
            continue
        if source_counts.get(src, 0) >= MAX_PER_SOURCE:
            continue
        seen_docs.add(doc_id)
        source_counts[src] = source_counts.get(src, 0) + 1
        deduped.append(chunk)
        if len(deduped) == final_k:
            break
    return deduped
