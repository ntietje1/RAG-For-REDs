"""LLM-based temporal scope and source authority classifier."""

import json
import logging

from config.client import get_client
from config.pipeline_config import GENERATION_MODEL

logger = logging.getLogger(__name__)

_VALID_SCOPES = {"evergreen", "version-sensitive", "mixed"}
_VALID_SOURCES = {"riot_patch_notes", "lolalytics", "wiki", "reddit"}

_SYSTEM_PROMPT = """\
You are a query classifier for a League of Legends knowledge system.
Given a user query, output a JSON object with exactly two fields:

1. "temporal_scope": one of "evergreen", "version-sensitive", or "mixed"
   - "evergreen": the answer is unlikely to change across patches (ability descriptions, lore, general mechanics)
   - "version-sensitive": the answer depends on a specific patch or the current game state (nerfs, buffs, meta, win rates)
   - "mixed": the query needs both stable knowledge and recent/patch-specific information

2. "authority_weights": an object with weights for each source type, between 0.1 and 1.0:
   - "riot_patch_notes": official Riot Games patch notes
   - "lolalytics": champion statistics (win rate, pick rate, tier)
   - "wiki": League of Legends community wiki (abilities, items, mechanics)
   - "reddit": community discussion from r/leagueoflegends

Higher weight means the source is more relevant for answering this query.

Respond with ONLY the JSON object, no other text."""

_FEW_SHOT_EXAMPLES = [
    ("What does Zeri's W ability do?", {
        "temporal_scope": "evergreen",
        "authority_weights": {"riot_patch_notes": 0.3, "lolalytics": 0.1, "wiki": 1.0, "reddit": 0.2},
    }),
    ("Was Zeri nerfed in patch 25.S1.3?", {
        "temporal_scope": "version-sensitive",
        "authority_weights": {"riot_patch_notes": 1.0, "lolalytics": 0.4, "wiki": 0.3, "reddit": 0.5},
    }),
    ("Is Zeri good right now?", {
        "temporal_scope": "version-sensitive",
        "authority_weights": {"riot_patch_notes": 0.6, "lolalytics": 1.0, "wiki": 0.2, "reddit": 0.8},
    }),
    ("How has Jinx changed over recent patches and what are her core mechanics?", {
        "temporal_scope": "mixed",
        "authority_weights": {"riot_patch_notes": 0.9, "lolalytics": 0.6, "wiki": 0.8, "reddit": 0.5},
    }),
    ("What items should I build on Kayn?", {
        "temporal_scope": "mixed",
        "authority_weights": {"riot_patch_notes": 0.4, "lolalytics": 0.9, "wiki": 0.7, "reddit": 0.8},
    }),
]


def _validate(raw: dict) -> dict:
    """Ensure temporal_scope is valid and authority_weights are clamped to [0.1, 1.0]."""
    temporal_scope = raw.get("temporal_scope", "mixed")
    if temporal_scope not in _VALID_SCOPES:
        logger.warning("Invalid temporal_scope '%s', defaulting to 'mixed'", temporal_scope)
        temporal_scope = "mixed"

    raw_weights = raw.get("authority_weights", {})
    authority_weights = {}
    for source in _VALID_SOURCES:
        w = raw_weights.get(source, 0.5)
        if not isinstance(w, (int, float)):
            logger.warning("Non-numeric weight for '%s': %r, defaulting to 0.5", source, w)
            w = 0.5
        clamped = max(0.1, min(1.0, float(w)))
        if clamped != w:
            logger.warning("Clamped authority weight for '%s': %.3f -> %.3f", source, w, clamped)
        authority_weights[source] = clamped

    return {"temporal_scope": temporal_scope, "authority_weights": authority_weights}


def classify_query(query: str) -> dict:
    """Classify a query's temporal scope and source authority weights."""
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for q, response in _FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": json.dumps(response)})
    messages.append({"role": "user", "content": query})

    client = get_client()
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.0,
    )
    content = response.choices[0].message.content.strip()

    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        logger.error("Classifier returned invalid JSON: %s", content)
        raw = {}

    return _validate(raw)
