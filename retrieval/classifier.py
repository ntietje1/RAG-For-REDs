"""LLM-based temporal scope and source authority classifier using LangChain."""

import json
import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config.pipeline_config import AUTHORITY_LEVELS, GENERATION_MODEL

load_dotenv()

logger = logging.getLogger(__name__)

_BASE_URL = "https://openrouter.ai/api/v1"

_VALID_SCOPES = {"evergreen", "version-sensitive", "mixed"}
_VALID_SOURCES = {"riot_patch_notes", "lolalytics", "wiki", "reddit"}
_AUTHORITY_LEVEL_MAP = AUTHORITY_LEVELS  # {"low": 0.2, "medium": 0.5, "high": 1.0}

_classifier_llm: ChatOpenAI | None = None


def _get_classifier_llm() -> ChatOpenAI:
    """Return the shared LangChain ChatOpenAI instance for classification."""
    global _classifier_llm
    if _classifier_llm is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set")
        _classifier_llm = ChatOpenAI(
            model=GENERATION_MODEL,
            openai_api_key=api_key,
            openai_api_base=_BASE_URL,
            temperature=0.0,
        )
    return _classifier_llm


# ---------------------------------------------------------------------------
# System prompts — shared preamble + mode-specific authority_weights section
# ---------------------------------------------------------------------------

_PREAMBLE = """\
You are a query classifier for a League of Legends knowledge system.
The current patch is {current_patch}.

Given a user query, output a JSON object with exactly five fields:

1. "reasoning": a 1-2 sentence explanation of why you chose the values below. \
Think step-by-step before committing to a classification.

2. "temporal_scope": one of "evergreen", "version-sensitive", or "mixed"
   - "evergreen": the answer is unlikely to change across patches (ability descriptions, lore, general mechanics)
   - "version-sensitive": the answer depends on a specific patch or the current game state (nerfs, buffs, meta, win rates)
   - "mixed": the query needs both stable knowledge and recent/patch-specific information

3. "target_patch": if the query explicitly references a specific patch version \
(e.g. "patch 25.S1.3"), output that version string. \
For queries about the current state, comparing multiple patches, or with no patch reference, output null.

4. "authority_weights": an object with a weight for each source type:
   - "riot_patch_notes": official Riot Games patch notes
   - "lolalytics": champion statistics (win rate, pick rate, tier)
   - "wiki": League of Legends community wiki (abilities, items, mechanics)
   - "reddit": community discussion from r/leagueoflegends
"""

_AUTHORITY_CONTINUOUS = """\
   Each weight should be a number between 0.1 and 1.0. \
Higher weight means the source is more relevant for answering this query.
"""

_AUTHORITY_DISCRETE = """\
   Each weight must be one of:
   - "high": this source is primary/essential for answering the query
   - "medium": this source provides useful supporting context
   - "low": this source is unlikely to be helpful for this query
"""

_FOOTER = """\
5. "alternate_queries": a list of 2-3 rephrased versions of the query that use \
different wording, synonyms, or more specific terms to help retrieve relevant information.

Respond with ONLY the JSON object, no other text."""

_SYSTEM_PROMPT_CONTINUOUS = _PREAMBLE + _AUTHORITY_CONTINUOUS + _FOOTER
_SYSTEM_PROMPT_DISCRETE = _PREAMBLE + _AUTHORITY_DISCRETE + _FOOTER

# ---------------------------------------------------------------------------
# Few-shot examples — 12 archetypes, one set per mode
# ---------------------------------------------------------------------------

_EXAMPLE_QUERIES = [
    # 1. Ability description (evergreen)
    "What does Zeri's W ability do?",
    # 2. Specific patch nerf (version-sensitive, target_patch set)
    "Was Zeri nerfed in patch 25.S1.3?",
    # 3. Current meta strength (version-sensitive)
    "Is Zeri good right now?",
    # 4. Mixed mechanics + history
    "How has Jinx changed over recent patches and what are her core mechanics?",
    # 5. Item build advice (mixed)
    "What items should I build on Kayn?",
    # 6. Multi-champion comparison (version-sensitive)
    "Who is stronger in the current meta, Jinx or Caitlyn?",
    # 7. Historical patch changelog (version-sensitive, target_patch set)
    "What changed in patch 25.S1.2?",
    # 8. Mechanic interaction (evergreen)
    "Does Yasuo's Wind Wall block Zeri's Q?",
    # 9. Community sentiment (mixed)
    "What does the community think about the new jungle changes?",
    # 10. Lore question (evergreen)
    "What is the relationship between Vi and Jinx in the lore?",
    # 11. Recent hotfix (version-sensitive, target_patch set)
    "Were there any hotfixes in patch 25.S1.4?",
    # 12. Cross-patch comparison (version-sensitive, target_patch null)
    "How has Volibear changed from patch 26.4 to 26.6?",
]

# Shared fields that are identical across both modes
_EXAMPLE_SHARED = [
    # 1
    {"reasoning": "This asks about a specific ability, which is stable game knowledge documented on the wiki.",
     "temporal_scope": "evergreen", "target_patch": None,
     "alternate_queries": ["Zeri Ultrashock Laser ability description", "Zeri W spell effect"]},
    # 2
    {"reasoning": "This asks about balance changes in a specific patch. Patch notes are the primary source.",
     "temporal_scope": "version-sensitive", "target_patch": "25.S1.3",
     "alternate_queries": ["Zeri balance changes patch 25.S1.3", "Zeri nerfs buffs 25.S1.3"]},
    # 3
    {"reasoning": "Asking about current strength requires up-to-date stats and community perception.",
     "temporal_scope": "version-sensitive", "target_patch": None,
     "alternate_queries": ["Zeri win rate current patch", "Zeri tier strength current meta"]},
    # 4
    {"reasoning": "This needs both stable mechanic knowledge and recent patch history, spanning multiple sources.",
     "temporal_scope": "mixed", "target_patch": None,
     "alternate_queries": ["Jinx patch notes changes recent", "Jinx abilities passive rockets"]},
    # 5
    {"reasoning": "Item builds depend on the current meta but also require knowledge of item mechanics.",
     "temporal_scope": "mixed", "target_patch": None,
     "alternate_queries": ["Kayn recommended item build guide", "Kayn Shadow Assassin Rhaast items current patch"]},
    # 6
    {"reasoning": "Comparing champion strength requires current stats and meta context.",
     "temporal_scope": "version-sensitive", "target_patch": None,
     "alternate_queries": ["Jinx vs Caitlyn win rate current patch", "Jinx Caitlyn ADC comparison tier"]},
    # 7
    {"reasoning": "This asks for the changelog of a specific historical patch. Patch notes are authoritative.",
     "temporal_scope": "version-sensitive", "target_patch": "25.S1.2",
     "alternate_queries": ["patch 25.S1.2 changelog all changes", "25.S1.2 balance updates champion items"]},
    # 8
    {"reasoning": "This is about game mechanic interactions, which are stable and documented on the wiki.",
     "temporal_scope": "evergreen", "target_patch": None,
     "alternate_queries": ["Yasuo Wind Wall projectile block interactions", "Zeri Q blocked by Wind Wall"]},
    # 9
    {"reasoning": "Community sentiment requires Reddit discussion alongside official context about the changes.",
     "temporal_scope": "mixed", "target_patch": None,
     "alternate_queries": ["jungle changes community reaction opinion", "new jungle update player feedback reddit"]},
    # 10
    {"reasoning": "Lore is stable game knowledge best documented on the wiki.",
     "temporal_scope": "evergreen", "target_patch": None,
     "alternate_queries": ["Vi Jinx sisters lore Arcane", "Vi Jinx relationship backstory"]},
    # 11
    {"reasoning": "Hotfixes are tied to a specific patch. Patch notes and community discussion are key.",
     "temporal_scope": "version-sensitive", "target_patch": "25.S1.4",
     "alternate_queries": ["patch 25.S1.4 hotfix emergency changes", "25.S1.4 hotfix bug fix balance"]},
    # 12
    {"reasoning": "This compares changes across multiple patches, so no single target patch applies.",
     "temporal_scope": "version-sensitive", "target_patch": None,
     "alternate_queries": ["Volibear changes patch 26.4 26.5 26.6", "Volibear balance history recent patches"]},
]

# Authority weights per example — continuous (floats)
_WEIGHTS_CONTINUOUS = [
    {"riot_patch_notes": 0.3, "lolalytics": 0.1, "wiki": 1.0, "reddit": 0.2},   # 1
    {"riot_patch_notes": 1.0, "lolalytics": 0.4, "wiki": 0.3, "reddit": 0.5},   # 2
    {"riot_patch_notes": 0.6, "lolalytics": 1.0, "wiki": 0.2, "reddit": 0.8},   # 3
    {"riot_patch_notes": 0.9, "lolalytics": 0.6, "wiki": 0.8, "reddit": 0.5},   # 4
    {"riot_patch_notes": 0.4, "lolalytics": 0.9, "wiki": 0.7, "reddit": 0.8},   # 5
    {"riot_patch_notes": 0.5, "lolalytics": 1.0, "wiki": 0.2, "reddit": 0.7},   # 6
    {"riot_patch_notes": 1.0, "lolalytics": 0.3, "wiki": 0.3, "reddit": 0.4},   # 7
    {"riot_patch_notes": 0.2, "lolalytics": 0.1, "wiki": 1.0, "reddit": 0.3},   # 8
    {"riot_patch_notes": 0.6, "lolalytics": 0.3, "wiki": 0.3, "reddit": 1.0},   # 9
    {"riot_patch_notes": 0.1, "lolalytics": 0.1, "wiki": 1.0, "reddit": 0.2},   # 10
    {"riot_patch_notes": 1.0, "lolalytics": 0.3, "wiki": 0.2, "reddit": 0.6},   # 11
    {"riot_patch_notes": 1.0, "lolalytics": 0.6, "wiki": 0.3, "reddit": 0.5},   # 12
]

# Authority weights per example — discrete (strings)
_WEIGHTS_DISCRETE = [
    {"riot_patch_notes": "low",    "lolalytics": "low",    "wiki": "high",   "reddit": "low"},     # 1
    {"riot_patch_notes": "high",   "lolalytics": "medium", "wiki": "low",    "reddit": "medium"},   # 2
    {"riot_patch_notes": "medium", "lolalytics": "high",   "wiki": "low",    "reddit": "high"},     # 3
    {"riot_patch_notes": "high",   "lolalytics": "medium", "wiki": "high",   "reddit": "medium"},   # 4
    {"riot_patch_notes": "medium", "lolalytics": "high",   "wiki": "medium", "reddit": "high"},     # 5
    {"riot_patch_notes": "medium", "lolalytics": "high",   "wiki": "low",    "reddit": "medium"},   # 6
    {"riot_patch_notes": "high",   "lolalytics": "low",    "wiki": "low",    "reddit": "medium"},   # 7
    {"riot_patch_notes": "low",    "lolalytics": "low",    "wiki": "high",   "reddit": "low"},      # 8
    {"riot_patch_notes": "medium", "lolalytics": "low",    "wiki": "low",    "reddit": "high"},     # 9
    {"riot_patch_notes": "low",    "lolalytics": "low",    "wiki": "high",   "reddit": "low"},      # 10
    {"riot_patch_notes": "high",   "lolalytics": "low",    "wiki": "low",    "reddit": "medium"},   # 11
    {"riot_patch_notes": "high",   "lolalytics": "medium", "wiki": "low",    "reddit": "medium"},   # 12
]


def _build_examples(discrete: bool) -> list[tuple[str, dict]]:
    """Assemble the few-shot example list for the selected mode."""
    weights = _WEIGHTS_DISCRETE if discrete else _WEIGHTS_CONTINUOUS
    examples = []
    for query, shared, w in zip(_EXAMPLE_QUERIES, _EXAMPLE_SHARED, weights):
        response = {**shared, "authority_weights": w}
        examples.append((query, response))
    return examples


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(raw: dict) -> dict:
    """Validate and normalise classifier output. Handles both continuous and discrete weights."""
    reasoning = raw.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = ""

    temporal_scope = raw.get("temporal_scope", "mixed")
    if temporal_scope not in _VALID_SCOPES:
        logger.warning("Invalid temporal_scope '%s', defaulting to 'mixed'", temporal_scope)
        temporal_scope = "mixed"

    target_patch = raw.get("target_patch")
    if target_patch is not None:
        if not isinstance(target_patch, str) or not target_patch.strip():
            logger.warning("Invalid target_patch %r, defaulting to None", target_patch)
            target_patch = None

    raw_weights = raw.get("authority_weights", {})
    authority_weights = {}
    for source in _VALID_SOURCES:
        w = raw_weights.get(source, 0.5)
        if isinstance(w, str):
            # Discrete mode: map string level to float
            if w.lower() in _AUTHORITY_LEVEL_MAP:
                authority_weights[source] = _AUTHORITY_LEVEL_MAP[w.lower()]
            else:
                logger.warning("Unknown authority level '%s' for '%s', defaulting to 0.5", w, source)
                authority_weights[source] = 0.5
        elif isinstance(w, (int, float)):
            # Continuous mode: clamp to [0.1, 1.0]
            clamped = max(0.1, min(1.0, float(w)))
            if clamped != float(w):
                logger.warning("Clamped authority weight for '%s': %.3f -> %.3f", source, w, clamped)
            authority_weights[source] = clamped
        else:
            logger.warning("Non-numeric weight for '%s': %r, defaulting to 0.5", source, w)
            authority_weights[source] = 0.5

    alternate_queries = raw.get("alternate_queries", [])
    if not isinstance(alternate_queries, list):
        alternate_queries = []

    return {
        "reasoning": reasoning,
        "temporal_scope": temporal_scope,
        "target_patch": target_patch,
        "authority_weights": authority_weights,
        "alternate_queries": alternate_queries,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_query(
    query: str, current_patch: str, discrete_weights: bool = False
) -> dict:
    """Classify a query's temporal scope, authority weights, and alternate phrasings.

    Args:
        query: The user's question.
        current_patch: The latest patch version string (e.g. "26.6").
        discrete_weights: If True, instruct the LLM to use discrete authority
            levels ("low"/"medium"/"high") instead of continuous floats.
    """
    template = _SYSTEM_PROMPT_DISCRETE if discrete_weights else _SYSTEM_PROMPT_CONTINUOUS
    system_prompt = template.format(current_patch=current_patch)
    examples = _build_examples(discrete_weights)

    messages = [SystemMessage(content=system_prompt)]
    for q, response in examples:
        messages.append(HumanMessage(content=q))
        messages.append(AIMessage(content=json.dumps(response)))
    messages.append(HumanMessage(content=query))

    llm = _get_classifier_llm()
    result = llm.invoke(messages)
    content = result.content.strip()

    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        logger.error("Classifier returned invalid JSON: %s", content)
        raw = {}

    return _validate(raw)
