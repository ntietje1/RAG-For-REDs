"""Per-source cleaning functions for raw scraped JSON content."""

import re

_FANDOM_NAV_TERMS = frozenset({
    "Games",
    "Universe",
    "•",
    "League of Legends",
    "Teamfight Tactics",
    "Legends of Runeterra",
    "Wild Rift",
})

_EDIT_NOTICE_RE = re.compile(
    r"This article was last edited by[^\n]*\n?", re.IGNORECASE
)

TOP_N_COMMENTS = 5


def is_redirect(raw_text: str) -> bool:
    """Return True if a wiki page is a redirect stub (no real content)."""
    return raw_text.strip().startswith("Redirect to:")


def strip_boilerplate(raw_text: str) -> str:
    """Remove the Fandom navigation bar that prefixes many wiki pages.

    The bar always starts with 'Games\\nUniverse' and lists game titles
    separated by '•'. Skip all leading lines that belong to this set.
    """
    if not raw_text.startswith("Games\nUniverse"):
        return raw_text
    lines = raw_text.split("\n")
    i = 0
    while i < len(lines) and lines[i].strip() in _FANDOM_NAV_TERMS:
        i += 1
    return "\n".join(lines[i:]).strip()


def clean_wiki(raw_text: str) -> str:
    """Strip navigation boilerplate and inline edit notices from a wiki page."""
    text = strip_boilerplate(raw_text)
    text = _EDIT_NOTICE_RE.sub("", text)
    return text.strip()


def clean_reddit(content: dict, top_n: int = TOP_N_COMMENTS) -> str:
    """Concatenate a Reddit post's title, body, and top-scoring comments.

    Posts with no text body (link/image posts) still include the title and
    whatever comments are present.
    """
    parts: list[str] = []

    title = (content.get("title") or "").strip()
    if title:
        parts.append(title)

    selftext = (content.get("selftext") or "").strip()
    if selftext:
        parts.append(selftext)

    comments: list[dict] = content.get("comments") or []
    top_comments = sorted(comments, key=lambda c: c.get("score", 0), reverse=True)
    for comment in top_comments[:top_n]:
        body = (comment.get("body") or "").strip()
        # Skip deleted/removed placeholders
        if body and body not in ("[deleted]", "[removed]"):
            parts.append(body)

    return "\n\n".join(parts)


def serialize_stats_champion(champ: dict, patch_version: str) -> str:
    """Serialise one champion's patch stats row to a retrievable natural-language sentence.

    Example:
        "Patch 26.1: Nunu - S+ tier jungle. Win rate: 55.8% (+2.3%),
         Pick: 5.0%, Ban: 4.0%, 117k games."
    """
    delta = champ.get("win_rate_delta", 0) or 0
    delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"

    games = champ.get("games", 0) or 0
    games_str = f"{games // 1000}k" if games >= 1000 else str(games)

    return (
        f"Patch {patch_version}: {champ['name']} \u2014 {champ.get('tier', '?')} tier "
        f"{champ.get('lane', '?')}. "
        f"Win rate: {champ.get('win_rate', 0):.1f}% ({delta_str}%), "
        f"Pick: {champ.get('pick_rate', 0):.1f}%, "
        f"Ban: {champ.get('ban_rate', 0):.1f}%, "
        f"{games_str} games."
    )
