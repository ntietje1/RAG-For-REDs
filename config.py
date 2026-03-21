from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIRS = {
    "patch_notes": DATA_DIR / "patch_notes",
    "wiki": DATA_DIR / "wiki",
    "reddit": DATA_DIR / "reddit",
    "stats": DATA_DIR / "stats",
}

USER_AGENT = "RAG-For-REDs/1.0 (League of Legends research project)"

# Rate limits (seconds between requests)
RATE_LIMITS = {
    "patch_notes": 2.0,
    "wiki": 1.0,
    "reddit": 2.0,
    "stats": 4.0,
}

# Riot Patch Notes
RIOT_BASE_URL = "https://www.leagueoflegends.com"
RIOT_PATCH_URL_PATTERNS = [
    "/en-us/news/game-updates/patch-{slug}-notes/",
    "/en-us/news/game-updates/league-of-legends-patch-{slug}-notes/",
]

# Wiki
WIKI_API_URL = "https://leagueoflegends.fandom.com/api.php"

# Lolalytics
LOLALYTICS_BASE_URL = "https://lolalytics.com"

# Reddit
REDDIT_SUBREDDIT = "leagueoflegends"
REDDIT_SEARCH_QUERIES = [
    "patch notes",
    "meta",
    "tier list",
    "nerf buff",
    "champion changes",
    "item changes",
    "build guide",
    "ranked",
]
REDDIT_MIN_SCORE = 10
REDDIT_TOP_LIMIT = 500


def generate_patch_versions():
    """Generate patch version strings for ~1 year back.

    2025 season started with S1 patches, then numbered patches from 25.17 onward.
    """
    versions = []

    # Season 2025 special patches (S1.1 and S1.2 confirmed, S1.3 may not exist)
    for i in range(1, 4):
        versions.append(f"25.S1.{i}")

    # 2025 numbered patches start at 25.17 (not 25.01)
    for i in range(17, 25):
        versions.append(f"25.{i}")

    # 2026 patches (26.1 through current ~26.6)
    for i in range(1, 7):
        versions.append(f"26.{i}")

    return versions


def patch_version_to_slug(version):
    """Convert a patch version like '25.S1.3' or '25.04' to URL slug like '25-s1-3' or '25-4'."""
    slug = version.lower().replace(".", "-")
    # Remove leading zeros in patch number (25-04 -> 25-4)
    parts = slug.split("-")
    parts = [p.lstrip("0") or "0" for p in parts]
    return "-".join(parts)
