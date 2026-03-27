from config.settings import DATA_DIR

OUTPUT_DIRS = {
    "patch_notes": DATA_DIR / "patch_notes",
    "wiki": DATA_DIR / "wiki",
    "reddit": DATA_DIR / "reddit",
    "stats": DATA_DIR / "stats",
}

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
    """Generate patch version strings covering 2024 through current.

    2024 season (14.x): Riot used season-based numbering, no leading zeros in URLs.
    2025 season: S1.1-S1.3 for first three patches, then 25.04-25.24 with leading zeros.
    2026 season: 26.1 through current.
    """
    versions = []

    # 2024 season patches (Riot used season 14 numbering, no leading zeros)
    for i in range(1, 25):
        versions.append(f"14.{i}")

    # 2025 season: first three patches used S1 naming
    for i in range(1, 4):
        versions.append(f"25.S1.{i}")

    # 2025 numbered patches (25.04 onward, leading zeros for single digits)
    for i in range(4, 25):
        versions.append(f"25.{i:02d}")

    # 2026 patches (26.1 through current ~26.6)
    for i in range(1, 7):
        versions.append(f"26.{i}")

    return versions


def patch_version_to_slug(version):
    """Convert a patch version like '25.S1.3' or '25.04' to URL slug like '25-s1-3' or '25-04'.

    Preserves leading zeros since Riot URLs require them (e.g. patch-25-04-notes, not patch-25-4-notes).
    """
    return version.lower().replace(".", "-")
