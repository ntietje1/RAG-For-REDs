import time

from config import OUTPUT_DIRS, RATE_LIMITS
from scrapers.base import BaseScraper

# Lolalytics uses its own patch numbering: Riot 25.x = Lola 15.x, Riot 26.x = Lola 16.x
LOLALYTICS_PATCHES = []
# 2025 season: 15.17 through 15.24
for i in range(17, 25):
    LOLALYTICS_PATCHES.append(f"15.{i}")
# 2026 season: 16.1 through 16.6
for i in range(1, 7):
    LOLALYTICS_PATCHES.append(f"16.{i}")

# Map Lolalytics patch -> Riot patch for metadata
LOLA_TO_RIOT = {}
for i in range(17, 25):
    LOLA_TO_RIOT[f"15.{i}"] = f"25.{i}"
for i in range(1, 7):
    LOLA_TO_RIOT[f"16.{i}"] = f"26.{i}"

LOLALYTICS_API_URL = "https://a1.lolalytics.com/mega/"
DDRAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMPIONS_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"


class LolalyticsScraper(BaseScraper):
    source_name = "lolalytics"
    rate_limit = RATE_LIMITS["stats"]

    def __init__(self):
        super().__init__(OUTPUT_DIRS["stats"])
        self._cid_to_name = None

    def _get_champion_names(self):
        """Fetch champion ID -> name mapping from Riot Data Dragon."""
        if self._cid_to_name is not None:
            return self._cid_to_name

        self.logger.info("Fetching champion names from Data Dragon...")
        resp = self.make_request(DDRAGON_VERSIONS_URL)
        if not resp:
            self.logger.error("Failed to fetch Data Dragon versions")
            return {}

        latest_version = resp.json()[0]
        url = DDRAGON_CHAMPIONS_URL.format(version=latest_version)
        resp = self.make_request(url)
        if not resp:
            self.logger.error("Failed to fetch champion data from Data Dragon")
            return {}

        data = resp.json()
        self._cid_to_name = {}
        for name, info in data["data"].items():
            self._cid_to_name[info["key"]] = name

        self.logger.info(f"Loaded {len(self._cid_to_name)} champion names")
        return self._cid_to_name

    def _scrape_patch(self, lola_patch):
        """Fetch tierlist data for a single patch from the Lolalytics API."""
        params = {
            "ep": "list",
            "v": "1",
            "patch": lola_patch,
            "tier": "emerald_plus",
            "queue": "420",
            "region": "all",
            "lane": "all",
        }
        self.logger.info(f"Fetching API: patch={lola_patch}")

        response = self.make_request(LOLALYTICS_API_URL, params=params)
        if not response:
            return None

        data = response.json()
        if "cid" not in data:
            self.logger.error(f"No champion data in API response for patch {lola_patch}")
            return None

        cid_to_name = self._get_champion_names()
        avg_wr = data.get("avgWr")

        champions = []
        for cid, stats in data["cid"].items():
            if stats.get("games", 0) == 0:
                continue

            name = cid_to_name.get(cid, f"Unknown({cid})")
            champions.append({
                "rank": stats.get("rank"),
                "name": name,
                "tier": self._tier_number_to_label(stats.get("tier")),
                "lane": stats.get("lane") or stats.get("defaultLane"),
                "lane_rate": stats.get("pctLane"),
                "win_rate": stats.get("wr"),
                "win_rate_delta": stats.get("avgWrDelta"),
                "pick_rate": stats.get("pr"),
                "ban_rate": stats.get("br"),
                "pbi": stats.get("pbi"),
                "games": stats.get("games"),
            })

        champions.sort(key=lambda c: (c["rank"] or 999))
        return champions, avg_wr

    @staticmethod
    def _tier_number_to_label(tier_num):
        """Convert numeric tier to label (S+, S, S-, A+, A, A-, B+, B, B-, etc.)."""
        tier_map = {
            1: "S+", 2: "S", 3: "S-",
            4: "A+", 5: "A", 6: "A-",
            7: "B+", 8: "B", 9: "B-",
            10: "C+", 11: "C", 12: "C-",
            13: "D+", 14: "D", 15: "D-",
        }
        return tier_map.get(tier_num, f"T{tier_num}" if tier_num else None)

    def run(self):
        self.ensure_output_dir()
        self.logger.info(f"Scraping stats for {len(LOLALYTICS_PATCHES)} patches...")

        success_count = 0
        for lola_patch in LOLALYTICS_PATCHES:
            riot_patch = LOLA_TO_RIOT.get(lola_patch, lola_patch)
            filename = f"patch_{riot_patch.replace('.', '_')}.json"

            if self.file_exists(filename):
                self.logger.info(f"Skipping {riot_patch} (already exists)")
                success_count += 1
                continue

            result = self._scrape_patch(lola_patch)
            if not result:
                self.logger.warning(f"No data for patch {lola_patch}")
                continue

            champions, avg_wr = result
            self.logger.info(f"Patch {riot_patch}: {len(champions)} champions found")

            doc = self.build_document(
                content={
                    "patch_version": riot_patch,
                    "lolalytics_patch": lola_patch,
                    "average_win_rate": avg_wr,
                    "champions": champions,
                    "total_champions": len(champions),
                },
                metadata_overrides={
                    "patch_version": riot_patch,
                    "url": f"{LOLALYTICS_API_URL}?ep=list&patch={lola_patch}",
                    "content_type": "champion_stats",
                },
            )

            self.save_json(doc, filename)
            success_count += 1
            time.sleep(self.rate_limit)

        self.logger.info(
            f"Stats scraping complete: {success_count}/{len(LOLALYTICS_PATCHES)} patches saved"
        )


if __name__ == "__main__":
    scraper = LolalyticsScraper()
    scraper.run()
