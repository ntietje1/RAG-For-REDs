import re

from bs4 import BeautifulSoup

from config import OUTPUT_DIRS, RATE_LIMITS, WIKI_API_URL
from scrapers.base import BaseScraper


# Substrings that indicate non-LoL pages to skip
SKIP_GAMES = ["Teamfight Tactics", "TFT", "Legends of Runeterra", "LoR", "Wild Rift"]


class WikiScraper(BaseScraper):
    source_name = "wiki"
    rate_limit = RATE_LIMITS["wiki"]

    def __init__(self):
        super().__init__(OUTPUT_DIRS["wiki"])

    def _enumerate_all_pages(self):
        """Get all page titles from the wiki using the MediaWiki API."""
        all_titles = []
        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": "500",
            "apnamespace": "0",  # Main namespace only
            "format": "json",
        }

        while True:
            response = self.make_request(WIKI_API_URL, params=params)
            if response is None:
                break

            data = response.json()
            pages = data.get("query", {}).get("allpages", [])
            all_titles.extend(page["title"] for page in pages)

            # Check for continuation
            cont = data.get("continue")
            if cont and "apcontinue" in cont:
                params["apcontinue"] = cont["apcontinue"]
                self.logger.info(f"Enumerated {len(all_titles)} pages so far...")
            else:
                break

        self.logger.info(f"Total pages enumerated: {len(all_titles)}")
        return all_titles

    def _should_skip(self, title):
        """Check if a page title should be skipped (non-LoL games)."""
        # Keep /LoL subpages (e.g. "Ahri/LoL") — these are the LoL-specific pages
        if title.endswith("/LoL"):
            return False
        # Skip subpages for other games
        if title.endswith(("/LoR", "/TFT", "/WR")):
            return True
        # Skip pages for other games (TFT, LoR, Wild Rift)
        for game in SKIP_GAMES:
            if game in title:
                return True
        return False

    def _fetch_page_content(self, title):
        """Fetch parsed content for a single wiki page."""
        params = {
            "action": "parse",
            "page": title,
            "prop": "text|categories|displaytitle",
            "format": "json",
        }
        response = self.make_request(WIKI_API_URL, params=params)
        if response is None:
            return None

        data = response.json()
        if "error" in data:
            self.logger.warning(f"API error for '{title}': {data['error'].get('info', 'unknown')}")
            return None

        parse_data = data.get("parse", {})
        html_content = parse_data.get("text", {}).get("*", "")
        categories = [
            cat["*"] for cat in parse_data.get("categories", [])
        ]
        display_title = parse_data.get("displaytitle", title)
        page_id = parse_data.get("pageid")

        # Strip HTML to plain text
        soup = BeautifulSoup(html_content, "lxml")

        # Remove unwanted elements
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()
        # Remove navigation/sidebar elements
        for tag in soup.find_all(class_=re.compile(r"navbox|sidebar|mw-editsection|toc")):
            tag.decompose()

        raw_text = soup.get_text(separator="\n", strip=True)
        # Clean up excessive whitespace
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)

        return {
            "title": display_title,
            "page_id": page_id,
            "categories": categories,
            "raw_text": raw_text,
        }

    def _sanitize_filename(self, title):
        """Convert a page title to a safe filename."""
        safe = re.sub(r'[<>:"/\\|?*]', "_", title)
        safe = safe.strip(". ")
        # Truncate to avoid OS filename limits
        if len(safe) > 200:
            safe = safe[:200]
        return safe + ".json"

    def run(self):
        self.ensure_output_dir()

        self.logger.info("Phase 1: Enumerating all wiki pages...")
        all_titles = self._enumerate_all_pages()

        # Filter out non-LoL pages
        titles = [t for t in all_titles if not self._should_skip(t)]
        self.logger.info(f"After filtering: {len(titles)} pages to scrape")

        self.logger.info("Phase 2: Fetching page content...")
        success_count = 0
        skip_count = 0

        for i, title in enumerate(titles):
            filename = self._sanitize_filename(title)

            if self.file_exists(filename):
                skip_count += 1
                continue

            content = self._fetch_page_content(title)
            if content is None:
                continue

            # Skip pages with very little content
            if len(content["raw_text"]) < 50:
                self.logger.debug(f"Skipping '{title}' (too short)")
                continue

            url = f"https://leagueoflegends.fandom.com/wiki/{title.replace(' ', '_')}"
            doc = self.build_document(
                content=content,
                metadata_overrides={
                    "url": url,
                    "content_type": "wiki_page",
                },
            )

            self.save_json(doc, filename)
            success_count += 1

            if (i + 1) % 100 == 0:
                self.logger.info(f"Progress: {i + 1}/{len(titles)} pages processed ({success_count} saved, {skip_count} skipped)")

        self.logger.info(
            f"Wiki scraping complete: {success_count} saved, {skip_count} skipped (already existed)"
        )


if __name__ == "__main__":
    scraper = WikiScraper()
    scraper.run()
