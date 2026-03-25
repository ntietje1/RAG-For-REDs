from bs4 import BeautifulSoup

from config.scraper_config import (
    OUTPUT_DIRS,
    RATE_LIMITS,
    RIOT_BASE_URL,
    RIOT_PATCH_URL_PATTERNS,
    generate_patch_versions,
    patch_version_to_slug,
)
from scrapers.base import BaseScraper


class PatchNotesScraper(BaseScraper):
    source_name = "riot_patch_notes"
    rate_limit = RATE_LIMITS["patch_notes"]

    def __init__(self):
        super().__init__(OUTPUT_DIRS["patch_notes"])

    def _try_fetch_patch(self, version):
        """Try multiple URL patterns for a given patch version."""
        slug = patch_version_to_slug(version)

        for pattern in RIOT_PATCH_URL_PATTERNS:
            url = RIOT_BASE_URL + pattern.format(slug=slug)
            self.logger.info(f"Trying: {url}")
            response = self.make_request(url)
            if response is not None:
                return response, url

        self.logger.warning(f"Could not find patch page for version {version}")
        return None, None

    def _parse_patch_page(self, html, url):
        """Extract patch note content from HTML."""
        soup = BeautifulSoup(html, "lxml")

        # Title from h1
        title = None
        title_tag = soup.find("h1")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Date from time element
        date = None
        time_tag = soup.find("time")
        if time_tag:
            date = time_tag.get("datetime", time_tag.get_text(strip=True))

        # find all heading + content elements.
        for tag in soup.find_all(["nav", "header", "footer", "script", "style", "noscript"]):
            tag.decompose()

        # Find all content elements in document order
        content_tags = soup.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol", "blockquote", "table"])

        sections = []
        raw_text_parts = []
        current_section = {"heading": "Introduction", "content": ""}
        found_first_heading = False

        for element in content_tags:
            text = element.get_text(separator=" ", strip=True)
            if not text:
                continue

            if element.name in ("h1", "h2", "h3"):
                if found_first_heading and current_section["content"].strip():
                    sections.append(current_section)
                current_section = {"heading": text, "content": ""}
                found_first_heading = True
                raw_text_parts.append(f"\n## {text}\n")
            elif element.name == "h4":
                current_section["content"] += f"\n### {text}\n"
                raw_text_parts.append(f"\n### {text}\n")
            else:
                current_section["content"] += text + "\n"
                raw_text_parts.append(text)

        if current_section["content"].strip():
            sections.append(current_section)

        raw_text = "\n".join(raw_text_parts)

        return {
            "title": title,
            "date": date,
            "sections": sections,
            "raw_text": raw_text,
        }

    def run(self):
        self.ensure_output_dir()
        versions = generate_patch_versions()
        self.logger.info(f"Scraping {len(versions)} patch versions...")

        success_count = 0
        for version in versions:
            filename = f"patch_{version.replace('.', '_')}.json"

            if self.file_exists(filename):
                self.logger.info(f"Skipping {version} (already exists)")
                success_count += 1
                continue

            response, url = self._try_fetch_patch(version)
            if response is None:
                continue

            parsed = self._parse_patch_page(response.text, url)

            doc = self.build_document(
                content=parsed,
                metadata_overrides={
                    "date": parsed["date"],
                    "patch_version": version,
                    "url": url,
                    "content_type": "patch_notes",
                },
            )

            self.save_json(doc, filename)
            success_count += 1

        self.logger.info(f"Patch notes scraping complete: {success_count}/{len(versions)} patches saved")


if __name__ == "__main__":
    scraper = PatchNotesScraper()
    scraper.run()
