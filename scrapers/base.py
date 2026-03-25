import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from config.settings import USER_AGENT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


class BaseScraper:
    """Base class for all scrapers with shared utilities."""

    source_name = "base"
    rate_limit = 1.0

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.logger = logging.getLogger(self.source_name)

    def ensure_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def make_request(self, url, delay=None, max_retries=3, **kwargs):
        """GET request with rate limiting and exponential backoff retry."""
        if delay is None:
            delay = self.rate_limit

        for attempt in range(max_retries):
            try:
                if attempt > 0 or delay > 0:
                    time.sleep(delay if attempt == 0 else delay * (2 ** attempt))

                response = self.session.get(url, timeout=30, **kwargs)
                response.raise_for_status()
                return response

            except requests.exceptions.HTTPError as e:
                if response.status_code == 404:
                    self.logger.warning(f"404 Not Found: {url}")
                    return None
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", delay * (2 ** attempt)))
                    self.logger.warning(f"Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                self.logger.error(f"HTTP {response.status_code} for {url}: {e}")
                if attempt == max_retries - 1:
                    return None
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None

        return None

    def save_json(self, data, filename):
        """Save data as a JSON file in the output directory."""
        self.ensure_output_dir()
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved: {filepath}")

    def build_document(self, content, metadata_overrides=None):
        """Build a standardized document with metadata."""
        metadata = {
            "source": self.source_name,
            "date": None,
            "patch_version": None,
            "url": None,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "content_type": self.source_name,
        }
        if metadata_overrides:
            metadata.update(metadata_overrides)

        return {"metadata": metadata, "content": content}

    def file_exists(self, filename):
        """Check if output file already exists (for resume capability)."""
        return (self.output_dir / filename).exists()

    def run(self):
        raise NotImplementedError
