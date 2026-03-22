from datetime import datetime, timezone

from config.scraper_config import (
    OUTPUT_DIRS,
    REDDIT_MIN_SCORE,
    REDDIT_SEARCH_QUERIES,
    REDDIT_SUBREDDIT,
)
from scrapers.base import BaseScraper

REDDIT_JSON_BASE = f"https://www.reddit.com/r/{REDDIT_SUBREDDIT}"

# Sorting categories to scrape (each paginated separately)
LISTING_ENDPOINTS = [
    "/top.json?t=year",
    "/hot.json",
    "/new.json",
]


class RedditScraper(BaseScraper):
    source_name = "reddit"
    rate_limit = 2.0  # Be polite without OAuth

    def __init__(self):
        super().__init__(OUTPUT_DIRS["reddit"])
        # Reddit requires a non-generic User-Agent or it returns 429
        self.session.headers.update({
            "User-Agent": "RAG-For-REDs/1.0 (university research project)",
        })
        self.seen_ids = set()

    def _fetch_listing(self, url, max_pages=10):
        """Fetch a Reddit JSON listing with pagination via 'after' token."""
        posts = []
        after = None

        for page in range(max_pages):
            page_url = url
            if after:
                separator = "&" if "?" in url else "?"
                page_url = f"{url}{separator}after={after}&limit=100"
            else:
                separator = "&" if "?" in url else "?"
                page_url = f"{url}{separator}limit=100"

            response = self.make_request(page_url)
            if response is None:
                break

            try:
                data = response.json()
            except Exception:
                self.logger.error(f"Failed to parse JSON from {page_url}")
                break

            listing = data.get("data", {})
            children = listing.get("children", [])
            if not children:
                break

            for child in children:
                if child.get("kind") == "t3":  # t3 = link/post
                    posts.append(child["data"])

            after = listing.get("after")
            if not after:
                break

            self.logger.info(f"  Page {page + 1}: {len(children)} posts (total: {len(posts)})")

        return posts

    def _fetch_comments(self, permalink):
        """Fetch top-level comments for a post via JSON endpoint."""
        url = f"https://www.reddit.com{permalink}.json?limit=50&sort=top"
        response = self.make_request(url)
        if response is None:
            return []

        try:
            data = response.json()
        except Exception:
            return []

        # Reddit returns [post_listing, comments_listing]
        if not isinstance(data, list) or len(data) < 2:
            return []

        comments = []
        children = data[1].get("data", {}).get("children", [])
        for child in children:
            if child.get("kind") != "t1":  # t1 = comment
                continue
            c = child["data"]
            if c.get("score", 0) < 1:
                continue
            comments.append({
                "author": c.get("author", "[deleted]"),
                "body": c.get("body", ""),
                "score": c.get("score", 0),
                "created_utc": datetime.fromtimestamp(
                    c.get("created_utc", 0), tz=timezone.utc
                ).isoformat(),
            })

        comments.sort(key=lambda x: x["score"], reverse=True)
        return comments[:50]

    def _process_post(self, post_data):
        """Process a single post dict from Reddit JSON API."""
        post_id = post_data.get("id")
        if not post_id or post_id in self.seen_ids:
            return None

        score = post_data.get("score", 0)
        if score < REDDIT_MIN_SCORE:
            return None

        self.seen_ids.add(post_id)

        permalink = post_data.get("permalink", "")
        created_utc = post_data.get("created_utc", 0)
        created = datetime.fromtimestamp(created_utc, tz=timezone.utc)

        # Fetch comments for this post
        comments = self._fetch_comments(permalink)

        content = {
            "title": post_data.get("title", ""),
            "author": post_data.get("author", "[deleted]"),
            "score": score,
            "upvote_ratio": post_data.get("upvote_ratio"),
            "num_comments": post_data.get("num_comments", 0),
            "selftext": post_data.get("selftext", ""),
            "link_flair_text": post_data.get("link_flair_text"),
            "created_utc": created.isoformat(),
            "comments": comments,
        }

        url = f"https://www.reddit.com{permalink}"

        return post_id, self.build_document(
            content=content,
            metadata_overrides={
                "date": created.strftime("%Y-%m-%d"),
                "url": url,
                "content_type": "reddit_post",
            },
        )

    def _scrape_listings(self):
        """Scrape posts from listing endpoints (top/hot/new)."""
        count = 0
        for endpoint in LISTING_ENDPOINTS:
            url = REDDIT_JSON_BASE + endpoint
            self.logger.info(f"Fetching: {endpoint}")
            posts = self._fetch_listing(url)
            self.logger.info(f"  Got {len(posts)} posts")

            for post_data in posts:
                result = self._process_post(post_data)
                if result is None:
                    continue
                post_id, doc = result
                filename = f"{post_id}.json"
                if not self.file_exists(filename):
                    self.save_json(doc, filename)
                    count += 1

        return count

    def _scrape_search(self):
        """Scrape posts matching search queries via JSON search endpoint."""
        count = 0
        for query in REDDIT_SEARCH_QUERIES:
            url = f"{REDDIT_JSON_BASE}/search.json?q={query}&restrict_sr=1&sort=relevance&t=year"
            self.logger.info(f"Searching: '{query}'")
            posts = self._fetch_listing(url, max_pages=3)
            self.logger.info(f"  Got {len(posts)} posts")

            for post_data in posts:
                result = self._process_post(post_data)
                if result is None:
                    continue
                post_id, doc = result
                filename = f"{post_id}.json"
                if not self.file_exists(filename):
                    self.save_json(doc, filename)
                    count += 1

        return count

    def run(self):
        self.ensure_output_dir()

        # Load already-seen IDs from existing files
        if self.output_dir.exists():
            for f in self.output_dir.glob("*.json"):
                self.seen_ids.add(f.stem)
            if self.seen_ids:
                self.logger.info(f"Found {len(self.seen_ids)} existing posts, will skip them")

        total = 0
        total += self._scrape_listings()
        total += self._scrape_search()

        self.logger.info(f"Reddit scraping complete: {total} new posts saved, {len(self.seen_ids)} total unique posts")


if __name__ == "__main__":
    scraper = RedditScraper()
    scraper.run()
