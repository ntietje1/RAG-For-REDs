import argparse
import sys

from scrapers.lolalytics import LolalyticsScraper
from scrapers.patch_notes import PatchNotesScraper
from scrapers.reddit_scraper import RedditScraper
from scrapers.wiki import WikiScraper

SCRAPERS = {
    "patch_notes": PatchNotesScraper,
    "wiki": WikiScraper,
    "reddit": RedditScraper,
    "stats": LolalyticsScraper,
}


def main():
    parser = argparse.ArgumentParser(description="Run League of Legends data scrapers")
    parser.add_argument(
        "--scraper",
        choices=list(SCRAPERS.keys()),
        help="Run a specific scraper",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all scrapers",
    )

    args = parser.parse_args()

    if not args.scraper and not args.all:
        parser.print_help()
        sys.exit(1)

    if args.all:
        to_run = list(SCRAPERS.keys())
    else:
        to_run = [args.scraper]

    for name in to_run:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}\n")

        scraper = SCRAPERS[name]()
        try:
            scraper.run()
        except KeyboardInterrupt:
            print(f"\nInterrupted while running {name}")
            sys.exit(1)
        except Exception as e:
            print(f"Error running {name}: {e}")
            if not args.all:
                sys.exit(1)


if __name__ == "__main__":
    main()
