"""Shim — FlashscoreCollector moved to src.data_collection.flashscore.collector."""
from src.data_collection.flashscore.collector import FlashscoreCollector  # noqa: F401

__all__ = ["FlashscoreCollector"]

if __name__ == "__main__":
    import logging
    import sys

    from src.data_collection.flashscore.collector import _parse_args
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)
    args = _parse_args()
    FlashscoreCollector(leagues=args.leagues, dry_run=args.dry_run).run_sync()
