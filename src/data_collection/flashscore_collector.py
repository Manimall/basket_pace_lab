"""Shim — FlashscoreCollector moved to src.data_collection.flashscore.collector."""
from src.data_collection.flashscore.collector import FlashscoreCollector  # noqa: F401

__all__ = ["FlashscoreCollector"]

if __name__ == "__main__":
    import logging
    import sys
    from src.data_collection.flashscore.collector import _parse_args
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)
    opts = _parse_args()
    FlashscoreCollector(leagues=opts["leagues"], dry_run=opts["dry_run"]).run_sync()
