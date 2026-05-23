"""Shim — MassScheduler moved to src.data_collection.sofascore.collector."""
from src.data_collection.sofascore.collector import MassScheduler  # noqa: F401

__all__ = ["MassScheduler"]

if __name__ == "__main__":
    import logging
    import sys
    from src.data_collection.sofascore.collector import _parse_args
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)
    opts = _parse_args()
    MassScheduler(league_filter=opts["leagues"], season_filter=opts["seasons"], dry_run=opts["dry_run"]).run_sync()
