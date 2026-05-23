"""
Shim — FlashscoreEnricher moved to src.data_collection.flashscore.enricher.

Re-exports _norm, _sim, _NAME_ALIASES for backward compatibility with tests.
"""
from src.data_collection.flashscore.enricher import FlashscoreEnricher  # noqa: F401
from src.data_collection.flashscore.norm import (  # noqa: F401
    _NAME_ALIASES, _norm, _sim,
)

__all__ = ["FlashscoreEnricher", "_NAME_ALIASES", "_norm", "_sim"]

if __name__ == "__main__":
    import logging
    import sys
    from src.data_collection.flashscore.enricher import _parse_args
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)
    opts = _parse_args()
    FlashscoreEnricher(
        leagues=opts["leagues"], season_codes=opts["seasons"],
        limit=opts["limit"], dry_run=opts["dry_run"],
    ).run_sync()
