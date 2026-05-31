"""Shim — content moved to src.data_collection.flashscore.db."""
from src.data_collection.flashscore.db import (  # noqa: F401
    SEASON_FILTERS,
    ensure_flashscore_id_column,
    load_db_matches,
    save_enriched,
)

__all__ = ["SEASON_FILTERS", "ensure_flashscore_id_column", "load_db_matches", "save_enriched"]
