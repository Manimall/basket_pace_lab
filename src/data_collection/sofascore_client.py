"""Shim — SofascoreClient moved to src.data_collection.sofascore.client."""
from src.data_collection.sofascore.client import SofascoreClient  # noqa: F401
from src.data_collection.sofascore.parsers import (  # noqa: F401
    FTA_TO_POSS_FACTOR,
    PERIOD_MAP,
    _calc_pace,
    _calc_possessions,
)

__all__ = [
    "SofascoreClient",
    "FTA_TO_POSS_FACTOR", "PERIOD_MAP", "_calc_pace", "_calc_possessions",
]
