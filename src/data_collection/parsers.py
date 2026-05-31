"""Shim — parsers moved to src.data_collection.sofascore.parsers."""
from src.data_collection.sofascore.parsers import (  # noqa: F401
    FTA_TO_POSS_FACTOR,
    PERIOD_MAP,
    SCORE_KEY_MAP,
    _calc_pace,
    _calc_possessions,
    build_metrics,
    parse_full_game,
    parse_period,
    parse_shot,
    safe_int,
)

__all__ = [
    "FTA_TO_POSS_FACTOR", "PERIOD_MAP", "SCORE_KEY_MAP",
    "_calc_pace", "_calc_possessions",
    "build_metrics", "parse_full_game", "parse_period", "parse_shot", "safe_int",
]
