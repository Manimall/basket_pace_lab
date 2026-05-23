"""Shim — content moved to src.data_collection.flashscore.norm."""
from src.data_collection.flashscore.norm import (  # noqa: F401
    DbMatch, FsMatch, QScore, _NAME_ALIASES, _norm, _sim,
)

__all__ = ["DbMatch", "FsMatch", "QScore", "_NAME_ALIASES", "_norm", "_sim"]
