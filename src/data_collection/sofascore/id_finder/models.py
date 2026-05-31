"""Typed data structures for the Sofascore ID finder."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class SofaEvent:
    """One finished Sofascore event, normalised for fuzzy matching."""

    event_id:   int
    event_date: date
    home_raw:   str
    away_raw:   str
    home_norm:  str
    away_norm:  str


@dataclass(frozen=True)
class DbRow:
    """A DB match still carrying a Flashscore ``fs_…`` external_id."""

    match_id:   int
    ext_id:     str
    match_date: date
    home_raw:   str
    away_raw:   str
    home_norm:  str
    away_norm:  str
