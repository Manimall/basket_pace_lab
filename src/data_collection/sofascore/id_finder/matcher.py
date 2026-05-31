"""Pure fuzzy-matching logic for the Sofascore ID finder.

I/O-free and independently testable: given a DB row and a list of Sofascore
events, decide the best date+name match. Scoring delegates to the shared
``norm._sim`` so it stays consistent with the Flashscore matcher.
"""
from __future__ import annotations

from src.data_collection.flashscore.norm import _sim
from src.data_collection.sofascore.id_finder.config import (
    DATE_WINDOW_DAYS,
    HINT_WINDOW_DAYS,
)
from src.data_collection.sofascore.id_finder.models import DbRow, SofaEvent

_SIDES: int = 2  # home + away → average of the two similarity scores


def score_pair(db_row: DbRow, sofa_ev: SofaEvent) -> float:
    """Average of home- and away-team name similarity in [0, 1]."""
    home_sim = _sim(db_row.home_norm, sofa_ev.home_norm)
    away_sim = _sim(db_row.away_norm, sofa_ev.away_norm)
    return (home_sim + away_sim) / _SIDES


def _within(db_row: DbRow, ev: SofaEvent, window_days: int) -> bool:
    return ev.event_date is not None and abs((ev.event_date - db_row.match_date).days) <= window_days


def find_best_match(
    db_row: DbRow,
    events: list[SofaEvent],
    threshold: float,
    date_window: int = DATE_WINDOW_DAYS,
) -> SofaEvent | None:
    """Return the best date+name match for ``db_row`` at or above ``threshold``."""
    candidates = [ev for ev in events if _within(db_row, ev, date_window)]
    if not candidates:
        return None
    best = max(candidates, key=lambda ev: score_pair(db_row, ev))
    return best if score_pair(db_row, best) >= threshold else None


def closest_hint(db_row: DbRow, events: list[SofaEvent]) -> str:
    """Human-readable nearest candidate for an unmatched row (for logging)."""
    nearby = sorted(
        (ev for ev in events if _within(db_row, ev, HINT_WINDOW_DAYS)),
        key=lambda ev: -score_pair(db_row, ev),
    )
    if not nearby:
        return f"no candidates in ±{HINT_WINDOW_DAYS} days"
    top = nearby[0]
    return f"{top.home_raw} vs {top.away_raw} (score={score_pair(db_row, top):.2f})"
