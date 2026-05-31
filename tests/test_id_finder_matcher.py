"""Tests for the pure matcher of src.data_collection.sofascore.id_finder.

I/O-free: no HTTP, no DB. Covers date-window filtering, threshold gating, and
the unmatched-hint fallback.
"""
from __future__ import annotations

from datetime import date

from src.data_collection.flashscore.norm import _norm
from src.data_collection.sofascore.id_finder.matcher import (
    closest_hint,
    find_best_match,
    score_pair,
)
from src.data_collection.sofascore.id_finder.models import DbRow, SofaEvent


def _db_row(home: str, away: str, d: date) -> DbRow:
    return DbRow(
        match_id=1, ext_id="fs_x", match_date=d,
        home_raw=home, away_raw=away, home_norm=_norm(home), away_norm=_norm(away),
    )


def _event(eid: int, home: str, away: str, d: date) -> SofaEvent:
    return SofaEvent(
        event_id=eid, event_date=d,
        home_raw=home, away_raw=away, home_norm=_norm(home), away_norm=_norm(away),
    )


def test_exact_name_and_date_matches() -> None:
    row = _db_row("Milano", "Roma", date(2026, 1, 15))
    ev  = _event(999, "Milano", "Roma", date(2026, 1, 15))
    assert find_best_match(row, [ev], threshold=0.5) is ev


def test_outside_date_window_is_skipped() -> None:
    row = _db_row("Milano", "Roma", date(2026, 1, 15))
    ev  = _event(999, "Milano", "Roma", date(2026, 1, 20))  # 5 days off
    assert find_best_match(row, [ev], threshold=0.5) is None


def test_one_day_offset_still_matches() -> None:
    row = _db_row("Milano", "Roma", date(2026, 1, 15))
    ev  = _event(999, "Milano", "Roma", date(2026, 1, 16))
    assert find_best_match(row, [ev], threshold=0.5) is ev


def test_below_threshold_returns_none() -> None:
    row = _db_row("Milano", "Roma", date(2026, 1, 15))
    ev  = _event(999, "Berlin", "Paris", date(2026, 1, 15))
    assert find_best_match(row, [ev], threshold=0.5) is None


def test_best_of_several_candidates_chosen() -> None:
    row  = _db_row("Real Madrid", "Barcelona", date(2026, 1, 15))
    weak = _event(1, "Real Madrid", "Valencia", date(2026, 1, 15))
    good = _event(2, "Real Madrid", "Barcelona", date(2026, 1, 15))
    assert find_best_match(row, [weak, good], threshold=0.5) is good


def test_score_pair_symmetric_range() -> None:
    row = _db_row("Milano", "Roma", date(2026, 1, 15))
    ev  = _event(1, "Milano", "Roma", date(2026, 1, 15))
    assert 0.0 <= score_pair(row, ev) <= 1.0


def test_closest_hint_reports_no_candidates() -> None:
    row = _db_row("Milano", "Roma", date(2026, 1, 15))
    assert "no candidates" in closest_hint(row, [])
