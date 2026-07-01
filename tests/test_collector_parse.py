"""Tests for the pure parsers in collector_parse.py."""
from __future__ import annotations

from datetime import date, datetime, timezone

from src.data_collection.flashscore.collector_parse import abbrev, parse_fs_datetime


def test_abbrev_multiword() -> None:
    assert abbrev("Los Angeles Lakers") == "LAL"


def test_abbrev_two_words() -> None:
    assert abbrev("Real Madrid") == "RM"


def test_abbrev_single_word() -> None:
    assert abbrev("Barcelona") == "BAR"


def test_parse_full_date_with_time() -> None:
    dt = parse_fs_datetime("15.01.2026 19:30")
    assert dt == datetime(2026, 1, 15, 19, 30, tzinfo=timezone.utc)


def test_parse_full_date_without_time_defaults_to_noon() -> None:
    dt = parse_fs_datetime("15.01.2026")
    assert dt == datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


def test_parse_invalid_returns_none() -> None:
    assert parse_fs_datetime("not a date") is None


def test_parse_impossible_date_returns_none() -> None:
    assert parse_fs_datetime("31.02.2026") is None


def test_parse_yearless_resolves_to_recent_past() -> None:
    """A bare DD.MM. resolves to a past date within the look-back window."""
    dt = parse_fs_datetime("01.01. 18:00")
    assert dt is not None
    assert dt.month == 1 and dt.day == 1
    assert dt.date() <= date.today()


def test_parse_kbl_season_crosses_year_boundary() -> None:
    """KBL 25/26 spans Oct 2025–Apr 2026; full dates keep each side's own year.

    Flashscore results pages carry the explicit year for finished-season
    backfill, so a Dec-2025 and a Jan-2026 fixture must NOT collapse onto one
    year — this guards the Korean cross-boundary calendar.
    """
    autumn = parse_fs_datetime("18.10.2025 19:00")   # early season, 2025
    winter = parse_fs_datetime("05.01.2026 16:00")    # crosses into 2026
    spring = parse_fs_datetime("12.04.2026 15:00")    # late season, 2026
    assert autumn == datetime(2025, 10, 18, 19, 0, tzinfo=timezone.utc)
    assert winter == datetime(2026, 1, 5, 16, 0, tzinfo=timezone.utc)
    assert spring == datetime(2026, 4, 12, 15, 0, tzinfo=timezone.utc)
