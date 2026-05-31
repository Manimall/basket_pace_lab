"""Tests for src.data_collection.flashscore.odds_db.

I/O-free: the async session factory is mocked, so no real Postgres is needed.
Covers OddsTarget construction, the DEFAULT_LEAGUE fallback for NULL
tournament_name, and the save_odds UPDATE parametrisation.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.data_collection.flashscore.odds_db import (
    DEFAULT_LEAGUE,
    OddsTarget,
    load_odds_targets,
    save_odds,
)


def _make_session_factory(rows: list[dict[str, Any]]) -> tuple[Any, MagicMock]:
    """Build a mock async session factory yielding ``rows`` from execute().

    Returns:
        Tuple of (session_factory_callable, execute_mock) so a test can both
        drive the query result and assert on the SQL/params passed to execute.
    """
    execute_mock = AsyncMock()
    result = MagicMock()
    result.mappings.return_value.all.return_value = rows
    execute_mock.return_value = result

    db = MagicMock()
    db.execute = execute_mock
    # async context manager for `async with sf() as db`
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    # nested `async with db.begin()`
    begin_ctx = MagicMock()
    begin_ctx.__aenter__ = AsyncMock(return_value=None)
    begin_ctx.__aexit__ = AsyncMock(return_value=False)
    db.begin = MagicMock(return_value=begin_ctx)

    def factory() -> Any:
        return db

    return factory, execute_mock


def test_load_targets_maps_rows_to_dataclass() -> None:
    rows = [
        {
            "id": 1, "flashscore_id": "fs_abc", "tournament_name": "LegaA",
            "match_date": date(2026, 1, 15), "home_team": "Milano", "away_team": "Roma",
        },
    ]
    sf, _ = _make_session_factory(rows)
    targets = asyncio.run(load_odds_targets(sf, leagues=["LegaA"], limit=10))

    assert len(targets) == 1
    t = targets[0]
    assert isinstance(t, OddsTarget)
    assert t.match_id == 1
    assert t.flashscore_id == "fs_abc"
    assert t.league == "LegaA"
    assert t.home_team == "Milano"
    assert t.match_date == date(2026, 1, 15)


def test_load_targets_null_league_falls_back_to_default() -> None:
    rows = [
        {
            "id": 2, "flashscore_id": "fs_xyz", "tournament_name": None,
            "match_date": date(2026, 2, 1), "home_team": "A", "away_team": "B",
        },
    ]
    sf, _ = _make_session_factory(rows)
    targets = asyncio.run(load_odds_targets(sf))
    assert targets[0].league == DEFAULT_LEAGUE


def test_load_targets_passes_default_league_param_when_filtering() -> None:
    sf, execute_mock = _make_session_factory([])
    asyncio.run(load_odds_targets(sf, leagues=["NBA"]))

    # Assert the bound params include the DEFAULT_LEAGUE substitution.
    params = execute_mock.call_args.args[1]
    assert params["leagues"] == ["NBA"]
    assert params["default_league"] == DEFAULT_LEAGUE


def test_save_odds_issues_update_with_correct_params() -> None:
    sf, execute_mock = _make_session_factory([])
    odds = SimpleNamespace(
        total_close=185.5,
        total_open=184.0,
        bookmaker="bet365",
        scraped_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    asyncio.run(save_odds(sf, match_id=42, odds=odds))

    params = execute_mock.call_args.args[1]
    assert params["close"] == 185.5
    assert params["open"] == 184.0
    assert params["source"] == "bet365"
    assert params["mid"] == 42
