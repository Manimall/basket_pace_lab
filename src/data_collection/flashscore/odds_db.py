"""Shared DB helpers for odds enrichers.

Single source of truth for:
  - OddsTarget dataclass (shared between OddsEnricher and DirectOddsEnricher)
  - load_odds_targets()  — query matches needing a total_line
  - save_odds()          — persist scraped O/U line

Both enrichers previously duplicated this logic verbatim. Extracted here
to enforce DRY and keep each enricher under 250 lines.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import text

from src.data_collection.flashscore.odds import OddsRow

log = logging.getLogger(__name__)

# Project-wide convention: NULL tournament_name is treated as this league.
# Matches the COALESCE(tournament_name, DEFAULT_LEAGUE) pattern in SQL.
DEFAULT_LEAGUE: str = "NBA"


@dataclass(frozen=True)
class OddsTarget:
    """A finished match that needs its O/U total line populated."""

    match_id:      int
    flashscore_id: str
    league:        str
    home_team:     str
    away_team:     str
    match_date:    date


async def load_odds_targets(
    sf: Any,
    leagues: list[str] | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> list[OddsTarget]:
    """Load finished matches with a flashscore_id but no total_line yet.

    Args:
        sf:      Async session factory.
        leagues: Optional whitelist of ``tournament_name`` values.
                 Uses ``COALESCE(tournament_name, DEFAULT_LEAGUE)`` so that
                 ``leagues=["NBA"]`` correctly matches rows with NULL name.
        limit:   Maximum number of rows to return (None = no cap).
        offset:  Row offset for parallel processing chunks (None = 0).

    Returns:
        OddsTarget records ordered by scheduled_at DESC.
    """
    conditions = [
        "m.flashscore_id IS NOT NULL",
        "m.total_line IS NULL",
        "m.home_score_final IS NOT NULL",
    ]
    params: dict[str, Any] = {}
    if leagues:
        conditions.append(
            "COALESCE(m.tournament_name, :default_league) = ANY(:leagues)"
        )
        params["leagues"] = leagues
        params["default_league"] = DEFAULT_LEAGUE

    where = " AND ".join(conditions)
    lim   = f"LIMIT {int(limit)}"   if limit  else ""
    off   = f"OFFSET {int(offset)}" if offset else ""
    sql   = f"""
        SELECT m.id, m.flashscore_id, m.tournament_name,
               m.scheduled_at::date AS match_date,
               ht.name AS home_team, at.name AS away_team
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE {where}
        ORDER BY m.scheduled_at DESC
        {lim}
        {off}
    """
    async with sf() as db:
        rows = (await db.execute(text(sql), params)).mappings().all()

    targets = [
        OddsTarget(
            match_id      = int(r["id"]),
            flashscore_id = str(r["flashscore_id"]),
            league        = r["tournament_name"] or DEFAULT_LEAGUE,
            match_date    = r["match_date"],
            home_team     = r["home_team"],
            away_team     = r["away_team"],
        )
        for r in rows
    ]
    log.debug("load_odds_targets: %d rows (limit=%s, offset=%s)", len(targets), limit, offset)
    return targets


async def save_odds(sf: Any, match_id: int, odds: OddsRow) -> None:
    """Persist scraped O/U closing line to the matches table.

    Args:
        sf:       Async session factory.
        match_id: Primary key of the row to update.
        odds:     Scraped odds data from Flashscore.
    """
    async with sf() as db:
        async with db.begin():
            await db.execute(
                text(
                    "UPDATE matches SET "
                    "  total_line            = :close, "
                    "  total_line_open       = :open, "
                    "  total_line_source     = :source, "
                    "  total_line_scraped_at = :scraped_at "
                    "WHERE id = :mid"
                ),
                {
                    "close":      odds.total_close,
                    "open":       odds.total_open,
                    "source":     odds.bookmaker,
                    "scraped_at": odds.scraped_at,
                    "mid":        match_id,
                },
            )
