"""
DB access helpers for FlashscoreEnricher.

Isolated here to keep enricher.py under 250 lines.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

from src.database.crud import QuarterStatRow, save_quarter_stats
from src.data_collection.flashscore.norm import DbMatch

log = logging.getLogger(__name__)

SEASON_FILTERS: dict[str, str] = {
    "2526": "%25/26%",
    "2425": "%24/25%",
    "2324": "%23/24%",
}


async def save_enriched(
    db_session_factory: Any,
    match_id: int,
    fs_id: str,
    quarter_rows: list[QuarterStatRow],
) -> None:
    async with db_session_factory() as db:
        async with db.begin():
            await save_quarter_stats(db, match_id, quarter_rows)
            await db.execute(
                text(
                    "UPDATE matches "
                    "SET has_quarter_breakdown = TRUE, flashscore_id = :fid "
                    "WHERE id = :mid"
                ),
                {"fid": fs_id, "mid": match_id},
            )


async def load_db_matches(
    db_session_factory: Any,
    leagues: list[str] | None,
    season_codes: list[str] | None,
    limit: int | None,
) -> list[DbMatch]:
    conditions = ["m.has_quarter_breakdown = FALSE", "m.home_score_final IS NOT NULL"]
    params: dict = {}

    if leagues:
        conditions.append("m.tournament_name = ANY(:leagues)")
        params["leagues"] = leagues

    if season_codes:
        season_likes = [SEASON_FILTERS[c] for c in season_codes if c in SEASON_FILTERS]
        if season_likes:
            season_clauses = " OR ".join(
                f"m.season LIKE :season_{i}" for i in range(len(season_likes))
            )
            conditions.append(f"({season_clauses})")
            for i, pat in enumerate(season_likes):
                params[f"season_{i}"] = pat

    where = " AND ".join(conditions)
    lim   = f"LIMIT {int(limit)}" if limit else ""

    sql = f"""
        SELECT m.id, m.external_id, m.tournament_name,
               m.scheduled_at::date AS match_date,
               ht.name AS home_team, at.name AS away_team
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE {where}
        ORDER BY m.scheduled_at DESC
        {lim}
    """

    async with db_session_factory() as db:
        rows = (await db.execute(text(sql), params)).mappings().all()

    return [
        DbMatch(
            match_id=r["id"],
            ext_id=r["external_id"] or "",
            league=r["tournament_name"] or "",
            match_date=r["match_date"],
            home_team=r["home_team"],
            away_team=r["away_team"],
        )
        for r in rows
    ]


async def ensure_flashscore_id_column(db_session_factory: Any) -> None:
    async with db_session_factory() as db:
        async with db.begin():
            await db.execute(text(
                "ALTER TABLE matches ADD COLUMN IF NOT EXISTS flashscore_id VARCHAR(16);"
            ))
