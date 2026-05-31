"""DB access for the Sofascore ID finder: load fs_-id rows, patch external_id."""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

from src.data_collection.flashscore.norm import _norm
from src.data_collection.sofascore.id_finder.models import DbRow

log = logging.getLogger(__name__)

_LOAD_FS_ROWS_SQL: str = """
    SELECT m.id, m.external_id, m.scheduled_at::date AS match_date,
           ht.name AS home_team, at.name AS away_team
    FROM matches m
    JOIN teams ht ON ht.id = m.home_team_id
    JOIN teams at ON at.id = m.away_team_id
    WHERE m.tournament_name = :tname
      AND m.external_id LIKE 'fs_%'
      AND m.home_score_final IS NOT NULL
    ORDER BY m.scheduled_at DESC
"""

_PATCH_SQL: str = "UPDATE matches SET external_id = :eid WHERE id = :mid"


async def load_fs_rows(sf: Any, db_tournament_name: str) -> list[DbRow]:
    """Load matches with Flashscore IDs that still need a Sofascore event ID."""
    async with sf() as db:
        rows = (
            await db.execute(text(_LOAD_FS_ROWS_SQL), {"tname": db_tournament_name})
        ).mappings().all()
    return [
        DbRow(
            match_id   = int(r["id"]),
            ext_id     = r["external_id"],
            match_date = r["match_date"],
            home_raw   = r["home_team"],
            away_raw   = r["away_team"],
            home_norm  = _norm(r["home_team"]),
            away_norm  = _norm(r["away_team"]),
        )
        for r in rows
    ]


async def patch_external_id(sf: Any, match_id: int, new_ext_id: str) -> None:
    """Overwrite a row's external_id with the resolved numeric Sofascore ID."""
    async with sf() as db:
        async with db.begin():
            await db.execute(text(_PATCH_SQL), {"eid": new_ext_id, "mid": match_id})
