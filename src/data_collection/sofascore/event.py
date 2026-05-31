"""
Sofascore per-event fetch, parse, and persist helpers.

Used by MassScheduler. Isolated here to keep collector.py under 250 lines.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from playwright.async_api import Page
from sqlalchemy import text

from src.database import crud
from src.database.engine import SessionFactory
from src.database.crud import QuarterStatRow
from src.database.models import MatchStatus, SeasonType
from src.data_collection.constants import SOFASCORE_API_HEADERS
from src.data_collection.sofascore.parsers import (
    PERIOD_MAP,
    SCORE_KEY_MAP,
    parse_full_game,
    parse_period,
    safe_int,
)

log = logging.getLogger(__name__)

_STATUS_MAP = {
    "finished":   MatchStatus.FINISHED,
    "inprogress": MatchStatus.LIVE,
    "notstarted": MatchStatus.SCHEDULED,
}


async def api_get(page: Page, path: str) -> dict[str, Any] | None:
    """GET a Sofascore API path via the Playwright page request.

    Args:
        page: Active Playwright page (carries the warmed-up session).
        path: API path appended to the Sofascore v1 base URL.

    Returns:
        Parsed JSON dict, or None on non-OK status / network error.
    """
    url = f"https://www.sofascore.com/api/v1{path}"
    try:
        resp = await page.request.get(url, headers=SOFASCORE_API_HEADERS, timeout=15_000)
        if not resp.ok:
            return None
        return await resp.json()
    except Exception:
        return None


async def fetch_event_ids(
    page: Page, tournament_id: int, season_id: int, max_pages: int = 50
) -> list[int]:
    """Page through a season's event list and collect finished event ids.

    Args:
        page: Active Playwright page.
        tournament_id: Sofascore unique-tournament id.
        season_id: Sofascore season id.
        max_pages: Hard cap on pages to request.

    Returns:
        List of finished-event ids (possibly empty).
    """
    ids: list[int] = []
    for p in range(max_pages):
        data = await api_get(
            page,
            f"/unique-tournament/{tournament_id}/season/{season_id}/events/last/{p}",
        )
        if not data:
            break
        events = data.get("events", [])
        ids.extend(e["id"] for e in events if e.get("status", {}).get("type") == "finished")
        if not data.get("hasNextPage", True) or not events:
            break
    return ids


async def load_existing_external_ids(session_factory: SessionFactory) -> set[str]:
    """Return external_ids of matches that already have quarter_stats."""
    async with session_factory() as db:
        rows = await db.execute(text("""
            SELECT DISTINCT m.external_id
            FROM matches m
            JOIN quarter_stats qs ON qs.match_id = m.id
            WHERE m.external_id IS NOT NULL
        """))
        return {r[0] for r in rows.all()}


async def process_event(
    page: Page,
    event_id: int,
    tournament_id: int,
    tournament_name: str,
    session_factory: SessionFactory,
    existing_ids: set[str],
) -> str:
    """Fetch, parse, and persist one event. Returns 'ok'|'skip'|'exists'|'error'."""
    ext_id = str(event_id)
    if ext_id in existing_ids:
        return "exists"

    event = await api_get(page, f"/event/{event_id}")
    if not event:
        return "skip"
    ev = event.get("event", event)
    if not ev or not ev.get("homeTeam"):
        return "skip"

    home_raw       = ev["homeTeam"]
    away_raw       = ev["awayTeam"]
    home_score_raw = ev.get("homeScore", {})
    away_score_raw = ev.get("awayScore", {})

    stats = await api_get(page, f"/event/{event_id}/statistics")
    if not stats:
        return "skip"

    rows: list[QuarterStatRow] = []
    for period_data in stats.get("statistics", []):
        label = period_data.get("period", "").upper()
        if label == "ALL":
            continue
        score_key = SCORE_KEY_MAP.get(label)
        h_score   = safe_int(home_score_raw.get(score_key)) if score_key else None
        a_score   = safe_int(away_score_raw.get(score_key)) if score_key else None
        row       = parse_period(label, period_data, h_score, a_score)
        if row:
            rows.append(row)

    has_quarter_breakdown = True
    if not rows:
        all_data = next(
            (pd for pd in stats.get("statistics", []) if pd.get("period", "").upper() == "ALL"),
            None,
        )
        if all_data:
            h_score  = safe_int(home_score_raw.get("current"))
            a_score  = safe_int(away_score_raw.get("current"))
            fallback = parse_full_game(all_data, h_score, a_score)
            if fallback:
                rows = [fallback]
                has_quarter_breakdown = False

    if not rows:
        return "skip"

    def _sum_q(score: dict) -> int | None:
        vals  = [score.get(f"period{i}") for i in range(1, 5)]
        total = sum(int(v) for v in vals if v is not None)
        return total if total > 0 else None

    went_ot      = bool(home_score_raw.get("overtime") or away_score_raw.get("overtime"))
    status       = _STATUS_MAP.get(ev.get("status", {}).get("type", ""), MatchStatus.FINISHED)
    ts           = ev.get("startTimestamp")
    scheduled_at = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(tz=timezone.utc)
    season_name  = ev.get("season", {}).get("name", "unknown")

    async with session_factory() as db:
        async with db.begin():
            home_team = await crud.get_or_create_team(
                db, str(home_raw["id"]),
                home_raw.get("name", "?"),
                home_raw.get("shortName", home_raw.get("name", "?"))[:8],
            )
            away_team = await crud.get_or_create_team(
                db, str(away_raw["id"]),
                away_raw.get("name", "?"),
                away_raw.get("shortName", away_raw.get("name", "?"))[:8],
            )
            match = await crud.upsert_match(
                db, ext_id,
                home_team.id, away_team.id,
                scheduled_at, season_name,
                status=status,
                season_type=SeasonType.REGULAR,
                tournament_id=tournament_id,
                tournament_name=tournament_name,
                home_score_final=safe_int(home_score_raw.get("current")),
                away_score_final=safe_int(away_score_raw.get("current")),
                home_score_regulation=_sum_q(home_score_raw),
                away_score_regulation=_sum_q(away_score_raw),
                went_to_overtime=went_ot,
                has_quarter_breakdown=has_quarter_breakdown,
            )
            await crud.save_quarter_stats(db, match.id, rows)
            existing_ids.add(ext_id)

    return "ok"
