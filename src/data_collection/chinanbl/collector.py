"""
ChinaNBL collector — wraps the scraper into a BaseCollector.

Usage:
    python -m src.data_collection.chinanbl.collector
    python -m src.data_collection.chinanbl.collector --no-stats
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from sqlalchemy.ext.asyncio import AsyncSession as DbSession

from src.data_collection.base import BaseCollector
from src.data_collection.chinanbl.page import (
    SEASONS, TOURNAMENT_ID, TOURNAMENT_NAME,
    _SofaApiPage, _safe_int,
    extract_game_stats_row, extract_ot_rows, extract_quarter_rows,
)
from src.data_collection.constants import DEFAULT_USER_AGENT
from src.database import crud
from src.database.engine import dispose_engine, get_session_factory
from src.database.models import MatchStatus, SeasonType

log = logging.getLogger(__name__)


async def _process_event(
    event: dict[str, Any],
    season_name: str,
    season_type: SeasonType,
    stats_data: dict[str, Any] | None,
    db: DbSession,
) -> str:
    if event.get("status", {}).get("type") != "finished":
        return "skip"

    event_id = event["id"]
    home_raw = event["homeTeam"]
    away_raw = event["awayTeam"]

    home_team = await crud.get_or_create_team(
        db, external_id=f"ss_{home_raw['id']}",
        name=home_raw.get("name", "Unknown"),
        abbreviation=(home_raw.get("shortName") or home_raw.get("name", "?"))[:8],
    )
    away_team = await crud.get_or_create_team(
        db, external_id=f"ss_{away_raw['id']}",
        name=away_raw.get("name", "Unknown"),
        abbreviation=(away_raw.get("shortName") or away_raw.get("name", "?"))[:8],
    )

    hs  = event.get("homeScore", {})
    as_ = event.get("awayScore", {})
    home_final = _safe_int(hs.get("current"))
    away_final = _safe_int(as_.get("current"))
    home_reg   = _safe_int(hs.get("normaltime"))
    away_reg   = _safe_int(as_.get("normaltime"))
    went_ot    = bool(home_final is not None and home_reg is not None and home_final != home_reg)
    ot_count   = sum(1 for q in range(5, 10) if _safe_int(hs.get(f"period{q}")) is not None)

    start_ts = event.get("startTimestamp")
    scheduled_at = (
        datetime.fromtimestamp(start_ts, tz=timezone.utc)
        if start_ts else datetime.now(tz=timezone.utc)
    )

    q_rows   = extract_quarter_rows(event)
    ot_rows  = extract_ot_rows(event)
    game_row = extract_game_stats_row(stats_data) if stats_data else None
    all_rows = q_rows + ot_rows + ([game_row] if game_row else [])

    match = await crud.upsert_match(
        db, external_id=f"ss_{event_id}",
        home_team_id=home_team.id, away_team_id=away_team.id,
        scheduled_at=scheduled_at, season=season_name,
        status=MatchStatus.FINISHED, season_type=season_type,
        tournament_name=TOURNAMENT_NAME,
        home_score_final=home_final, away_score_final=away_final,
        home_score_regulation=home_reg, away_score_regulation=away_reg,
        went_to_overtime=went_ot, overtime_periods_count=ot_count,
        has_quarter_breakdown=(len(q_rows) == 4),
    )
    await crud.save_quarter_stats(db, match.id, all_rows)
    return "ok"


class ChinaNBLCollector(BaseCollector):
    def __init__(self, session_factory: Any = None, fetch_stats: bool = True) -> None:
        self._sf          = session_factory or get_session_factory()
        self._fetch_stats = fetch_stats

    async def run(self) -> None:
        """Scrape ChinaNBL events for the configured seasons and persist them."""
        async with async_playwright() as pw:
            browser: Browser = await pw.chromium.launch(headless=True)
            ctx: BrowserContext = await browser.new_context(
                user_agent=DEFAULT_USER_AGENT,
                viewport={"width": 1280, "height": 800},
                locale="en-US",
            )
            page: Page = await ctx.new_page()
            api = _SofaApiPage(page)

            for season_id, (season_name, season_type) in SEASONS.items():
                log.info("━" * 60)
                log.info("[ChinaNBL][%s]  Fetching event list…", season_name)
                all_events = await api.fetch_all_events(season_id)
                finished = [e for e in all_events if e.get("status", {}).get("type") == "finished"]
                log.info("[ChinaNBL][%s]  %d finished events found", season_name, len(finished))

                ok = skip = err = 0
                for i, event in enumerate(finished, 1):
                    eid = event["id"]
                    stats = await api.fetch_stats(eid) if self._fetch_stats else None
                    try:
                        async with self._sf() as db:
                            result = await _process_event(event, season_name, season_type, stats, db)
                            await db.commit()
                        if result == "ok":
                            ok += 1
                        else:
                            skip += 1
                    except Exception as e:
                        log.error("  event %d failed: %s", eid, e)
                        err += 1

                    if i % 20 == 0 or i == len(finished):
                        log.info("[ChinaNBL][%s]  %d/%d | ok=%d skip=%d err=%d",
                                 season_name, i, len(finished), ok, skip, err)

                log.info("[ChinaNBL][%s]  DONE  ok=%d  skip=%d  err=%d", season_name, ok, skip, err)

            await browser.close()

        await dispose_engine()
        log.info("Collection complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Collect ChinaNBL from Sofascore")
    parser.add_argument("--no-stats", action="store_true", help="Skip per-match statistics (scores only)")
    args = parser.parse_args()
    ChinaNBLCollector(fetch_stats=not args.no_stats).run_sync()
