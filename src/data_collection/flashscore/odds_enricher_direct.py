"""
Direct-URL odds enricher — bypasses results-page click navigation.

Constructs each match URL directly from flashscore_id already stored in DB:
    https://www.flashscore.com/match/{flashscore_id}/

Removes the DOM-recycling ceiling (~100 visible elements) of the click-based
enricher and lets us cover all records that already have a flashscore_id.

Usage:
    python -m src.data_collection.flashscore.odds_enricher_direct
    python -m src.data_collection.flashscore.odds_enricher_direct --leagues NBA
    python -m src.data_collection.flashscore.odds_enricher_direct --limit 500 --offset 0
    python -m src.data_collection.flashscore.odds_enricher_direct --dry-run

Parallel chunks (3 processes):
    python -m ... --leagues NBA --limit 400 --offset 0   &
    python -m ... --leagues NBA --limit 400 --offset 400 &
    python -m ... --leagues NBA --limit 400 --offset 800 &
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

from playwright.async_api import async_playwright
from sqlalchemy import text

from src.config import settings
from src.data_collection.base import BaseCollector
from src.data_collection.constants import FLASHSCORE_BASE_URL
from src.data_collection.flashscore.browser import make_flashscore_session
from src.data_collection.flashscore.odds import OddsRow, scrape_match_odds
from src.database.engine import dispose_engine, get_session_factory

log = logging.getLogger("odds_enricher_direct")


@dataclass
class _Target:
    match_id:      int
    flashscore_id: str
    league:        str
    home_team:     str
    away_team:     str
    match_date:    date


async def _load_targets(
    sf: Any,
    leagues: list[str] | None,
    limit: int | None,
    offset: int | None,
) -> list[_Target]:
    conditions = [
        "m.flashscore_id IS NOT NULL",
        "m.total_line IS NULL",
        "m.home_score_final IS NOT NULL",
    ]
    params: dict[str, Any] = {}
    if leagues:
        conditions.append("COALESCE(m.tournament_name, 'NBA') = ANY(:leagues)")
        params["leagues"] = leagues

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

    return [
        _Target(
            match_id=int(r["id"]),
            flashscore_id=str(r["flashscore_id"]),
            league=r["tournament_name"] or "NBA",
            match_date=r["match_date"],
            home_team=r["home_team"],
            away_team=r["away_team"],
        )
        for r in rows
    ]


async def _save_odds(sf: Any, match_id: int, odds: OddsRow) -> None:
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


class DirectOddsEnricher(BaseCollector):
    def __init__(
        self,
        session_factory: Any = None,
        leagues: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        dry_run: bool = False,
    ) -> None:
        self._sf      = session_factory or get_session_factory()
        self._leagues = leagues
        self._limit   = limit
        self._offset  = offset
        self._dry_run = dry_run

    async def run(self) -> None:
        log.info("DB: %s:%s/%s", settings.db.host, settings.db.port, settings.db.name)

        targets = await _load_targets(
            self._sf, self._leagues, self._limit, self._offset,
        )
        log.info("Targets (flashscore_id set, no total_line): %d", len(targets))
        if not targets:
            log.info("Nothing to enrich.")
            await dispose_engine()
            return

        if self._dry_run:
            for t in targets[:20]:
                log.info(
                    "  [dry] %s | %s vs %s | %s | fs=%s",
                    t.league, t.home_team, t.away_team, t.match_date, t.flashscore_id,
                )
            log.info("  [dry] Total: %d", len(targets))
            return

        ok = noodds = err = 0
        t0 = time.monotonic()

        async with async_playwright() as pw:
            browser, odds_page = await make_flashscore_session(pw)
            try:
                for i, target in enumerate(targets, 1):
                    match_url = f"{FLASHSCORE_BASE_URL}/match/{target.flashscore_id}/"
                    try:
                        odds = await scrape_match_odds(
                            odds_page, target.flashscore_id, match_url,
                        )
                        if odds is None:
                            noodds += 1
                            log.debug(
                                "  noodds [%s] %s vs %s",
                                target.flashscore_id, target.home_team, target.away_team,
                            )
                        else:
                            await _save_odds(self._sf, target.match_id, odds)
                            log.debug(
                                "  ok [%s] %s vs %s → close=%.1f bm=%s",
                                target.flashscore_id, target.home_team, target.away_team,
                                odds.total_close or 0.0, odds.bookmaker,
                            )
                            ok += 1
                    except Exception as exc:
                        log.warning(
                            "  err [%s] %s vs %s → %s",
                            target.flashscore_id, target.home_team, target.away_team,
                            str(exc)[:120],
                        )
                        err += 1

                    if i % 25 == 0 or i == len(targets):
                        elapsed = time.monotonic() - t0
                        rate    = i / elapsed if elapsed > 0 else 0
                        eta_min = (len(targets) - i) / rate / 60 if rate > 0 else 0
                        log.info(
                            "%d/%d | ok=%d noodds=%d err=%d | %.0fs | ETA ~%.0f min",
                            i, len(targets), ok, noodds, err, elapsed, eta_min,
                        )
                    await asyncio.sleep(settings.collector.delay_min)
            finally:
                await browser.close()

        log.info("━" * 60)
        log.info(
            "DONE  ok=%d  noodds=%d  err=%d  |  %.1f min",
            ok, noodds, err, (time.monotonic() - t0) / 60,
        )
        await dispose_engine()


def _parse_args() -> dict[str, Any]:
    args = sys.argv[1:]
    opts: dict[str, Any] = {
        "leagues": None, "limit": None, "offset": None, "dry_run": False,
    }
    i = 0
    while i < len(args):
        if args[i] == "--leagues" and i + 1 < len(args):
            vals: list[str] = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                vals.append(args[i]); i += 1
            opts["leagues"] = vals
        elif args[i] == "--limit" and i + 1 < len(args):
            opts["limit"] = int(args[i + 1]); i += 2
        elif args[i] == "--offset" and i + 1 < len(args):
            opts["offset"] = int(args[i + 1]); i += 2
        elif args[i] == "--dry-run":
            opts["dry_run"] = True; i += 1
        else:
            i += 1
    return opts


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        stream=sys.stdout,
    )
    opts = _parse_args()
    DirectOddsEnricher(
        leagues=opts["leagues"],
        limit=opts["limit"],
        offset=opts["offset"],
        dry_run=opts["dry_run"],
    ).run_sync()
