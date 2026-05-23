"""
OddsEnricher — populates pre-match Over/Under lines for historical matches.

Strategy:
  1. Load finished matches that have flashscore_id but no total_line.
  2. Group by tournament_name.
  3. For each league, build href index from Flashscore results page (one page load).
  4. Navigate to each match's odds tab, extract Pinnacle (or fallback) line.
  5. UPDATE matches: total_line, total_line_open, total_line_source, total_line_scraped_at.

Usage:
    python -m src.data_collection.flashscore.odds_enricher
    python -m src.data_collection.flashscore.odds_enricher --leagues LegaA PBA_PhilCup
    python -m src.data_collection.flashscore.odds_enricher --limit 50 --dry-run
"""
from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

from playwright.async_api import async_playwright
from sqlalchemy import text

from src.config import settings
from src.data_collection.base import BaseCollector
from src.data_collection.constants import FLASHSCORE_BASE_URL, USER_AGENTS
from src.data_collection.flashscore.odds import OddsRow, build_href_index, scrape_match_odds
from src.data_collection.flashscore.page import LEAGUE_PATHS
from src.database.engine import dispose_engine, get_session_factory

log = logging.getLogger("odds_enricher")


# -----------------------------------------------------------------------
# Internal data structures
# -----------------------------------------------------------------------

@dataclass
class _OddsTarget:
    match_id:      int
    flashscore_id: str
    league:        str
    home_team:     str
    away_team:     str
    match_date:    date


# -----------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------

async def _load_targets(
    sf: Any,
    leagues: list[str] | None,
    limit: int | None,
) -> list[_OddsTarget]:
    """Return finished matches that have a flashscore_id but no total_line yet."""
    conditions = [
        "m.flashscore_id IS NOT NULL",
        "m.total_line IS NULL",
        "m.home_score_final IS NOT NULL",
    ]
    params: dict[str, Any] = {}

    if leagues:
        conditions.append("m.tournament_name = ANY(:leagues)")
        params["leagues"] = leagues

    where = " AND ".join(conditions)
    lim   = f"LIMIT {int(limit)}" if limit else ""

    sql = f"""
        SELECT m.id, m.flashscore_id, m.tournament_name,
               m.scheduled_at::date AS match_date,
               ht.name AS home_team, at.name AS away_team
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE {where}
        ORDER BY m.scheduled_at DESC
        {lim}
    """
    async with sf() as db:
        rows = (await db.execute(text(sql), params)).mappings().all()

    return [
        _OddsTarget(
            match_id=int(r["id"]),
            flashscore_id=str(r["flashscore_id"]),
            league=r["tournament_name"] or "",
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


# -----------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------

class OddsEnricher(BaseCollector):
    def __init__(
        self,
        session_factory: Any = None,
        leagues: list[str] | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> None:
        self._sf      = session_factory or get_session_factory()
        self._leagues = leagues
        self._limit   = limit
        self._dry_run = dry_run

    async def run(self) -> None:
        log.info("DB: %s:%s/%s", settings.db.host, settings.db.port, settings.db.name)

        targets = await _load_targets(self._sf, self._leagues, self._limit)
        log.info("Matches to enrich with odds: %d", len(targets))

        if not targets:
            log.info("Nothing to enrich — all matches already have a total_line.")
            return

        if self._dry_run:
            for t in targets[:20]:
                log.info(
                    "  [dry] %s | %s vs %s | %s | fs_id=%s",
                    t.league, t.home_team, t.away_team, t.match_date, t.flashscore_id,
                )
            log.info("  [dry] Total: %d matches", len(targets))
            return

        by_league: dict[str, list[_OddsTarget]] = {}
        for t in targets:
            by_league.setdefault(t.league, []).append(t)

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            ua  = random.choice(USER_AGENTS)
            ctx = await browser.new_context(user_agent=ua)
            page = await ctx.new_page()
            log.info("Establishing Flashscore session (UA: %s…)", ua[:45])
            await page.goto(
                FLASHSCORE_BASE_URL + "/basketball/",
                wait_until="domcontentloaded",
                timeout=20_000,
            )
            await asyncio.sleep(3)

            grand: dict[str, int] = {"ok": 0, "nohref": 0, "noodds": 0, "err": 0}
            t_total = time.monotonic()

            for league, league_targets in by_league.items():
                if league not in LEAGUE_PATHS:
                    log.warning(
                        "[%s] No Flashscore path mapping — skipping %d matches",
                        league, len(league_targets),
                    )
                    grand["nohref"] += len(league_targets)
                    continue

                log.info("━" * 60)
                log.info("[%s]  Building href index…", league)
                try:
                    href_index = await build_href_index(page, LEAGUE_PATHS[league])
                except Exception as exc:
                    log.error("[%s] Failed to build href index: %s", league, exc)
                    grand["err"] += len(league_targets)
                    continue

                ok = nohref = noodds = err = 0
                t0 = time.monotonic()

                for i, target in enumerate(league_targets, 1):
                    pathname = href_index.get(target.flashscore_id)
                    if pathname is None:
                        log.debug("  [%s] href not in index", target.flashscore_id)
                        nohref += 1
                        continue

                    try:
                        odds = await scrape_match_odds(page, target.flashscore_id, pathname)
                        if odds is None:
                            noodds += 1
                        else:
                            await _save_odds(self._sf, target.match_id, odds)
                            log.debug(
                                "  [%s] %s vs %s → close=%.1f open=%.1f bm=%s",
                                target.flashscore_id,
                                target.home_team, target.away_team,
                                odds.total_close or 0.0,
                                odds.total_open  or 0.0,
                                odds.bookmaker,
                            )
                            ok += 1
                    except Exception as exc:
                        log.warning(
                            "  err [%s] %s vs %s → %s",
                            target.flashscore_id,
                            target.home_team, target.away_team,
                            str(exc)[:120],
                        )
                        err += 1

                    if i % 20 == 0 or i == len(league_targets):
                        log.info(
                            "[%s]  %d/%d | ok=%d nohref=%d noodds=%d err=%d | %.0fs",
                            league, i, len(league_targets),
                            ok, nohref, noodds, err,
                            time.monotonic() - t0,
                        )
                    await asyncio.sleep(settings.collector.delay_min)

                log.info(
                    "[%s]  DONE  ok=%d  nohref=%d  noodds=%d  err=%d",
                    league, ok, nohref, noodds, err,
                )
                for k, v in [("ok", ok), ("nohref", nohref), ("noodds", noodds), ("err", err)]:
                    grand[k] += v

            await browser.close()

        log.info("━" * 60)
        log.info(
            "ALL DONE  ok=%d  nohref=%d  noodds=%d  err=%d  |  %.0f min",
            grand["ok"], grand["nohref"], grand["noodds"], grand["err"],
            (time.monotonic() - t_total) / 60,
        )
        await dispose_engine()


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

def _parse_args() -> dict[str, Any]:
    args = sys.argv[1:]
    opts: dict[str, Any] = {"leagues": None, "limit": None, "dry_run": False}
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
    OddsEnricher(
        leagues=opts["leagues"],
        limit=opts["limit"],
        dry_run=opts["dry_run"],
    ).run_sync()
