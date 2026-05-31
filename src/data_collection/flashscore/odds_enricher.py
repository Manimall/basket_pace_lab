"""OddsEnricher — populates pre-match Over/Under lines for historical matches.

Two-page strategy:
  - nav_page:  stays on league results page; used only for clicking matches to
               resolve their full URLs (React SPA has no <a href> on match rows).
  - odds_page: dedicated page for scraping odds; never affects nav_page state.

After each click on nav_page, go_back() restores results page via bfcache.
If bfcache miss is detected (match count drops), results page is reloaded.

Usage:
    python -m src.data_collection.flashscore.odds_enricher
    python -m src.data_collection.flashscore.odds_enricher --leagues LegaA PBA_PhilCup
    python -m src.data_collection.flashscore.odds_enricher --limit 50 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import time
from dataclasses import dataclass

from playwright.async_api import async_playwright

from src.config import settings
from src.data_collection.base import BaseCollector
from src.data_collection.constants import FLASHSCORE_BASE_URL, USER_AGENTS
from src.data_collection.flashscore.odds import (
    dismiss_overlays,
    get_match_url_via_click,
    prepare_results_page,
    scrape_match_odds,
)
from src.data_collection.flashscore.odds_db import OddsTarget, load_odds_targets, save_odds
from src.data_collection.flashscore.page import LEAGUE_PATHS
from src.database.engine import SessionFactory, dispose_engine, get_session_factory

log = logging.getLogger(__name__)

# Minimum match elements visible after go_back(); below this → reload results page.
_BFCACHE_MIN_MATCHES: int = 5
_MATCH_SELECTOR: str = ".event__match"


@dataclass
class _RunStats:
    """Mutable counters for one enrichment run."""

    ok:     int = 0
    nohref: int = 0
    noodds: int = 0
    err:    int = 0

    def update(self, other: _RunStats) -> None:
        """Add another stats bundle's counters into this one in place."""
        self.ok     += other.ok
        self.nohref += other.nohref
        self.noodds += other.noodds
        self.err    += other.err


class OddsEnricher(BaseCollector):
    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        leagues: list[str] | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> None:
        self._sf      = session_factory or get_session_factory()
        self._leagues = leagues
        self._limit   = limit
        self._dry_run = dry_run

    async def run(self) -> None:
        """Scrape and persist closing O/U lines for matches needing them."""
        log.info("DB: %s:%s/%s", settings.db.host, settings.db.port, settings.db.name)

        targets = await load_odds_targets(self._sf, self._leagues, self._limit)
        log.info("Matches to enrich with odds: %d", len(targets))
        if not targets:
            log.info("Nothing to enrich.")
            return

        if self._dry_run:
            for t in targets[:20]:
                log.info(
                    "  [dry] %s | %s vs %s | %s | fs=%s",
                    t.league, t.home_team, t.away_team, t.match_date, t.flashscore_id,
                )
            log.info("  [dry] Total: %d", len(targets))
            return

        by_league: dict[str, list[OddsTarget]] = {}
        for t in targets:
            by_league.setdefault(t.league, []).append(t)

        grand = _RunStats()
        t_total = time.monotonic()

        async with async_playwright() as pw:
            browser  = await pw.chromium.launch(headless=True)
            ua       = random.choice(USER_AGENTS)
            ctx      = await browser.new_context(user_agent=ua)
            nav_page  = await ctx.new_page()
            odds_page = await ctx.new_page()

            log.info("Establishing Flashscore session (UA: %s…)", ua[:45])
            await nav_page.goto(
                FLASHSCORE_BASE_URL + "/basketball/",
                wait_until="domcontentloaded", timeout=20_000,
            )
            await asyncio.sleep(3)
            await dismiss_overlays(nav_page)

            for league, league_targets in by_league.items():
                if league not in LEAGUE_PATHS:
                    log.warning("[%s] No Flashscore path — skipping %d", league, len(league_targets))
                    grand.nohref += len(league_targets)
                    continue

                log.info("━" * 60)
                log.info("[%s]  Loading results page…", league)
                league_path = LEAGUE_PATHS[league]
                try:
                    await prepare_results_page(nav_page, league_path)
                    await dismiss_overlays(nav_page)
                except Exception as exc:
                    log.error("[%s] Results page failed: %s", league, exc)
                    grand.err += len(league_targets)
                    continue

                stats = _RunStats()
                t0 = time.monotonic()

                for i, target in enumerate(league_targets, 1):
                    match_url = await get_match_url_via_click(nav_page, target.flashscore_id)
                    if match_url is None:
                        stats.nohref += 1
                    else:
                        visible: int = await nav_page.evaluate(
                            f"() => document.querySelectorAll('{_MATCH_SELECTOR}').length"
                        )
                        if visible < _BFCACHE_MIN_MATCHES:
                            log.debug("  bfcache miss — reloading results page")
                            await prepare_results_page(nav_page, league_path)

                        try:
                            row = await scrape_match_odds(odds_page, target.flashscore_id, match_url)
                            if row is None:
                                stats.noodds += 1
                            else:
                                await save_odds(self._sf, target.match_id, row)
                                log.debug(
                                    "  ok [%s] %s vs %s → close=%.1f bm=%s",
                                    target.flashscore_id,
                                    target.home_team, target.away_team,
                                    row.total_close or 0.0, row.bookmaker,
                                )
                                stats.ok += 1
                        except Exception as exc:
                            log.warning(
                                "  err [%s] %s vs %s → %s",
                                target.flashscore_id, target.home_team, target.away_team,
                                str(exc)[:120],
                            )
                            stats.err += 1

                    if i % 10 == 0 or i == len(league_targets):
                        log.info(
                            "[%s]  %d/%d | ok=%d nohref=%d noodds=%d err=%d | %.0fs",
                            league, i, len(league_targets),
                            stats.ok, stats.nohref, stats.noodds, stats.err,
                            time.monotonic() - t0,
                        )
                    await asyncio.sleep(settings.collector.delay_min)

                log.info(
                    "[%s]  DONE  ok=%d  nohref=%d  noodds=%d  err=%d",
                    league, stats.ok, stats.nohref, stats.noodds, stats.err,
                )
                grand.update(stats)

            await browser.close()

        log.info("━" * 60)
        log.info(
            "ALL DONE  ok=%d  nohref=%d  noodds=%d  err=%d  |  %.0f min",
            grand.ok, grand.nohref, grand.noodds, grand.err,
            (time.monotonic() - t_total) / 60,
        )
        await dispose_engine()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich historical matches with Flashscore O/U lines.")
    p.add_argument("--leagues", nargs="+", metavar="LEAGUE", help="Limit to these leagues.")
    p.add_argument("--limit", type=int, default=None, help="Max matches to process.")
    p.add_argument("--dry-run", action="store_true", help="Preview targets, no DB writes.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        stream=sys.stdout,
    )
    args = _parse_args()
    OddsEnricher(
        leagues=args.leagues,
        limit=args.limit,
        dry_run=args.dry_run,
    ).run_sync()
