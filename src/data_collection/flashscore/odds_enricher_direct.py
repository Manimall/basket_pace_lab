"""Direct-URL odds enricher — bypasses results-page click navigation.

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

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

from playwright.async_api import async_playwright

from src.config import settings
from src.data_collection.base import BaseCollector
from src.data_collection.constants import FLASHSCORE_BASE_URL
from src.data_collection.flashscore.browser import make_flashscore_session
from src.data_collection.flashscore.odds import scrape_match_odds
from src.data_collection.flashscore.odds_db import load_odds_targets, save_odds
from src.database.engine import dispose_engine, get_session_factory

log = logging.getLogger(__name__)

# Progress is logged every N processed matches.
_PROGRESS_EVERY: int = 25


@dataclass
class _RunStats:
    """Mutable counters for one direct-enrichment run."""

    ok:     int = 0
    noodds: int = 0
    err:    int = 0


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
        """Scrape closing O/U lines via direct match URLs and persist them."""
        log.info("DB: %s:%s/%s", settings.db.host, settings.db.port, settings.db.name)

        targets = await load_odds_targets(
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

        stats = _RunStats()
        t0 = time.monotonic()

        async with async_playwright() as pw:
            browser, odds_page = await make_flashscore_session(pw)
            try:
                for i, target in enumerate(targets, 1):
                    match_url = f"{FLASHSCORE_BASE_URL}/match/{target.flashscore_id}/"
                    try:
                        row = await scrape_match_odds(
                            odds_page, target.flashscore_id, match_url,
                        )
                        if row is None:
                            stats.noodds += 1
                            log.debug(
                                "  noodds [%s] %s vs %s",
                                target.flashscore_id, target.home_team, target.away_team,
                            )
                        else:
                            await save_odds(self._sf, target.match_id, row)
                            log.debug(
                                "  ok [%s] %s vs %s → close=%.1f bm=%s",
                                target.flashscore_id, target.home_team, target.away_team,
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

                    if i % _PROGRESS_EVERY == 0 or i == len(targets):
                        elapsed = time.monotonic() - t0
                        rate    = i / elapsed if elapsed > 0 else 0
                        eta_min = (len(targets) - i) / rate / 60 if rate > 0 else 0
                        log.info(
                            "%d/%d | ok=%d noodds=%d err=%d | %.0fs | ETA ~%.0f min",
                            i, len(targets), stats.ok, stats.noodds, stats.err, elapsed, eta_min,
                        )
                    await asyncio.sleep(settings.collector.delay_min)
            finally:
                await browser.close()

        log.info("━" * 60)
        log.info(
            "DONE  ok=%d  noodds=%d  err=%d  |  %.1f min",
            stats.ok, stats.noodds, stats.err, (time.monotonic() - t0) / 60,
        )
        await dispose_engine()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enrich matches with O/U lines via direct match URLs.",
    )
    p.add_argument("--leagues", nargs="+", metavar="LEAGUE", help="Limit to these leagues.")
    p.add_argument("--limit", type=int, default=None, help="Max matches to process.")
    p.add_argument("--offset", type=int, default=None, help="Row offset for parallel chunks.")
    p.add_argument("--dry-run", action="store_true", help="Preview targets, no DB writes.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        stream=sys.stdout,
    )
    args = _parse_args()
    DirectOddsEnricher(
        leagues=args.leagues,
        limit=args.limit,
        offset=args.offset,
        dry_run=args.dry_run,
    ).run_sync()
