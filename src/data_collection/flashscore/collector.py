"""
Flashscore collector — collects matches from scratch for leagues not on Sofascore.

For each configured league:
  1. Scrape results page (click 'show more' until all loaded)
  2. Extract: teams, date/time, final score, Q1-Q4 scores, flashscore_id
  3. Upsert teams → upsert match → insert 4 quarter_stats rows

Usage:
    python -m src.data_collection.flashscore.collector
    python -m src.data_collection.flashscore.collector --leagues LegaA PBA_PhilCup
    python -m src.data_collection.flashscore.collector --dry-run
"""
from __future__ import annotations

import asyncio
import logging
import random
import argparse
import sys
import time

from playwright.async_api import async_playwright

from src.config import settings
from src.data_collection.base import BaseCollector
from src.data_collection.constants import FLASHSCORE_BASE_URL, USER_AGENTS
from src.data_collection.flashscore.collector_page import (
    COLLECTOR_LEAGUES, FsCollectedMatch, save_match, scrape_results_page,
)
from src.database.engine import SessionFactory, dispose_engine, get_session_factory

log = logging.getLogger("flashscore_collector")


class FlashscoreCollector(BaseCollector):
    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        leagues: list[str] | None = None,
        dry_run: bool = False,
    ) -> None:
        self._sf      = session_factory or get_session_factory()
        self._leagues = leagues
        self._dry_run = dry_run

    async def run(self) -> None:
        """Scrape configured leagues' results pages and upsert finished matches."""
        log.info("DB: %s:%s/%s", settings.db.host, settings.db.port, settings.db.name)

        targets = {
            k: v for k, v in COLLECTOR_LEAGUES.items()
            if self._leagues is None or k in self._leagues
        }
        log.info("Leagues to collect: %s", list(targets.keys()))

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            ua  = random.choice(USER_AGENTS)
            ctx = await browser.new_context(user_agent=ua)
            page = await ctx.new_page()
            log.info("Establishing Flashscore session (UA: %s…)", ua[:45])
            await page.goto(FLASHSCORE_BASE_URL + "/basketball/", wait_until="domcontentloaded", timeout=20_000)
            await asyncio.sleep(3)

            grand_ok = grand_skip = grand_err = 0
            t_total  = time.monotonic()

            for league_key, config in targets.items():
                log.info("━" * 60)
                log.info("[%s]  Collecting from Flashscore…", league_key)

                try:
                    matches = await scrape_results_page(page, config["path"])
                except Exception as exc:
                    log.error("[%s] Failed to scrape: %s", league_key, exc)
                    continue

                if self._dry_run:
                    for m in matches[:5]:
                        log.info(
                            "  [dry] %s | %s vs %s | final %s-%s | Q1-4: %s",
                            m.time_str, m.home_raw, m.away_raw,
                            m.final_home, m.final_away,
                            [(qs[0], qs[1]) for qs in m.q_scores],
                        )
                    log.info("  [dry] Total: %d matches found for %s", len(matches), league_key)
                    continue

                ok = skip = err = 0
                t0 = time.monotonic()

                for i, m in enumerate(matches, 1):
                    try:
                        success = await save_match(self._sf, league_key, config, m)
                        if success:
                            ok += 1
                        else:
                            skip += 1
                    except Exception as exc:
                        log.warning("  err [%d]: %s vs %s → %s", i, m.home_raw, m.away_raw, str(exc)[:100])
                        err += 1

                    if i % 50 == 0 or i == len(matches):
                        log.info("[%s]  %d/%d | ok=%d skip=%d err=%d | %.0fs",
                                 league_key, i, len(matches), ok, skip, err,
                                 time.monotonic() - t0)

                log.info("[%s]  DONE  ok=%d  skip=%d  err=%d", league_key, ok, skip, err)
                grand_ok += ok; grand_skip += skip; grand_err += err

            await browser.close()

        log.info("━" * 60)
        log.info("ALL DONE  ok=%d  skip=%d  err=%d  |  %.0f min",
                 grand_ok, grand_skip, grand_err, (time.monotonic() - t_total) / 60)
        await dispose_engine()


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for the Flashscore collector.

    Returns:
        Namespace with ``leagues`` (list[str] | None) and ``dry_run`` (bool).
    """
    p = argparse.ArgumentParser(description="Flashscore multi-league results collector.")
    p.add_argument("--leagues", nargs="+", default=None, help="League keys to collect.")
    p.add_argument("--dry-run", action="store_true", help="List targets, do not write.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)
    args = _parse_args()
    FlashscoreCollector(leagues=args.leagues, dry_run=args.dry_run).run_sync()
