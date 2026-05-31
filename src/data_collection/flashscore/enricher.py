"""
Flashscore enricher — replaces game-fallback quarter_stats with real per-quarter data.

Strategy:
  - Build an in-memory index from the Flashscore results page for each league.
  - Fuzzy-match DB matches to Flashscore entries by date (±1 day) + team names.
  - Replace the single GAME row with 4 QUARTER rows; set has_quarter_breakdown=True.

Usage:
    python -m src.data_collection.flashscore.enricher
    python -m src.data_collection.flashscore.enricher --leagues EuroLeague VTB BBL
    python -m src.data_collection.flashscore.enricher --leagues LegaA ACB --seasons 2526 2425
    python -m src.data_collection.flashscore.enricher --leagues EuroLeague --limit 20
    python -m src.data_collection.flashscore.enricher --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import time
from typing import Any

from playwright.async_api import async_playwright

from src.config import settings
from src.data_collection.base import BaseCollector
from src.data_collection.constants import FLASHSCORE_BASE_URL, USER_AGENTS
from src.data_collection.flashscore.db import ensure_flashscore_id_column, load_db_matches, save_enriched
from src.data_collection.flashscore.norm import (  # noqa: F401 — re-exported for test backward compat
    DbMatch, FsMatch, QScore, _NAME_ALIASES, _norm, _sim,
)
from src.data_collection.flashscore.page import (
    LEAGUE_PATHS, build_league_index, build_quarter_rows, find_match,
)
from src.database.engine import SessionFactory, dispose_engine, get_session_factory

log = logging.getLogger("flashscore_enricher")


class FlashscoreEnricher(BaseCollector):
    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        leagues: list[str] | None = None,
        season_codes: list[str] | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> None:
        self._sf           = session_factory or get_session_factory()
        self._leagues      = leagues
        self._season_codes = season_codes
        self._limit        = limit
        self._dry_run      = dry_run

    async def run(self) -> None:
        """Enrich DB matches with Flashscore quarter scores for the targets."""
        await ensure_flashscore_id_column(self._sf)
        log.info("DB: %s:%s/%s", settings.db.host, settings.db.port, settings.db.name)

        db_matches = await load_db_matches(self._sf, self._leagues, self._season_codes, self._limit)
        log.info("Matches to enrich: %d", len(db_matches))

        if self._dry_run:
            for m in db_matches[:20]:
                log.info("  [dry] %s | %s vs %s | %s", m.league, m.home_team, m.away_team, m.match_date)
            return

        if not db_matches:
            log.info("Nothing to enrich.")
            return

        by_league: dict[str, list[DbMatch]] = {}
        for m in db_matches:
            by_league.setdefault(m.league, []).append(m)

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            ua  = random.choice(USER_AGENTS)
            ctx = await browser.new_context(user_agent=ua)
            page = await ctx.new_page()
            log.info("Establishing Flashscore session (UA: %s…)", ua[:45])
            await page.goto(FLASHSCORE_BASE_URL + "/basketball/", wait_until="domcontentloaded", timeout=20_000)
            await asyncio.sleep(3)

            grand = {"ok": 0, "nomatch": 0, "noscores": 0, "err": 0}
            t_total = time.monotonic()

            for league_key, matches in by_league.items():
                if league_key not in LEAGUE_PATHS:
                    log.warning("[%s] No Flashscore URL mapping — skipping", league_key)
                    grand["nomatch"] += len(matches)
                    continue
                log.info("━" * 60)
                log.info("[%s]  Building results index…", league_key)
                try:
                    index = await build_league_index(page, league_key)
                except Exception as exc:
                    log.error("[%s] Failed to build index: %s", league_key, exc)
                    grand["err"] += len(matches)
                    continue

                ok = nomatch = noscores = err = 0
                t0 = time.monotonic()
                for i, m in enumerate(matches, 1):
                    try:
                        fs_entry = find_match(index, m.match_date, m.home_team, m.away_team)
                        if fs_entry is None:
                            nomatch += 1
                        else:
                            quarter_rows = build_quarter_rows(fs_entry)
                            if quarter_rows is None:
                                noscores += 1
                            else:
                                await save_enriched(self._sf, m.match_id, fs_entry.fs_id, quarter_rows)
                                ok += 1
                    except Exception as exc:
                        log.warning("  err: %s vs %s → %s", m.home_team, m.away_team, str(exc)[:100])
                        err += 1

                    if i % 50 == 0 or i == len(matches):
                        log.info("[%s]  %d/%d | ok=%d nomatch=%d noscores=%d err=%d | %.0fs",
                                 league_key, i, len(matches), ok, nomatch, noscores, err,
                                 time.monotonic() - t0)

                for k, v in [("ok", ok), ("nomatch", nomatch), ("noscores", noscores), ("err", err)]:
                    grand[k] += v
                log.info("[%s]  DONE  ok=%d  nomatch=%d  noscores=%d  err=%d",
                         league_key, ok, nomatch, noscores, err)

            await browser.close()

        log.info("━" * 60)
        log.info("ALL DONE  ok=%d  nomatch=%d  noscores=%d  err=%d  |  %.0f min",
                 grand["ok"], grand["nomatch"], grand["noscores"], grand["err"],
                 (time.monotonic() - t_total) / 60)
        await dispose_engine()


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for the Flashscore quarter-score enricher.

    Returns:
        Namespace with ``leagues`` / ``seasons`` (list[str] | None),
        ``limit`` (int | None), and ``dry_run`` (bool).
    """
    p = argparse.ArgumentParser(description="Enrich matches with Flashscore quarter scores.")
    p.add_argument("--leagues", nargs="+", default=None, help="League keys to enrich.")
    p.add_argument("--seasons", nargs="+", default=None, help="Season codes to restrict to.")
    p.add_argument("--limit", type=int, default=None, help="Max matches to process.")
    p.add_argument("--dry-run", action="store_true", help="List targets, do not write.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)
    args = _parse_args()
    FlashscoreEnricher(
        leagues=args.leagues, season_codes=args.seasons,
        limit=args.limit, dry_run=args.dry_run,
    ).run_sync()
