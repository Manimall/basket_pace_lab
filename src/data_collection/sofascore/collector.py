"""
MassScheduler — Sofascore multi-league × multi-season bulk collector.

Usage:
    python -m src.data_collection.sofascore.collector
    python -m src.data_collection.sofascore.collector --leagues NBA EuroLeague
    python -m src.data_collection.sofascore.collector --leagues NBA --seasons 2324 2425
    python -m src.data_collection.sofascore.collector --dry-run
"""
from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
from typing import Any

from playwright.async_api import async_playwright
from sqlalchemy import text

from src.config import settings
from src.data_collection.base import BaseCollector
from src.data_collection.constants import USER_AGENTS
from src.data_collection.sofascore.catalog import LEAGUE_CATALOG
from src.data_collection.sofascore.event import (
    fetch_event_ids,
    load_existing_external_ids,
    process_event,
)
from src.database.engine import create_tables, dispose_engine, get_session_factory

log = logging.getLogger("mass_scheduler")


class MassScheduler(BaseCollector):
    def __init__(
        self,
        session_factory: Any = None,
        max_pages: int = 50,
        league_filter: list[str] | None = None,
        season_filter: list[str] | None = None,
        dry_run: bool = False,
    ) -> None:
        self._sf            = session_factory or get_session_factory()
        self._max_pages     = max_pages
        self._league_filter = [l.upper() for l in league_filter] if league_filter else None
        self._season_filter = season_filter
        self._dry_run       = dry_run

    def _build_queue(self) -> list[tuple[str, int, int, str]]:
        queue = []
        for league_key, cfg in LEAGUE_CATALOG.items():
            if self._league_filter and league_key.upper() not in self._league_filter:
                continue
            for season in cfg["seasons"]:
                if not season.get("id"):
                    continue
                if self._season_filter and season.get("code") not in self._season_filter:
                    continue
                queue.append((league_key, cfg["tournament_id"], season["id"], season["name"]))
        return queue

    async def run(self) -> None:
        queue = self._build_queue()
        cfg   = settings.collector

        if self._dry_run:
            log.info("DRY-RUN — queue (%d tasks):", len(queue))
            for league, tid, sid, sname in queue:
                log.info("  %-12s  tournament=%-6d  season=%-6d  %s", league, tid, sid, sname)
            return

        if not queue:
            log.error("Queue is empty. Check --leagues / --seasons filters.")
            return

        log.info("DB: %s:%s/%s", settings.db.host, settings.db.port, settings.db.name)
        log.info("Queue: %d league-seasons to collect", len(queue))
        await create_tables()

        log.info("Loading existing event IDs from DB…")
        existing_ids = await load_existing_external_ids(self._sf)
        log.info("  %d events already in DB (will be skipped)", len(existing_ids))

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            ua  = random.choice(USER_AGENTS)
            ctx = await browser.new_context(user_agent=ua)
            page = await ctx.new_page()
            log.info("Establishing session (UA: %s…)", ua[:40])
            await page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=20_000)
            await asyncio.sleep(3)

            grand = {"ok": 0, "skip": 0, "exists": 0, "err": 0}
            t_total = time.monotonic()

            for q_idx, (league_key, tid, sid, sname) in enumerate(queue, 1):
                tag = f"[{league_key}][{sname}]"
                log.info("━" * 60)
                log.info("%s  Fetching event list…  (task %d/%d)", tag, q_idx, len(queue))

                event_ids = await fetch_event_ids(page, tid, sid, self._max_pages)
                log.info("%s  %d finished events found", tag, len(event_ids))
                if not event_ids:
                    continue

                ok = skip = exists = err = 0
                t0 = time.monotonic()

                for i, event_id in enumerate(event_ids, 1):
                    if i % 100 == 0:
                        new_ua = random.choice(USER_AGENTS)
                        await ctx.close()
                        ctx  = await browser.new_context(user_agent=new_ua)
                        page = await ctx.new_page()
                        await page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=20_000)
                        await asyncio.sleep(2)
                        log.info("%s  [UA rotated → %s…]", tag, new_ua[:40])

                    try:
                        result = await process_event(page, event_id, tid, league_key, self._sf, existing_ids)
                    except Exception as exc:
                        err += 1
                        log.debug("%s  event %d error: %s", tag, event_id, str(exc)[:120])
                        result = "error"

                    if result == "ok":       ok     += 1
                    elif result == "exists": exists += 1
                    elif result == "error":  pass
                    else:                    skip   += 1

                    if i % 20 == 0 or i == len(event_ids):
                        elapsed = time.monotonic() - t0
                        rps = i / elapsed
                        eta = (len(event_ids) - i) / rps if rps > 0 else 0
                        log.info("%s  %d/%d | ok=%d skip=%d exists=%d err=%d | %.1f r/s | ETA %.0fs",
                                 tag, i, len(event_ids), ok, skip, exists, err, rps, eta)

                    if result not in ("exists",):
                        await asyncio.sleep(random.uniform(cfg.delay_min, cfg.delay_max))

                for k, v in [("ok", ok), ("skip", skip), ("exists", exists), ("err", err)]:
                    grand[k] += v
                log.info("%s  DONE  ok=%d  skip=%d  exists=%d  err=%d", tag, ok, skip, exists, err)
                await asyncio.sleep(cfg.season_pause_sec)

            await browser.close()

        log.info("━" * 60)
        log.info("ALL DONE  ok=%d  skip=%d  exists=%d  err=%d  |  %.0f min",
                 grand["ok"], grand["skip"], grand["exists"], grand["err"],
                 (time.monotonic() - t_total) / 60)

        async with self._sf() as db:
            t = (await db.execute(text("SELECT COUNT(*) FROM teams"))).scalar()
            m = (await db.execute(text("SELECT COUNT(*) FROM matches"))).scalar()
            q = (await db.execute(text("SELECT COUNT(*) FROM quarter_stats"))).scalar()
        log.info("DB totals → teams: %d  matches: %d  quarter_stats: %d", t, m, q)
        await dispose_engine()


def _parse_args() -> dict:
    args = sys.argv[1:]
    opts: dict = {"leagues": None, "seasons": None, "dry_run": False}
    i = 0
    while i < len(args):
        if args[i] == "--leagues" and i + 1 < len(args):
            vals, i = [], i + 1
            while i < len(args) and not args[i].startswith("--"):
                vals.append(args[i]); i += 1
            opts["leagues"] = vals
        elif args[i] == "--seasons" and i + 1 < len(args):
            vals, i = [], i + 1
            while i < len(args) and not args[i].startswith("--"):
                vals.append(args[i]); i += 1
            opts["seasons"] = vals
        elif args[i] == "--dry-run":
            opts["dry_run"] = True; i += 1
        else:
            i += 1
    return opts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)
    opts = _parse_args()
    MassScheduler(league_filter=opts["leagues"], season_filter=opts["seasons"], dry_run=opts["dry_run"]).run_sync()
