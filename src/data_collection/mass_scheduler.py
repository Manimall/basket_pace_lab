"""
Mass data collector — multiple leagues × multiple seasons.

Collects per-quarter statistics from Sofascore for all configured
leagues and seasons. Uses a single Playwright browser for the entire
run (keeps session alive and avoids 403). Random per-event sleep and
User-Agent rotation reduce Cloudflare detection probability.

Run (collect everything):
    python -m src.data_collection.mass_scheduler

Run specific leagues only:
    python -m src.data_collection.mass_scheduler --leagues NBA EuroLeague

Run specific leagues + seasons:
    python -m src.data_collection.mass_scheduler --leagues NBA --seasons 2324 2425

Dry-run (print queue, no network):
    python -m src.data_collection.mass_scheduler --dry-run
"""
from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any

from playwright.async_api import Page, async_playwright
from sqlalchemy import text

from src.config import settings
from src.database import crud
from src.database.crud import QuarterStatRow
from src.database.engine import create_tables, dispose_engine, get_session_factory
from src.database.models import MatchStatus, SeasonType
from src.data_collection.parsers import (
    PERIOD_MAP,
    SCORE_KEY_MAP,
    parse_period,
    safe_int,
)

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("mass_scheduler")

# ── Sofascore request config ──────────────────────────────────────────────────

API_HEADERS = {"Accept": "application/json", "Referer": "https://www.sofascore.com/"}

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]

# Delay range between individual event requests (seconds)
DELAY_MIN = 2.0
DELAY_MAX = 5.0

# Longer pause after each full season (let Sofascore breathe)
SEASON_PAUSE = 10.0

# ── League / season catalog ───────────────────────────────────────────────────
# Each entry: tournament_id, human name, list of (season_id, season_label).
# Season labels use short codes: "2324", "2425", "2526" for filtering via CLI.
# Add future seasons here — the scheduler handles them automatically.

LEAGUE_CATALOG: dict[str, dict] = {
    "NBA": {
        "tournament_id": 132,
        "seasons": [
            {"id": 80229, "name": "NBA 25/26",  "code": "2526"},
            {"id": 65360, "name": "NBA 24/25",  "code": "2425"},
            {"id": 54105, "name": "NBA 23/24",  "code": "2324"},
        ],
    },
    "EuroLeague": {
        "tournament_id": 138,
        "seasons": [
            {"id": 78545, "name": "Euroleague 25/26", "code": "2526"},
            {"id": 63971, "name": "Euroleague 24/25", "code": "2425"},
            {"id": 53198, "name": "Euroleague 23/24", "code": "2324"},
        ],
    },
    "VTB": {
        "tournament_id": 1438,
        "seasons": [
            {"id": 80491, "name": "United League 25/26", "code": "2526"},
            {"id": 64482, "name": "United League 24/25", "code": "2425"},
            {"id": 53040, "name": "United League 23/24", "code": "2324"},
        ],
    },
    "ACB": {
        "tournament_id": 264,
        "seasons": [
            {"id": 80922, "name": "Liga ACB 25/26", "code": "2526"},
            {"id": 64689, "name": "Liga ACB 24/25", "code": "2425"},
            {"id": 53749, "name": "Liga ACB 23/24", "code": "2324"},
        ],
    },
    "LegaA": {
        "tournament_id": 262,
        "seasons": [
            {"id": 79529, "name": "Serie A 25/26", "code": "2526"},
            {"id": 64742, "name": "Serie A 24/25", "code": "2425"},
            {"id": 53536, "name": "Serie A 23/24", "code": "2324"},
        ],
    },
    "BBL": {
        "tournament_id": 227,
        "seasons": [
            {"id": 79994, "name": "BBL 25/26", "code": "2526"},
            {"id": 65031, "name": "BBL 24/25", "code": "2425"},
            {"id": 52951, "name": "BBL 23/24", "code": "2324"},
        ],
    },
    "LNB": {
        "tournament_id": 156,
        "seasons": [
            # LNB Élite (top French Pro A league)
            # Season IDs fetched from /seasons endpoint
            {"id": 79227, "name": "Pro A 25/26", "code": "2526"},
            {"id": 63977, "name": "Pro A 24/25", "code": "2425"},
            {"id": 53369, "name": "Pro A 23/24", "code": "2324"},
        ],
    },
    "NBL": {
        "tournament_id": 1524,
        "seasons": [
            {"id": 77205, "name": "NBL 25/26", "code": "2526"},
            {"id": 61848, "name": "NBL 24/25", "code": "2425"},
            {"id": 52012, "name": "NBL 23/24", "code": "2324"},
        ],
    },
    "CBA": {
        "tournament_id": 1566,
        "seasons": [
            {"id": 85375, "name": "CBA 25/26",       "code": "2526"},
            {"id": 67166, "name": "CBA 24/25",       "code": "2425"},
            {"id": 55486, "name": "CBA 23/24",       "code": "2324"},
        ],
    },
    "ChinaNBL": {
        "tournament_id": 27568,
        "seasons": [
            # China second division — season IDs TBD (verify before running)
            # {"id": ???, "name": "NBL 25/26", "code": "2526"},
        ],
    },
    "PBA_Phil": {
        "tournament_id": 1956,
        "seasons": [
            {"id": 84100, "name": "PBA Philippine Cup 25/26", "code": "2526"},
            {"id": 74130, "name": "PBA Philippine Cup 2025",  "code": "2425"},
            {"id": 58687, "name": "PBA Philippine Cup 2024",  "code": "2324"},
        ],
    },
    "PBA_Comm": {
        "tournament_id": 1656,
        "seasons": [
            {"id": 90891, "name": "PBA Commissioner Cup 2026",  "code": "2526"},
            {"id": 69126, "name": "PBA Commissioner Cup 24/25", "code": "2425"},
            {"id": 56111, "name": "PBA Commissioner Cup 23/24", "code": "2324"},
        ],
    },
    "PBA_Gov": {
        "tournament_id": 1712,
        "seasons": [
            {"id": 65381, "name": "PBA Governors Cup 2024", "code": "2425"},
            {"id": 48362, "name": "PBA Governors Cup 22/23", "code": "2324"},
        ],
    },
    "LNBP": {
        "tournament_id": 1472,
        "seasons": [
            {"id": 75884, "name": "LNBP 2025", "code": "2526"},
            {"id": 61417, "name": "LNBP 2024", "code": "2425"},
            {"id": 52174, "name": "LNBP 2023", "code": "2324"},
        ],
    },
    "BLeague": {
        "tournament_id": 1502,
        "seasons": [
            {"id": 77915, "name": "B1 League 25/26", "code": "2526"},
            {"id": 64082, "name": "B1 League 24/25", "code": "2425"},
            {"id": 54374, "name": "B1 League 23/24", "code": "2324"},
        ],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────


async def api_get(page: Page, path: str) -> dict[str, Any] | None:
    url = f"https://www.sofascore.com/api/v1{path}"
    try:
        resp = await page.request.get(url, headers=API_HEADERS, timeout=15_000)
        if not resp.ok:
            return None
        return await resp.json()
    except Exception:
        return None


async def fetch_event_ids(
    page: Page, tournament_id: int, season_id: int, max_pages: int = 50
) -> list[int]:
    ids: list[int] = []
    for p in range(max_pages):
        data = await api_get(
            page,
            f"/unique-tournament/{tournament_id}/season/{season_id}/events/last/{p}",
        )
        if not data:
            break
        events = data.get("events", [])
        finished = [e["id"] for e in events if e.get("status", {}).get("type") == "finished"]
        ids.extend(finished)
        if not data.get("hasNextPage", True) or not events:
            break
    return ids


async def load_existing_external_ids(session_factory: Any) -> set[str]:
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
    session_factory: Any,
    existing_ids: set[str],
) -> str:
    """Fetch, parse, and persist one event. Returns 'ok'|'skip'|'exists'|'error'."""
    ext_id = str(event_id)

    # Fast-path: already in DB with stats
    if ext_id in existing_ids:
        return "exists"

    event = await api_get(page, f"/event/{event_id}")
    if not event:
        return "skip"

    ev = event.get("event", event)
    if not ev or not ev.get("homeTeam"):
        return "skip"

    home_raw        = ev["homeTeam"]
    away_raw        = ev["awayTeam"]
    home_score_raw  = ev.get("homeScore", {})
    away_score_raw  = ev.get("awayScore", {})

    def _sum_q(score: dict) -> int | None:
        vals  = [score.get(f"period{i}") for i in range(1, 5)]
        total = sum(int(v) for v in vals if v is not None)
        return total if total > 0 else None

    went_ot    = bool(home_score_raw.get("overtime") or away_score_raw.get("overtime"))
    status_map = {
        "finished":   MatchStatus.FINISHED,
        "inprogress": MatchStatus.LIVE,
        "notstarted": MatchStatus.SCHEDULED,
    }
    status      = status_map.get(ev.get("status", {}).get("type", ""), MatchStatus.FINISHED)
    ts          = ev.get("startTimestamp")
    scheduled_at = (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        if ts else datetime.now(tz=timezone.utc)
    )
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
            )

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

            if not rows:
                return "skip"

            await crud.save_quarter_stats(db, match.id, rows)
            existing_ids.add(ext_id)  # update in-memory cache

    return "ok"


# ── Scheduler ─────────────────────────────────────────────────────────────────


class MassScheduler:
    def __init__(
        self,
        session_factory: Any = None,
        max_pages: int = 50,
        league_filter: list[str] | None = None,
        season_filter: list[str] | None = None,
        dry_run: bool = False,
    ) -> None:
        self._sf           = session_factory or get_session_factory()
        self._max_pages    = max_pages
        self._league_filter = [l.upper() for l in league_filter] if league_filter else None
        self._season_filter = season_filter  # e.g. ["2324", "2425"]
        self._dry_run      = dry_run

    def _build_queue(self) -> list[tuple[str, int, int, str]]:
        """Returns list of (league_key, tournament_id, season_id, season_name)."""
        queue = []
        for league_key, cfg in LEAGUE_CATALOG.items():
            if self._league_filter and league_key.upper() not in self._league_filter:
                continue
            for season in cfg["seasons"]:
                if not season.get("id"):
                    continue  # placeholder without ID yet
                if self._season_filter and season.get("code") not in self._season_filter:
                    continue
                queue.append((
                    league_key,
                    cfg["tournament_id"],
                    season["id"],
                    season["name"],
                ))
        return queue

    async def run(self) -> None:
        queue = self._build_queue()

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
            ua      = random.choice(USER_AGENTS)
            ctx     = await browser.new_context(user_agent=ua)
            page    = await ctx.new_page()

            log.info("Establishing Sofascore session (UA: %s…)", ua[:40])
            await page.goto(
                "https://www.sofascore.com/basketball",
                wait_until="domcontentloaded",
                timeout=20_000,
            )
            await asyncio.sleep(3)

            grand_ok = grand_skip = grand_exists = grand_err = 0
            t_total  = time.monotonic()

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
                    # Rotate UA periodically (every ~100 events)
                    if i % 100 == 0:
                        new_ua = random.choice(USER_AGENTS)
                        await ctx.close()
                        ctx  = await browser.new_context(user_agent=new_ua)
                        page = await ctx.new_page()
                        await page.goto(
                            "https://www.sofascore.com/basketball",
                            wait_until="domcontentloaded",
                            timeout=20_000,
                        )
                        await asyncio.sleep(2)
                        log.info("%s  [UA rotated → %s…]", tag, new_ua[:40])

                    try:
                        result = await process_event(
                            page, event_id, tid, league_key, self._sf, existing_ids
                        )
                    except Exception as exc:
                        err += 1
                        log.debug("%s  event %d error: %s", tag, event_id, str(exc)[:120])
                        result = "error"

                    if result == "ok":
                        ok += 1
                    elif result == "exists":
                        exists += 1
                    elif result == "error":
                        pass  # already counted
                    else:
                        skip += 1

                    # Progress log every 20 events
                    if i % 20 == 0 or i == len(event_ids):
                        elapsed = time.monotonic() - t0
                        rps     = i / elapsed
                        eta     = (len(event_ids) - i) / rps if rps > 0 else 0
                        log.info(
                            "%s  %d/%d | ok=%d skip=%d exists=%d err=%d | "
                            "%.1f r/s | ETA %.0fs",
                            tag, i, len(event_ids), ok, skip, exists, err, rps, eta,
                        )

                    # Random delay — only if we actually hit the network
                    if result not in ("exists",):
                        await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

                grand_ok     += ok
                grand_skip   += skip
                grand_exists += exists
                grand_err    += err

                log.info(
                    "%s  DONE  ok=%d  skip=%d  exists=%d  err=%d",
                    tag, ok, skip, exists, err,
                )
                await asyncio.sleep(SEASON_PAUSE)

            await browser.close()

        elapsed_total = time.monotonic() - t_total
        log.info("━" * 60)
        log.info(
            "ALL DONE  ok=%d  skip=%d  exists=%d  err=%d  |  %.0f min",
            grand_ok, grand_skip, grand_exists, grand_err, elapsed_total / 60,
        )

        async with self._sf() as db:
            t = (await db.execute(text("SELECT COUNT(*) FROM teams"))).scalar()
            m = (await db.execute(text("SELECT COUNT(*) FROM matches"))).scalar()
            q = (await db.execute(text("SELECT COUNT(*) FROM quarter_stats"))).scalar()
        log.info("DB totals → teams: %d  matches: %d  quarter_stats: %d", t, m, q)
        await dispose_engine()


# ── CLI entry point ───────────────────────────────────────────────────────────


def _parse_args() -> dict:
    args    = sys.argv[1:]
    opts: dict = {"leagues": None, "seasons": None, "dry_run": False}
    i = 0
    while i < len(args):
        if args[i] == "--leagues" and i + 1 < len(args):
            leagues = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                leagues.append(args[i])
                i += 1
            opts["leagues"] = leagues
        elif args[i] == "--seasons" and i + 1 < len(args):
            seasons = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                seasons.append(args[i])
                i += 1
            opts["seasons"] = seasons
        elif args[i] == "--dry-run":
            opts["dry_run"] = True
            i += 1
        else:
            i += 1
    return opts


if __name__ == "__main__":
    opts = _parse_args()
    scheduler = MassScheduler(
        league_filter=opts["leagues"],
        season_filter=opts["seasons"],
        dry_run=opts["dry_run"],
    )
    asyncio.run(scheduler.run())
