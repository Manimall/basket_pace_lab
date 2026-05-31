"""
Historical data collector.

Использует Playwright page.request для обхода 403 на листинговых и
индивидуальных endpoints Sofascore. Один браузер на весь прогон.

Run:
    python collect_history.py                        # NBA 25/26, 5 страниц (~150 матчей)
    python collect_history.py --pages 20             # NBA 25/26, ~600 матчей
    python collect_history.py --season 65360         # NBA 24/25
    python collect_history.py --tournament 132 --season 54105   # NBA 23/24
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

from playwright.async_api import Page, async_playwright

from src.config import settings
from src.data_collection.parsers import SCORE_KEY_MAP, parse_period, safe_int
from src.database import crud
from src.database.engine import create_tables, dispose_engine, get_session_factory
from src.database.models import MatchStatus, SeasonType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("collect_history")

DEFAULT_TOURNAMENT_ID = 132
DEFAULT_SEASON_ID = 80229  # NBA 25/26
DEFAULT_PAGES = 5
DELAY_BETWEEN_EVENTS = 0.5
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
API_HEADERS = {"Accept": "application/json", "Referer": "https://www.sofascore.com/"}


def _parse_args() -> tuple[int, int, int]:
    args = sys.argv[1:]
    tournament, season, pages = DEFAULT_TOURNAMENT_ID, DEFAULT_SEASON_ID, DEFAULT_PAGES
    i = 0
    while i < len(args):
        if args[i] == "--tournament" and i + 1 < len(args):
            tournament = int(args[i + 1])
            i += 2
        elif args[i] == "--season" and i + 1 < len(args):
            season = int(args[i + 1])
            i += 2
        elif args[i] == "--pages" and i + 1 < len(args):
            pages = int(args[i + 1])
            i += 2
        else:
            i += 1
    return tournament, season, pages


async def api_get(page: Page, path: str) -> dict[str, Any] | None:
    url = f"https://www.sofascore.com/api/v1{path}"
    resp = await page.request.get(url, headers=API_HEADERS)
    if not resp.ok:
        return None
    return await resp.json()


async def fetch_event_ids(page: Page, tournament_id: int, season_id: int, max_pages: int) -> list[int]:
    ids: list[int] = []
    for p in range(max_pages):
        data = await api_get(page, f"/unique-tournament/{tournament_id}/season/{season_id}/events/last/{p}")
        if not data:
            log.warning("  page %d: failed, stopping", p)
            break
        events = data.get("events", [])
        finished = [e["id"] for e in events if e.get("status", {}).get("type") == "finished"]
        ids.extend(finished)
        log.info("  page %d: %d events, %d finished | total %d", p, len(events), len(finished), len(ids))
        if not data.get("hasNextPage", True) or not events:
            break
    return ids


def _safe_int(v: Any) -> int | None:
    return safe_int(v)


async def process_event(page: Page, event_id: int, session_factory: Any) -> str:
    """Returns: 'ok' | 'skip' | 'error'"""
    event = await api_get(page, f"/event/{event_id}")
    if not event:
        return "skip"

    ev = event.get("event", event)
    if not ev or not ev.get("homeTeam"):
        return "skip"

    # Команды и матч
    home_raw, away_raw = ev["homeTeam"], ev["awayTeam"]
    home_score_raw = ev.get("homeScore", {})
    away_score_raw = ev.get("awayScore", {})

    def _sum_q(score: dict) -> int | None:
        vals = [score.get(f"period{i}") for i in range(1, 5)]
        total = sum(int(v) for v in vals if v is not None)
        return total if total > 0 else None

    went_ot = bool(home_score_raw.get("overtime") or away_score_raw.get("overtime"))
    status_map = {"finished": MatchStatus.FINISHED, "inprogress": MatchStatus.LIVE,
                  "notstarted": MatchStatus.SCHEDULED}
    status = status_map.get(ev.get("status", {}).get("type", ""), MatchStatus.FINISHED)
    ts = ev.get("startTimestamp")
    scheduled_at = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(tz=timezone.utc)

    async with session_factory() as db:
        async with db.begin():
            home_team = await crud.get_or_create_team(
                db, str(home_raw["id"]), home_raw.get("name", "?"),
                home_raw.get("shortName", home_raw.get("name", "?"))[:8],
            )
            away_team = await crud.get_or_create_team(
                db, str(away_raw["id"]), away_raw.get("name", "?"),
                away_raw.get("shortName", away_raw.get("name", "?"))[:8],
            )
            match = await crud.upsert_match(
                db, str(event_id), home_team.id, away_team.id, scheduled_at,
                ev.get("season", {}).get("name", "unknown"),
                status=status, season_type=SeasonType.REGULAR,
                home_score_final=_safe_int(home_score_raw.get("current")),
                away_score_final=_safe_int(away_score_raw.get("current")),
                home_score_regulation=_sum_q(home_score_raw),
                away_score_regulation=_sum_q(away_score_raw),
                went_to_overtime=went_ot,
            )

            # Статистика по периодам
            stats = await api_get(page, f"/event/{event_id}/statistics")
            if not stats:
                return "skip"

            rows = []
            for period_data in stats.get("statistics", []):
                label = period_data.get("period", "").upper()
                if label == "ALL":
                    continue
                score_key = SCORE_KEY_MAP.get(label)
                h_score = _safe_int(home_score_raw.get(score_key)) if score_key else None
                a_score = _safe_int(away_score_raw.get(score_key)) if score_key else None
                row = parse_period(label, period_data, h_score, a_score)
                if row:
                    rows.append(row)

            if not rows:
                return "skip"

            await crud.save_quarter_stats(db, match.id, rows)

    return "ok"


async def main() -> None:
    tournament_id, season_id, max_pages = _parse_args()
    log.info("DB: %s:%s/%s | Tournament: %s  Season: %s  Pages: %s",
             settings.db.host, settings.db.port, settings.db.name,
             tournament_id, season_id, max_pages)

    await create_tables()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=USER_AGENT)
        page = await ctx.new_page()

        log.info("Establishing Sofascore session…")
        await page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=20000)
        await asyncio.sleep(3)

        log.info("Fetching event list…")
        event_ids = await fetch_event_ids(page, tournament_id, season_id, max_pages)
        log.info("Events to process: %d", len(event_ids))

        if not event_ids:
            log.error("No events found.")
            await browser.close()
            return

        session_factory = get_session_factory()
        ok = skipped = errors = 0
        t0 = time.monotonic()

        for i, event_id in enumerate(event_ids, 1):
            try:
                result = await process_event(page, event_id, session_factory)
                if result == "ok":
                    ok += 1
                else:
                    skipped += 1
            except Exception as exc:
                errors += 1
                log.debug("error %s: %s", event_id, str(exc)[:80])

            if i % 10 == 0 or i == len(event_ids):
                elapsed = time.monotonic() - t0
                rps = i / elapsed
                eta = (len(event_ids) - i) / rps if rps > 0 else 0
                log.info("Progress %d/%d | ok=%d skip=%d err=%d | %.1f r/s | ETA %.0fs",
                         i, len(event_ids), ok, skipped, errors, rps, eta)

            await asyncio.sleep(DELAY_BETWEEN_EVENTS)

        await browser.close()

    log.info("=== Done: saved=%d  skipped=%d  errors=%d ===", ok, skipped, errors)

    from sqlalchemy import text
    async with get_session_factory()() as db:
        t = (await db.execute(text("SELECT COUNT(*) FROM teams"))).scalar()
        m = (await db.execute(text("SELECT COUNT(*) FROM matches"))).scalar()
        q = (await db.execute(text("SELECT COUNT(*) FROM quarter_stats"))).scalar()
    log.info("DB → teams: %d  matches: %d  quarter_stats: %d", t, m, q)

    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
