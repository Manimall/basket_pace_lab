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

from playwright.async_api import async_playwright, Page

from src.config import settings
from src.database import crud
from src.database.crud import QuarterStatRow
from src.database.engine import create_tables, dispose_engine, get_session_factory
from src.database.models import MatchStatus, PeriodType, SeasonType
from src.data_collection.sofascore_client import _calc_possessions, _calc_pace

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

# NBA uses "1Q".."4Q"; EuroLeague uses "1ST".."4TH"
PERIOD_MAP: dict[str, tuple[int, PeriodType]] = {
    "1Q": (1, PeriodType.QUARTER),
    "2Q": (2, PeriodType.QUARTER),
    "3Q": (3, PeriodType.QUARTER),
    "4Q": (4, PeriodType.QUARTER),
    "1ST": (1, PeriodType.QUARTER),
    "2ND": (2, PeriodType.QUARTER),
    "3RD": (3, PeriodType.QUARTER),
    "4TH": (4, PeriodType.QUARTER),
    "OT":  (1, PeriodType.OVERTIME),
    "1OT": (1, PeriodType.OVERTIME),
    "2OT": (2, PeriodType.OVERTIME),
    "3OT": (3, PeriodType.OVERTIME),
}

# Maps period label → key in homeScore/awayScore event object
SCORE_KEY_MAP: dict[str, str] = {
    "1Q": "period1", "1ST": "period1",
    "2Q": "period2", "2ND": "period2",
    "3Q": "period3", "3RD": "period3",
    "4Q": "period4", "4TH": "period4",
    "OT": "overtime", "1OT": "overtime",
    "2OT": "overtime2", "3OT": "overtime3",
}


def _parse_args() -> tuple[int, int, int]:
    args = sys.argv[1:]
    tournament, season, pages = DEFAULT_TOURNAMENT_ID, DEFAULT_SEASON_ID, DEFAULT_PAGES
    i = 0
    while i < len(args):
        if args[i] == "--tournament" and i + 1 < len(args):
            tournament = int(args[i + 1]); i += 2
        elif args[i] == "--season" and i + 1 < len(args):
            season = int(args[i + 1]); i += 2
        elif args[i] == "--pages" and i + 1 < len(args):
            pages = int(args[i + 1]); i += 2
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
    try: return int(v)
    except: return None

def _parse_shot(s: str) -> tuple[int, int]:
    # Handles "27/78 (34%)" or plain "27/78"
    try:
        parts = str(s).split("/")
        made = int(parts[0].strip())
        att = int(parts[1].strip().split()[0].rstrip("(").strip())
        return made, att
    except:
        return 0, 0

def _build_metrics(period_data: dict) -> dict[str, dict]:
    m: dict[str, dict] = {}
    for g in period_data.get("groups", []):
        for item in g.get("statisticsItems", []):
            name = item.get("name", "").lower().strip()
            if name:
                m[name] = {"home": item.get("home"), "away": item.get("away")}
    return m

def _parse_period(
    label: str,
    period_data: dict,
    home_score: int | None,
    away_score: int | None,
) -> QuarterStatRow | None:
    mapping = PERIOD_MAP.get(label.upper())
    if not mapping:
        return None
    period_number, period_type = mapping
    m = _build_metrics(period_data)

    _, home_fga = _parse_shot(m.get("field goals", {}).get("home", ""))
    _, away_fga = _parse_shot(m.get("field goals", {}).get("away", ""))
    _, home_fta = _parse_shot(m.get("free throws", {}).get("home", ""))
    _, away_fta = _parse_shot(m.get("free throws", {}).get("away", ""))
    home_off = _safe_int(m.get("offensive rebounds", {}).get("home"))
    away_off = _safe_int(m.get("offensive rebounds", {}).get("away"))
    home_to = _safe_int(m.get("turnovers", {}).get("home"))
    away_to = _safe_int(m.get("turnovers", {}).get("away"))

    home_poss = _calc_possessions(home_fga or None, home_fta or None, home_off, home_to)
    away_poss = _calc_possessions(away_fga or None, away_fta or None, away_off, away_to)

    return QuarterStatRow(
        period_number=period_number, period_type=period_type,
        home_score=home_score, away_score=away_score,
        home_fga=home_fga or None, away_fga=away_fga or None,
        home_fta=home_fta or None, away_fta=away_fta or None,
        home_off_reb=home_off, away_off_reb=away_off,
        home_turnovers=home_to, away_turnovers=away_to,
        home_possessions=home_poss, away_possessions=away_poss,
        home_pace=_calc_pace(home_poss, period_type),
        away_pace=_calc_pace(away_poss, period_type),
    )


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
                row = _parse_period(label, period_data, h_score, a_score)
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
