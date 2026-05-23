"""
ChinaNBL scraper via Sofascore mobile API (api.sofascore.com).

Access method:
  Playwright browser navigates directly to api.sofascore.com URLs.
  Cloudflare challenge is bypassed by Chromium's real TLS fingerprint.
  No cookies.json required.

Data per match:
  - Q1-Q4 scores from events-list response (homeScore.period1-4)
  - OT period scores if present (period5, period6, …)
  - Full-game FGA/FTA from /event/{id}/statistics (stored as GAME row)

Seasons collected:
  87684 → NBL 25/26  (~202 matches)
  77353 → NBL 2025   (~90 matches)

Usage:
    python -m src.data_collection.chinanbl_scraper
    python -m src.data_collection.chinanbl_scraper --no-stats
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from sqlalchemy.ext.asyncio import AsyncSession as DbSession

from src.database import crud
from src.database.crud import QuarterStatRow
from src.database.engine import dispose_engine, get_session_factory
from src.database.models import MatchStatus, PeriodType, SeasonType

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL        = "https://api.sofascore.com/api/v1"
TOURNAMENT_ID   = 27568
TOURNAMENT_NAME = "ChinaNBL"

SEASONS: dict[int, tuple[str, SeasonType]] = {
    87684: ("NBL 25/26", SeasonType.REGULAR),
    77353: ("NBL 2025",  SeasonType.REGULAR),
}

MAX_PAGES       = 100
RATE_DELAY_MS   = 400   # ms between page.goto calls
USER_AGENT      = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _parse_shooting_attempts(raw: Any) -> int | None:
    """
    Parse Sofascore shooting stat → attempt count.
    Handles "42/66", "42/66 (64%)", plain int, plain str.
    """
    if raw is None:
        return None
    s = str(raw)
    if "/" in s:
        try:
            return int(s.split("/")[1].split()[0])
        except (IndexError, ValueError):
            return None
    try:
        return int(s)
    except ValueError:
        return None


# ── Playwright API client ─────────────────────────────────────────────────────

class _SofaApiPage:
    """
    Wraps a Playwright page to fetch Sofascore API JSON via page.goto().
    Direct navigation to api.sofascore.com bypasses Cloudflare without cookies.
    """

    def __init__(self, page: Page) -> None:
        self._page = page
        self._call_count = 0

    async def get(self, path: str) -> dict:
        url = BASE_URL + path
        resp = await self._page.goto(url, wait_until="domcontentloaded", timeout=20_000)
        if not resp or resp.status != 200:
            status = resp.status if resp else "?"
            raise RuntimeError(f"HTTP {status} for {url}")
        body = await self._page.evaluate("() => document.body.innerText")
        self._call_count += 1
        if self._call_count > 1:
            await self._page.wait_for_timeout(RATE_DELAY_MS)
        return json.loads(body)

    async def fetch_all_events(self, season_id: int) -> list[dict]:
        events: list[dict] = []
        for page_num in range(MAX_PAGES):
            data = await self.get(
                f"/unique-tournament/{TOURNAMENT_ID}/season/{season_id}/events/last/{page_num}"
            )
            batch = data.get("events", [])
            events.extend(batch)
            log.info("  page %d → %d events (total so far: %d)", page_num, len(batch), len(events))
            if not data.get("hasNextPage") or not batch:
                break
        return events

    async def fetch_stats(self, event_id: int) -> dict | None:
        try:
            return await self.get(f"/event/{event_id}/statistics")
        except Exception as e:
            log.warning("  stats fetch failed for event %d: %s", event_id, e)
            return None


# ── Parsing ───────────────────────────────────────────────────────────────────

def _extract_quarter_rows(event: dict) -> list[QuarterStatRow]:
    hs, as_ = event.get("homeScore", {}), event.get("awayScore", {})
    rows = []
    for q in range(1, 5):
        h = _safe_int(hs.get(f"period{q}"))
        a = _safe_int(as_.get(f"period{q}"))
        if h is None and a is None:
            continue
        rows.append(QuarterStatRow(
            period_number=q,
            period_type=PeriodType.QUARTER,
            home_score=h,
            away_score=a,
        ))
    return rows


def _extract_ot_rows(event: dict) -> list[QuarterStatRow]:
    hs, as_ = event.get("homeScore", {}), event.get("awayScore", {})
    rows = []
    ot_num = 1
    for q in range(5, 10):
        h = _safe_int(hs.get(f"period{q}"))
        a = _safe_int(as_.get(f"period{q}"))
        if h is None and a is None:
            break
        rows.append(QuarterStatRow(
            period_number=ot_num,
            period_type=PeriodType.OVERTIME,
            home_score=h,
            away_score=a,
        ))
        ot_num += 1
    return rows


def _extract_game_stats_row(stats_data: dict) -> QuarterStatRow | None:
    """Full-game FGA/FTA stored as a single GAME-type row (period_number=1)."""
    for block in stats_data.get("statistics", []):
        if block.get("period", "").upper() != "ALL":
            continue
        metrics: dict[str, dict] = {}
        for grp in block.get("groups", []):
            for item in grp.get("statisticsItems", []):
                name = item.get("name", "").lower().strip()
                if name:
                    metrics[name] = {"home": item.get("home"), "away": item.get("away")}

        h_fga = _parse_shooting_attempts(metrics.get("field goals", {}).get("home"))
        a_fga = _parse_shooting_attempts(metrics.get("field goals", {}).get("away"))
        h_fta = _parse_shooting_attempts(metrics.get("free throws", {}).get("home"))
        a_fta = _parse_shooting_attempts(metrics.get("free throws", {}).get("away"))

        if any(v is not None for v in (h_fga, a_fga, h_fta, a_fta)):
            return QuarterStatRow(
                period_number=1,
                period_type=PeriodType.GAME,
                home_fga=h_fga,
                away_fga=a_fga,
                home_fta=h_fta,
                away_fta=a_fta,
            )
    return None


# ── DB insertion ──────────────────────────────────────────────────────────────

async def _process_event(
    event: dict,
    season_name: str,
    season_type: SeasonType,
    stats_data: dict | None,
    db: DbSession,
) -> str:
    if event.get("status", {}).get("type") != "finished":
        return "skip"

    event_id = event["id"]
    home_raw = event["homeTeam"]
    away_raw = event["awayTeam"]

    home_team = await crud.get_or_create_team(
        db,
        external_id=f"ss_{home_raw['id']}",
        name=home_raw.get("name", "Unknown"),
        abbreviation=(home_raw.get("shortName") or home_raw.get("name", "?"))[:8],
    )
    away_team = await crud.get_or_create_team(
        db,
        external_id=f"ss_{away_raw['id']}",
        name=away_raw.get("name", "Unknown"),
        abbreviation=(away_raw.get("shortName") or away_raw.get("name", "?"))[:8],
    )

    hs  = event.get("homeScore", {})
    as_ = event.get("awayScore", {})
    home_final = _safe_int(hs.get("current"))
    away_final = _safe_int(as_.get("current"))
    home_reg   = _safe_int(hs.get("normaltime"))
    away_reg   = _safe_int(as_.get("normaltime"))
    went_ot    = bool(home_final is not None and home_reg is not None and home_final != home_reg)
    ot_count   = sum(1 for q in range(5, 10) if _safe_int(hs.get(f"period{q}")) is not None)

    start_ts = event.get("startTimestamp")
    scheduled_at = (
        datetime.fromtimestamp(start_ts, tz=timezone.utc)
        if start_ts else datetime.now(tz=timezone.utc)
    )

    q_rows   = _extract_quarter_rows(event)
    ot_rows  = _extract_ot_rows(event)
    game_row = _extract_game_stats_row(stats_data) if stats_data else None
    all_rows = q_rows + ot_rows + ([game_row] if game_row else [])

    match = await crud.upsert_match(
        db,
        external_id=f"ss_{event_id}",
        home_team_id=home_team.id,
        away_team_id=away_team.id,
        scheduled_at=scheduled_at,
        season=season_name,
        status=MatchStatus.FINISHED,
        season_type=season_type,
        tournament_name=TOURNAMENT_NAME,
        home_score_final=home_final,
        away_score_final=away_final,
        home_score_regulation=home_reg,
        away_score_regulation=away_reg,
        went_to_overtime=went_ot,
        overtime_periods_count=ot_count,
        has_quarter_breakdown=(len(q_rows) == 4),
    )

    await crud.save_quarter_stats(db, match.id, all_rows)
    return "ok"


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def run(fetch_stats: bool = True) -> None:
    sf = get_session_factory()

    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(headless=True)
        ctx: BrowserContext = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        page: Page = await ctx.new_page()
        api = _SofaApiPage(page)

        for season_id, (season_name, season_type) in SEASONS.items():
            log.info("━" * 60)
            log.info("[ChinaNBL][%s]  Fetching event list…", season_name)
            all_events = await api.fetch_all_events(season_id)
            finished = [e for e in all_events if e.get("status", {}).get("type") == "finished"]
            log.info("[ChinaNBL][%s]  %d finished events found", season_name, len(finished))

            ok = skip = err = 0
            for i, event in enumerate(finished, 1):
                eid = event["id"]
                stats = await api.fetch_stats(eid) if fetch_stats else None
                try:
                    async with sf() as db:
                        result = await _process_event(event, season_name, season_type, stats, db)
                        await db.commit()
                    if result == "ok":
                        ok += 1
                    else:
                        skip += 1
                except Exception as e:
                    log.error("  event %d failed: %s", eid, e)
                    err += 1

                if i % 20 == 0 or i == len(finished):
                    log.info(
                        "[ChinaNBL][%s]  %d/%d | ok=%d skip=%d err=%d",
                        season_name, i, len(finished), ok, skip, err,
                    )

            log.info(
                "[ChinaNBL][%s]  DONE  ok=%d  skip=%d  err=%d",
                season_name, ok, skip, err,
            )

        await browser.close()

    await dispose_engine()

    # Final DB counts
    log.info("━" * 60)
    log.info("Collection complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Collect ChinaNBL from Sofascore")
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip per-match statistics (scores only, much faster)",
    )
    args = parser.parse_args()
    asyncio.run(run(fetch_stats=not args.no_stats))


if __name__ == "__main__":
    main()
