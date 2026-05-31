"""
ChinaNBL Sofascore API client and parsing helpers.

Uses a Playwright browser navigating directly to api.sofascore.com URLs —
Chromium's real TLS fingerprint bypasses Cloudflare without cookies.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from playwright.async_api import Page

from src.database.crud import QuarterStatRow
from src.database.models import PeriodType, SeasonType

log = logging.getLogger(__name__)

BASE_URL        = "https://api.sofascore.com/api/v1"
TOURNAMENT_ID   = 27568
TOURNAMENT_NAME = "ChinaNBL"
MAX_PAGES       = 100
RATE_DELAY_MS   = 400  # ms between api page.goto calls

SEASONS: dict[int, tuple[str, SeasonType]] = {
    87684: ("NBL 25/26", SeasonType.REGULAR),
    77353: ("NBL 2025",  SeasonType.REGULAR),
}


def _safe_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _parse_shooting_attempts(raw: Any) -> int | None:
    """Parse 'made/attempted (pct%)' or int → attempt count."""
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


class _SofaApiPage:
    """Wraps a Playwright page to fetch Sofascore API JSON via page.goto()."""

    def __init__(self, page: Page) -> None:
        self._page = page
        self._call_count = 0

    async def get(self, path: str) -> dict[str, Any]:
        """GET a ChinaNBL API path and return parsed JSON.

        Args:
            path: API path appended to the base URL.

        Returns:
            Parsed JSON body.

        Raises:
            RuntimeError: If the response is missing or non-200.
        """
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

    async def fetch_all_events(self, season_id: int) -> list[dict[str, Any]]:
        """Page through all events for a season.

        Args:
            season_id: ChinaNBL season id.

        Returns:
            Concatenated event dicts across all pages.
        """
        events: list[dict[str, Any]] = []
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

    async def fetch_stats(self, event_id: int) -> dict[str, Any] | None:
        """Fetch an event's statistics, returning None on failure.

        Args:
            event_id: ChinaNBL event id.

        Returns:
            Statistics JSON, or None if the request failed.
        """
        try:
            return await self.get(f"/event/{event_id}/statistics")
        except Exception as e:
            log.warning("  stats fetch failed for event %d: %s", event_id, e)
            return None


def extract_quarter_rows(event: dict[str, Any]) -> list[QuarterStatRow]:
    """Extract regulation-quarter score rows from an event payload.

    Args:
        event: Event dict with ``homeScore`` / ``awayScore`` period fields.

    Returns:
        Quarter stat rows for periods that have at least one score.
    """
    hs, as_ = event.get("homeScore", {}), event.get("awayScore", {})
    rows = []
    for q in range(1, 5):
        h = _safe_int(hs.get(f"period{q}"))
        a = _safe_int(as_.get(f"period{q}"))
        if h is None and a is None:
            continue
        rows.append(QuarterStatRow(period_number=q, period_type=PeriodType.QUARTER,
                                   home_score=h, away_score=a))
    return rows


def extract_ot_rows(event: dict[str, Any]) -> list[QuarterStatRow]:
    """Extract overtime score rows from an event payload.

    Args:
        event: Event dict with ``homeScore`` / ``awayScore`` period fields.

    Returns:
        Overtime stat rows (renumbered from 1), stopping at the first empty OT.
    """
    hs, as_ = event.get("homeScore", {}), event.get("awayScore", {})
    rows = []
    ot_num = 1
    for q in range(5, 10):
        h = _safe_int(hs.get(f"period{q}"))
        a = _safe_int(as_.get(f"period{q}"))
        if h is None and a is None:
            break
        rows.append(QuarterStatRow(period_number=ot_num, period_type=PeriodType.OVERTIME,
                                   home_score=h, away_score=a))
        ot_num += 1
    return rows


def extract_game_stats_row(stats_data: dict[str, Any]) -> QuarterStatRow | None:
    """Full-game FGA/FTA stored as a single GAME-type row (period_number=1)."""
    for block in stats_data.get("statistics", []):
        if block.get("period", "").upper() != "ALL":
            continue
        metrics: dict[str, dict[str, Any]] = {}
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
                period_number=1, period_type=PeriodType.GAME,
                home_fga=h_fga, away_fga=a_fga,
                home_fta=h_fta, away_fta=a_fta,
            )
    return None
