"""Sofascore HTTP access for the ID finder: cookies, headers, event fetching."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from curl_cffi.requests import AsyncSession

from src.data_collection.flashscore.norm import _norm
from src.data_collection.sofascore.id_finder.config import (
    BASE_HEADERS,
    BASE_URL,
    COOKIES_PATH,
    HTTP_OK,
    HTTP_TIMEOUT_SEC,
    MAX_PAGES,
    PAGE_DELAY_SEC,
)
from src.data_collection.sofascore.id_finder.models import SofaEvent

log = logging.getLogger(__name__)

_FINISHED: str = "finished"


def load_cookie_header() -> str:
    """Render cookies.json as a ``name=value; …`` Cookie header (Go-scout style)."""
    if not COOKIES_PATH.exists():
        return ""
    try:
        cookies = json.loads(COOKIES_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Не удалось прочитать cookies.json: %s", exc)
        return ""
    if not isinstance(cookies, list):
        return ""
    return "; ".join(
        f"{c['name']}={c.get('value', '')}" for c in cookies if c.get("name")
    )


def build_headers(cookie_header: str) -> dict[str, str]:
    """Return request headers, adding the Cookie header when present."""
    headers = dict(BASE_HEADERS)
    if cookie_header:
        headers["Cookie"] = cookie_header
    return headers


async def api_get(
    session: AsyncSession, path: str, headers: dict[str, str],
) -> dict | None:
    """GET a Sofascore API path; return parsed JSON or None on non-200/error."""
    url = f"{BASE_URL}{path}"
    try:
        resp = await session.get(url, headers=headers, timeout=HTTP_TIMEOUT_SEC)
        if resp.status_code != HTTP_OK:
            log.debug("API %s → %d", path, resp.status_code)
            return None
        return resp.json()
    except Exception as exc:  # noqa: BLE001 — network errors are expected, treated as miss
        log.debug("API error %s: %s", path, exc)
        return None


def _parse_event(ev: dict) -> SofaEvent | None:
    if ev.get("status", {}).get("type") != _FINISHED:
        return None
    ts = ev.get("startTimestamp")
    home_raw = ev.get("homeTeam", {}).get("name", "")
    away_raw = ev.get("awayTeam", {}).get("name", "")
    if not ts or not home_raw or not away_raw:
        return None
    return SofaEvent(
        event_id   = int(ev["id"]),
        event_date = datetime.fromtimestamp(ts, tz=timezone.utc).date(),
        home_raw   = home_raw,
        away_raw   = away_raw,
        home_norm  = _norm(home_raw),
        away_norm  = _norm(away_raw),
    )


async def fetch_events_for_season(
    session: AsyncSession,
    tournament_id: int,
    season_id: int,
    headers: dict[str, str],
) -> list[SofaEvent]:
    """Page through all finished events for one season (rate-limit-friendly)."""
    events: list[SofaEvent] = []
    for page in range(MAX_PAGES):
        if page > 0:
            await asyncio.sleep(PAGE_DELAY_SEC)
        data = await api_get(
            session,
            f"/unique-tournament/{tournament_id}/season/{season_id}/events/last/{page}",
            headers,
        )
        if not data:
            break
        raw_events = data.get("events", [])
        events.extend(ev for ev in map(_parse_event, raw_events) if ev is not None)
        if not data.get("hasNextPage", True) or not raw_events:
            break
    return events
