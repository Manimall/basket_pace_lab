"""
Sofascore async client (curl_cffi — imitates Chrome TLS fingerprint).

Usage:
    async with SofascoreClient() as client:
        match = await client.process_and_save_match(event_id, db_session)

    async with SofascoreClient(mock=True) as client:
        ...  # reads from tests/fixtures/*.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from curl_cffi.requests import AsyncSession
from sqlalchemy.ext.asyncio import AsyncSession as DbSession

from src.data_collection.sofascore.parsers import (
    PERIOD_MAP,
    _calc_pace,
    _calc_possessions,
    build_metrics,
    parse_shot,
    safe_int,
)
from src.database import crud
from src.database.crud import QuarterStatRow
from src.database.models import MatchStatus, SeasonType

log = logging.getLogger(__name__)

BASE_URL   = "https://api.sofascore.com/api/v1"
WARMUP_URL = "https://www.sofascore.com/"

HEADERS: dict[str, str] = {
    "Accept":           "application/json, text/plain, */*",
    "Accept-Language":  "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Origin":           "https://www.sofascore.com",
    "Referer":          "https://www.sofascore.com/",
    "sec-fetch-dest":   "empty",
    "sec-fetch-mode":   "cors",
    "sec-fetch-site":   "same-site",
    "Cache-Control":    "no-cache",
    "Pragma":           "no-cache",
}

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures"
COOKIES_PATH = Path(__file__).parent.parent.parent.parent / "cookies.json"


def _load_cookies() -> list[dict] | None:
    if not COOKIES_PATH.exists():
        return None
    try:
        return json.loads(COOKIES_PATH.read_text())
    except Exception:
        return None


class SofascoreClient:
    """Async client for the Sofascore internal API."""

    def __init__(self, impersonate: str = "chrome124", mock: bool = False) -> None:
        self._impersonate = impersonate
        self._mock = mock
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> "SofascoreClient":
        if not self._mock:
            # _impersonate is a curl_cffi profile string; curl_cffi types the
            # param as a Literal, so a plain str needs an explicit ignore.
            self._session = AsyncSession(impersonate=self._impersonate, headers=HEADERS)  # type: ignore[arg-type]
            await self._load_saved_cookies()
        else:
            log.info("Mock mode: API requests replaced with local fixtures")
        return self

    async def _load_saved_cookies(self) -> None:
        assert self._session
        cookies = _load_cookies()
        if cookies:
            log.info("Loading %d cookies from %s", len(cookies), COOKIES_PATH.name)
            for c in cookies:
                try:
                    self._session.cookies.set(c["name"], c["value"], domain=c.get("domain", ""))
                except Exception:
                    pass
            log.info("Cookies loaded — skipping warmup request")
        else:
            log.warning(
                "cookies.json not found. Run 'python tools/fetch_cookies.py' first.\n"
                "  Falling back to warmup request (may result in 403)."
            )
            await self._warmup()

    async def _warmup(self) -> None:
        assert self._session
        try:
            resp = await self._session.get(
                WARMUP_URL,
                headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
                timeout=15,
            )
            log.info("Warmup status: %s | cookies: %d", resp.status_code, len(self._session.cookies))
        except Exception as exc:
            log.warning("Warmup failed: %s", exc)

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def _get(self, path: str) -> dict[str, Any]:
        assert self._session, "Use SofascoreClient as async context manager."
        url = BASE_URL + path
        log.debug("GET %s", url)
        response = await self._session.get(url, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(f"Sofascore API returned {response.status_code} for {url}")
        return response.json()

    def _load_fixture(self, name: str) -> dict[str, Any]:
        path = FIXTURES_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Fixture not found: {path}")
        return json.loads(path.read_text())

    async def _fetch_event(self, event_id: int) -> dict[str, Any]:
        if self._mock:
            return self._load_fixture(f"event_{event_id}.json")["event"]
        data = await self._get(f"/event/{event_id}")
        return data["event"]

    async def _fetch_match_stats(self, event_id: int) -> dict[str, Any]:
        if self._mock:
            return self._load_fixture(f"statistics_{event_id}.json")
        return await self._get(f"/event/{event_id}/statistics")

    def _parse_period(self, period_label: str, period_data: dict[str, Any]) -> QuarterStatRow | None:
        mapping = PERIOD_MAP.get(period_label.upper())
        if mapping is None:
            return None
        period_number, period_type = mapping
        m = build_metrics(period_data)

        home_score = safe_int(m.get("points", {}).get("home"))
        away_score = safe_int(m.get("points", {}).get("away"))

        _, home_fga = parse_shot(m.get("field goals", {}).get("home", ""))
        _, away_fga = parse_shot(m.get("field goals", {}).get("away", ""))
        _, home_fta = parse_shot(m.get("free throws", {}).get("home", ""))
        _, away_fta = parse_shot(m.get("free throws", {}).get("away", ""))

        home_off = safe_int(m.get("offensive rebounds", {}).get("home"))
        away_off = safe_int(m.get("offensive rebounds", {}).get("away"))
        home_to  = safe_int(m.get("turnovers", {}).get("home"))
        away_to  = safe_int(m.get("turnovers", {}).get("away"))

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

    async def process_and_save_match(self, event_id: int, db_session: DbSession) -> crud.Match:
        """Full pipeline: fetch event + stats → upsert teams/match → save QuarterStats."""
        log.info("=== Processing event_id=%s ===", event_id)
        event = await self._fetch_event(event_id)

        home_raw: dict[str, Any] = event["homeTeam"]
        away_raw: dict[str, Any] = event["awayTeam"]

        home_team = await crud.get_or_create_team(
            db_session, str(home_raw["id"]),
            home_raw.get("name", "Unknown"),
            home_raw.get("shortName", home_raw.get("name", "?"))[:8],
        )
        away_team = await crud.get_or_create_team(
            db_session, str(away_raw["id"]),
            away_raw.get("name", "Unknown"),
            away_raw.get("shortName", away_raw.get("name", "?"))[:8],
        )

        home_score_raw: dict[str, Any] = event.get("homeScore", {})
        away_score_raw: dict[str, Any] = event.get("awayScore", {})

        def _sum_quarters(score: dict[str, Any]) -> int | None:
            vals = [score.get(f"period{i}") for i in range(1, 5)]
            return sum(int(v) for v in vals if v is not None) or None

        went_ot = bool(home_score_raw.get("overtime") or away_score_raw.get("overtime"))
        status_map = {
            "finished": MatchStatus.FINISHED, "inprogress": MatchStatus.LIVE,
            "notstarted": MatchStatus.SCHEDULED, "postponed": MatchStatus.POSTPONED,
            "cancelled": MatchStatus.CANCELLED,
        }
        status = status_map.get(event.get("status", {}).get("type", "finished"), MatchStatus.FINISHED)
        start_ts = event.get("startTimestamp")
        scheduled_at = (
            datetime.fromtimestamp(start_ts, tz=timezone.utc)
            if start_ts else datetime.now(tz=timezone.utc)
        )

        match = await crud.upsert_match(
            db_session, external_id=str(event_id),
            home_team_id=home_team.id, away_team_id=away_team.id,
            scheduled_at=scheduled_at, season=event.get("season", {}).get("name", "unknown"),
            status=status, season_type=SeasonType.REGULAR,
            home_score_final=safe_int(home_score_raw.get("current")),
            away_score_final=safe_int(away_score_raw.get("current")),
            home_score_regulation=_sum_quarters(home_score_raw),
            away_score_regulation=_sum_quarters(away_score_raw),
            went_to_overtime=went_ot,
        )

        stats_data = await self._fetch_match_stats(event_id)
        rows: list[QuarterStatRow] = [
            row for pd in stats_data.get("statistics", [])
            if (row := self._parse_period(pd.get("period", ""), pd)) is not None
        ]
        await crud.save_quarter_stats(db_session, match.id, rows)
        log.info("=== Done: event_id=%s match_id=%s ===", event_id, match.id)
        return match
