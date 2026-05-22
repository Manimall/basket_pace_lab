"""
Sofascore async client.

Использует curl_cffi для имитации TLS-фингерпринта Chrome,
что позволяет обходить Cloudflare Bot Management без браузера.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from curl_cffi.requests import AsyncSession
from sqlalchemy.ext.asyncio import AsyncSession as DbSession

from src.database import crud
from src.database.crud import QuarterStatRow
from src.database.models import MatchStatus, PeriodType, SeasonType

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.sofascore.com/api/v1"
WARMUP_URL = "https://www.sofascore.com/"

HEADERS: dict[str, str] = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Origin": "https://www.sofascore.com",
    "Referer": "https://www.sofascore.com/",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# Sofascore period label → (period_number, PeriodType)
PERIOD_MAP: dict[str, tuple[int, PeriodType]] = {
    "1ST": (1, PeriodType.QUARTER),
    "2ND": (2, PeriodType.QUARTER),
    "3RD": (3, PeriodType.QUARTER),
    "4TH": (4, PeriodType.QUARTER),
    "OT":  (1, PeriodType.OVERTIME),
    "OT1": (1, PeriodType.OVERTIME),
    "OT2": (2, PeriodType.OVERTIME),
    "OT3": (3, PeriodType.OVERTIME),
}

# FIBA regulation quarter = 10 min. NBA = 12 min. OT = 5 min.
MINUTES_PER_QUARTER = 10
MINUTES_PER_OT = 5
MINUTES_PER_GAME = 40  # full FIBA game (4 × 10 min)

# Pace normalised to 40-minute game (FIBA)
PACE_NORMALISATION_MINUTES = 40

# Dean Oliver coefficient: average possessions consumed per FTA
FTA_TO_POSS_FACTOR = 0.44


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calc_possessions(
    fga: int | None,
    fta: int | None,
    off_reb: int | None,
    turnovers: int | None,
) -> float | None:
    """
    Standard possession estimate:
        Poss = FGA + FTA_TO_POSS_FACTOR * FTA - OffReb + TO
    Returns None if any component is missing.
    """
    if any(v is None for v in (fga, fta, off_reb, turnovers)):
        return None
    return fga + FTA_TO_POSS_FACTOR * fta - off_reb + turnovers  # type: ignore[operator]


def _calc_pace(possessions: float | None, period_type: PeriodType) -> float | None:
    if period_type == PeriodType.GAME:
        minutes = MINUTES_PER_GAME
    elif period_type == PeriodType.OVERTIME:
        minutes = MINUTES_PER_OT
    else:
        minutes = MINUTES_PER_QUARTER
    if possessions is None:
        return None
    return round(possessions / minutes * PACE_NORMALISATION_MINUTES, 2)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"
COOKIES_PATH = Path(__file__).parent.parent.parent / "cookies.json"


def _load_cookies() -> list[dict] | None:
    if not COOKIES_PATH.exists():
        return None
    try:
        return json.loads(COOKIES_PATH.read_text())
    except Exception:
        return None


class SofascoreClient:
    """
    Async client for the Sofascore internal API.

    Порядок аутентификации:
      1. Если есть cookies.json (от fetch_cookies.py) — использует их. Работает без браузера.
      2. Иначе — пробует warmup через curl_cffi (куки выставляются JS, поэтому часто 403).
      3. mock=True — читает данные из tests/fixtures/*.json, без сети.

    Usage:
        # Реальный API (нужен cookies.json):
        async with SofascoreClient() as client: ...

        # Локальные фикстуры:
        async with SofascoreClient(mock=True) as client: ...
    """

    def __init__(self, impersonate: str = "chrome124", mock: bool = False) -> None:
        self._impersonate = impersonate
        self._mock = mock
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> "SofascoreClient":
        if not self._mock:
            self._session = AsyncSession(impersonate=self._impersonate, headers=HEADERS)
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
                # curl_cffi принимает куки через атрибут .cookies (dict-like)
                try:
                    self._session.cookies.set(
                        c["name"], c["value"], domain=c.get("domain", "")
                    )
                except Exception:
                    pass
            log.info("Cookies loaded — skipping warmup request")
        else:
            log.warning(
                "cookies.json not found. Run 'python fetch_cookies.py' first for live API.\n"
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

    # ------------------------------------------------------------------
    # Private fetch methods
    # ------------------------------------------------------------------

    async def _get(self, path: str) -> dict[str, Any]:
        assert self._session, "Use SofascoreClient as async context manager."
        url = BASE_URL + path
        log.debug("GET %s", url)
        response = await self._session.get(url, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(
                f"Sofascore API returned {response.status_code} for {url}\n"
                f"Headers: {dict(response.headers)}"
            )
        return response.json()

    def _load_fixture(self, name: str) -> dict[str, Any]:
        path = FIXTURES_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Fixture not found: {path}")
        return json.loads(path.read_text())

    async def _fetch_event(self, event_id: int) -> dict[str, Any]:
        if self._mock:
            log.info("  [mock] loading event_%s.json", event_id)
            return self._load_fixture(f"event_{event_id}.json")["event"]
        data = await self._get(f"/event/{event_id}")
        return data["event"]

    async def _fetch_match_stats(self, event_id: int) -> dict[str, Any]:
        if self._mock:
            log.info("  [mock] loading statistics_%s.json", event_id)
            return self._load_fixture(f"statistics_{event_id}.json")
        return await self._get(f"/event/{event_id}/statistics")

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_shooting_stat(stat_string: str) -> tuple[int, int]:
        """
        "32/71" → (32, 71).  Если строка кривая — (0, 0).
        Sofascore отдаёт shooting stats именно в формате "made/attempted".
        """
        try:
            parts = str(stat_string).split("/")
            if len(parts) != 2:
                return (0, 0)
            return (int(parts[0]), int(parts[1]))
        except (ValueError, AttributeError):
            return (0, 0)

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        """Мягко конвертирует значение из JSON в int."""
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _build_metrics(period_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Разворачивает список groups/statisticsItems в плоский словарь:
            metric_name.lower() → {"home": ..., "away": ...}

        Это позволяет искать метрики без завязки на конкретный индекс группы.
        """
        metrics: dict[str, dict[str, Any]] = {}
        for group in period_data.get("groups", []):
            for item in group.get("statisticsItems", []):
                name: str = item.get("name", "").lower().strip()
                if name:
                    metrics[name] = {
                        "home": item.get("home"),
                        "away": item.get("away"),
                    }
        return metrics

    def _parse_period(
        self,
        period_label: str,
        period_data: dict[str, Any],
    ) -> QuarterStatRow | None:
        """
        Конвертирует один период из Sofascore JSON в QuarterStatRow.
        Возвращает None если period_label неизвестен (например, "ALL").
        """
        mapping = PERIOD_MAP.get(period_label.upper())
        if mapping is None:
            return None

        period_number, period_type = mapping
        metrics = self._build_metrics(period_data)

        # Score
        home_score = self._safe_int(metrics.get("points", {}).get("home"))
        away_score = self._safe_int(metrics.get("points", {}).get("away"))

        # Field Goals: "32/71" → made=32, attempted=71
        fg_raw_h = metrics.get("field goals", {}).get("home", "")
        fg_raw_a = metrics.get("field goals", {}).get("away", "")
        _, home_fga = self._parse_shooting_stat(fg_raw_h)
        _, away_fga = self._parse_shooting_stat(fg_raw_a)

        # Free Throws: "11/14"
        ft_raw_h = metrics.get("free throws", {}).get("home", "")
        ft_raw_a = metrics.get("free throws", {}).get("away", "")
        _, home_fta = self._parse_shooting_stat(ft_raw_h)
        _, away_fta = self._parse_shooting_stat(ft_raw_a)

        # Rebounds
        home_off_reb = self._safe_int(metrics.get("offensive rebounds", {}).get("home"))
        away_off_reb = self._safe_int(metrics.get("offensive rebounds", {}).get("away"))

        # Turnovers
        home_to = self._safe_int(metrics.get("turnovers", {}).get("home"))
        away_to = self._safe_int(metrics.get("turnovers", {}).get("away"))

        # Derived: Possessions & Pace
        home_poss = _calc_possessions(home_fga or None, home_fta or None, home_off_reb, home_to)
        away_poss = _calc_possessions(away_fga or None, away_fta or None, away_off_reb, away_to)
        home_pace = _calc_pace(home_poss, period_type)
        away_pace = _calc_pace(away_poss, period_type)

        log.info(
            "  [%s] score %s-%s | FGA %s/%s | FTA %s/%s | OffReb %s/%s | TO %s/%s | Poss %.1f/%.1f | Pace %.1f/%.1f",
            period_label,
            home_score, away_score,
            home_fga, away_fga,
            home_fta, away_fta,
            home_off_reb, away_off_reb,
            home_to, away_to,
            home_poss or 0, away_poss or 0,
            home_pace or 0, away_pace or 0,
        )

        return QuarterStatRow(
            period_number=period_number,
            period_type=period_type,
            home_score=home_score,
            away_score=away_score,
            home_fga=home_fga or None,
            away_fga=away_fga or None,
            home_fta=home_fta or None,
            away_fta=away_fta or None,
            home_off_reb=home_off_reb,
            away_off_reb=away_off_reb,
            home_turnovers=home_to,
            away_turnovers=away_to,
            home_possessions=home_poss,
            away_possessions=away_poss,
            home_pace=home_pace,
            away_pace=away_pace,
        )

    # ------------------------------------------------------------------
    # Public orchestration
    # ------------------------------------------------------------------

    async def process_and_save_match(
        self,
        event_id: int,
        db_session: DbSession,
    ) -> crud.Match:
        """
        Полный цикл для одного матча:
          1. Загрузить детали события → upsert команды → upsert матч
          2. Загрузить statistics → распарсить по периодам
          3. Сохранить QuarterStats в БД
        """
        log.info("=== Processing event_id=%s ===", event_id)

        # 1. Event metadata (команды, счёт, время)
        log.info("[1/3] Fetching event metadata …")
        event = await self._fetch_event(event_id)

        home_raw: dict[str, Any] = event["homeTeam"]
        away_raw: dict[str, Any] = event["awayTeam"]

        home_team = await crud.get_or_create_team(
            db_session,
            external_id=str(home_raw["id"]),
            name=home_raw.get("name", "Unknown"),
            abbreviation=home_raw.get("shortName", home_raw.get("name", "?"))[:8],
        )
        away_team = await crud.get_or_create_team(
            db_session,
            external_id=str(away_raw["id"]),
            name=away_raw.get("name", "Unknown"),
            abbreviation=away_raw.get("shortName", away_raw.get("name", "?"))[:8],
        )
        log.info("  Teams: %s vs %s", home_team.name, away_team.name)

        home_score_raw: dict[str, Any] = event.get("homeScore", {})
        away_score_raw: dict[str, Any] = event.get("awayScore", {})

        home_final = self._safe_int(home_score_raw.get("current"))
        away_final = self._safe_int(away_score_raw.get("current"))

        # Регуляционный счёт = сумма 4 четвертей
        def _sum_quarters(score: dict[str, Any]) -> int | None:
            vals = [score.get(f"period{i}") for i in range(1, 5)]
            return sum(int(v) for v in vals if v is not None) or None

        home_reg = _sum_quarters(home_score_raw)
        away_reg = _sum_quarters(away_score_raw)

        went_ot = bool(home_score_raw.get("overtime") or away_score_raw.get("overtime"))

        status_raw: str = event.get("status", {}).get("type", "finished")
        status_map = {
            "finished": MatchStatus.FINISHED,
            "inprogress": MatchStatus.LIVE,
            "notstarted": MatchStatus.SCHEDULED,
            "postponed": MatchStatus.POSTPONED,
            "cancelled": MatchStatus.CANCELLED,
        }
        status = status_map.get(status_raw, MatchStatus.FINISHED)

        start_ts: int | None = event.get("startTimestamp")
        scheduled_at = (
            datetime.fromtimestamp(start_ts, tz=timezone.utc)
            if start_ts else datetime.now(tz=timezone.utc)
        )

        season_name: str = event.get("season", {}).get("name", "unknown")

        match = await crud.upsert_match(
            db_session,
            external_id=str(event_id),
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            scheduled_at=scheduled_at,
            season=season_name,
            status=status,
            season_type=SeasonType.REGULAR,
            home_score_final=home_final,
            away_score_final=away_final,
            home_score_regulation=home_reg,
            away_score_regulation=away_reg,
            went_to_overtime=went_ot,
        )
        log.info(
            "  Match upserted: id=%s  score=%s-%s  reg=%s-%s  OT=%s",
            match.id, home_final, away_final, home_reg, away_reg, went_ot,
        )

        # 2. Statistics per period
        log.info("[2/3] Fetching statistics …")
        stats_data = await self._fetch_match_stats(event_id)
        periods: list[dict[str, Any]] = stats_data.get("statistics", [])
        log.info("  Found %d period blocks", len(periods))

        rows: list[QuarterStatRow] = []
        for period_data in periods:
            label: str = period_data.get("period", "")
            row = self._parse_period(label, period_data)
            if row is not None:
                rows.append(row)

        # 3. Persist
        log.info("[3/3] Saving %d QuarterStats rows for match_id=%s …", len(rows), match.id)
        await crud.save_quarter_stats(db_session, match.id, rows)
        log.info("=== Done: event_id=%s match_id=%s ===", event_id, match.id)

        return match
