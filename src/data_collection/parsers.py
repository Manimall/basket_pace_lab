"""
Shared Sofascore response parsers for period statistics.

Used by both collect_history.py and mass_scheduler.py.
"""
from __future__ import annotations

from typing import Any

from src.database.crud import QuarterStatRow
from src.database.models import PeriodType
from src.data_collection.sofascore_client import (
    FTA_TO_POSS_FACTOR,
    _calc_pace,
    _calc_possessions,
)

# Sofascore period label → (period_number, PeriodType)
# NBA uses "1Q".."4Q"; EuroLeague/other use "1ST".."4TH"
PERIOD_MAP: dict[str, tuple[int, PeriodType]] = {
    "1Q":  (1, PeriodType.QUARTER),
    "2Q":  (2, PeriodType.QUARTER),
    "3Q":  (3, PeriodType.QUARTER),
    "4Q":  (4, PeriodType.QUARTER),
    "1ST": (1, PeriodType.QUARTER),
    "2ND": (2, PeriodType.QUARTER),
    "3RD": (3, PeriodType.QUARTER),
    "4TH": (4, PeriodType.QUARTER),
    "OT":  (1, PeriodType.OVERTIME),
    "1OT": (1, PeriodType.OVERTIME),
    "2OT": (2, PeriodType.OVERTIME),
    "3OT": (3, PeriodType.OVERTIME),
}

# Period label → key in homeScore / awayScore event object
SCORE_KEY_MAP: dict[str, str] = {
    "1Q":  "period1", "1ST": "period1",
    "2Q":  "period2", "2ND": "period2",
    "3Q":  "period3", "3RD": "period3",
    "4Q":  "period4", "4TH": "period4",
    "OT":  "overtime", "1OT": "overtime",
    "2OT": "overtime2", "3OT": "overtime3",
}


def safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def parse_shot(s: str) -> tuple[int, int]:
    """Parse 'made/attempted (pct%)' or 'made/attempted' → (made, attempted)."""
    try:
        parts = str(s).split("/")
        made = int(parts[0].strip())
        att  = int(parts[1].strip().split()[0].rstrip("(").strip())
        return made, att
    except Exception:
        return 0, 0


def build_metrics(period_data: dict) -> dict[str, dict]:
    m: dict[str, dict] = {}
    for g in period_data.get("groups", []):
        for item in g.get("statisticsItems", []):
            name = item.get("name", "").lower().strip()
            if name:
                m[name] = {"home": item.get("home"), "away": item.get("away")}
    return m


def parse_full_game(
    period_data: dict,
    home_score: int | None,
    away_score: int | None,
) -> QuarterStatRow | None:
    """Parse 'ALL' full-game stats as a game-level pace proxy (EuroLeague fallback).

    Returns a single QuarterStatRow with period_type=GAME, period_number=1.
    Returns None if possession stats are unavailable.
    """
    m = build_metrics(period_data)

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

    if home_poss is None and away_poss is None:
        return None

    return QuarterStatRow(
        period_number=1,
        period_type=PeriodType.GAME,
        home_score=home_score,
        away_score=away_score,
        home_fga=home_fga or None,
        away_fga=away_fga or None,
        home_fta=home_fta or None,
        away_fta=away_fta or None,
        home_off_reb=home_off,
        away_off_reb=away_off,
        home_turnovers=home_to,
        away_turnovers=away_to,
        home_possessions=home_poss,
        away_possessions=away_poss,
        home_pace=_calc_pace(home_poss, PeriodType.GAME),
        away_pace=_calc_pace(away_poss, PeriodType.GAME),
    )


def parse_period(
    label: str,
    period_data: dict,
    home_score: int | None,
    away_score: int | None,
) -> QuarterStatRow | None:
    mapping = PERIOD_MAP.get(label.upper())
    if not mapping:
        return None
    period_number, period_type = mapping
    m = build_metrics(period_data)

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
        period_number=period_number,
        period_type=period_type,
        home_score=home_score,
        away_score=away_score,
        home_fga=home_fga or None,
        away_fga=away_fga or None,
        home_fta=home_fta or None,
        away_fta=away_fta or None,
        home_off_reb=home_off,
        away_off_reb=away_off,
        home_turnovers=home_to,
        away_turnovers=away_to,
        home_possessions=home_poss,
        away_possessions=away_poss,
        home_pace=_calc_pace(home_poss, period_type),
        away_pace=_calc_pace(away_poss, period_type),
    )
