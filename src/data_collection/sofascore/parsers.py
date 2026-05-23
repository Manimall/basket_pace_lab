"""
Shared Sofascore response parsers: possession math, period mapping, stat extraction.

Used by SofascoreClient, MassScheduler event processing, and legacy parsers.py.
"""
from __future__ import annotations

from typing import Any

from src.database.crud import QuarterStatRow
from src.database.models import PeriodType

# Dean Oliver coefficient: possessions consumed per free throw attempt
FTA_TO_POSS_FACTOR: float = 0.44

MINUTES_PER_QUARTER: int = 10
MINUTES_PER_OT: int = 5
MINUTES_PER_GAME: int = 40
PACE_NORMALISATION_MINUTES: int = 40  # normalise to FIBA 40-min game

# Sofascore period label → (period_number, PeriodType)
# Covers NBA ("1Q"…"4Q"), FIBA ("1ST"…"4TH"), and OT variants
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
    "OT1": (1, PeriodType.OVERTIME),
    "OT2": (2, PeriodType.OVERTIME),
    "OT3": (3, PeriodType.OVERTIME),
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
    "OT":  "overtime", "OT1": "overtime", "1OT": "overtime",
    "OT2": "overtime2", "2OT": "overtime2",
    "OT3": "overtime3", "3OT": "overtime3",
}


def _calc_possessions(
    fga: int | None,
    fta: int | None,
    off_reb: int | None,
    turnovers: int | None,
) -> float | None:
    if any(v is None for v in (fga, fta, off_reb, turnovers)):
        return None
    return fga + FTA_TO_POSS_FACTOR * fta - off_reb + turnovers  # type: ignore[operator]


def _calc_pace(possessions: float | None, period_type: PeriodType) -> float | None:
    if possessions is None:
        return None
    if period_type == PeriodType.GAME:
        minutes = MINUTES_PER_GAME
    elif period_type == PeriodType.OVERTIME:
        minutes = MINUTES_PER_OT
    else:
        minutes = MINUTES_PER_QUARTER
    return round(possessions / minutes * PACE_NORMALISATION_MINUTES, 2)


def safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def parse_shot(s: Any) -> tuple[int, int]:
    """'made/attempted (pct%)' or 'made/attempted' → (made, attempted)."""
    try:
        parts = str(s).split("/")
        made = int(parts[0].strip())
        att  = int(parts[1].strip().split()[0].rstrip("(").strip())
        return made, att
    except Exception:
        return 0, 0


def build_metrics(period_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Flatten groups/statisticsItems into {metric_name: {home, away}}."""
    m: dict[str, dict[str, Any]] = {}
    for g in period_data.get("groups", []):
        for item in g.get("statisticsItems", []):
            name = item.get("name", "").lower().strip()
            if name:
                m[name] = {"home": item.get("home"), "away": item.get("away")}
    return m


def _extract_shooting(
    metrics: dict[str, dict[str, Any]],
) -> tuple[int | None, int | None, int | None, int | None,
           int | None, int | None, int | None, int | None]:
    """Return (home_fga, away_fga, home_fta, away_fta, h_off, a_off, h_to, a_to)."""
    _, home_fga = parse_shot(metrics.get("field goals", {}).get("home", ""))
    _, away_fga = parse_shot(metrics.get("field goals", {}).get("away", ""))
    _, home_fta = parse_shot(metrics.get("free throws", {}).get("home", ""))
    _, away_fta = parse_shot(metrics.get("free throws", {}).get("away", ""))
    home_off = safe_int(metrics.get("offensive rebounds", {}).get("home"))
    away_off = safe_int(metrics.get("offensive rebounds", {}).get("away"))
    home_to  = safe_int(metrics.get("turnovers", {}).get("home"))
    away_to  = safe_int(metrics.get("turnovers", {}).get("away"))
    return (
        home_fga or None, away_fga or None,
        home_fta or None, away_fta or None,
        home_off, away_off, home_to, away_to,
    )


def parse_full_game(
    period_data: dict[str, Any],
    home_score: int | None,
    away_score: int | None,
) -> QuarterStatRow | None:
    """Parse 'ALL' full-game stats as a GAME-level pace proxy."""
    m = build_metrics(period_data)
    hfga, afga, hfta, afta, hoff, aoff, hto, ato = _extract_shooting(m)
    home_poss = _calc_possessions(hfga, hfta, hoff, hto)
    away_poss = _calc_possessions(afga, afta, aoff, ato)
    if home_poss is None and away_poss is None:
        return None
    return QuarterStatRow(
        period_number=1, period_type=PeriodType.GAME,
        home_score=home_score, away_score=away_score,
        home_fga=hfga, away_fga=afga,
        home_fta=hfta, away_fta=afta,
        home_off_reb=hoff, away_off_reb=aoff,
        home_turnovers=hto, away_turnovers=ato,
        home_possessions=home_poss, away_possessions=away_poss,
        home_pace=_calc_pace(home_poss, PeriodType.GAME),
        away_pace=_calc_pace(away_poss, PeriodType.GAME),
    )


def parse_period(
    label: str,
    period_data: dict[str, Any],
    home_score: int | None,
    away_score: int | None,
) -> QuarterStatRow | None:
    mapping = PERIOD_MAP.get(label.upper())
    if not mapping:
        return None
    period_number, period_type = mapping
    m = build_metrics(period_data)
    hfga, afga, hfta, afta, hoff, aoff, hto, ato = _extract_shooting(m)
    home_poss = _calc_possessions(hfga, hfta, hoff, hto)
    away_poss = _calc_possessions(afga, afta, aoff, ato)
    return QuarterStatRow(
        period_number=period_number, period_type=period_type,
        home_score=home_score, away_score=away_score,
        home_fga=hfga, away_fga=afga,
        home_fta=hfta, away_fta=afta,
        home_off_reb=hoff, away_off_reb=aoff,
        home_turnovers=hto, away_turnovers=ato,
        home_possessions=home_poss, away_possessions=away_poss,
        home_pace=_calc_pace(home_poss, period_type),
        away_pace=_calc_pace(away_poss, period_type),
    )
