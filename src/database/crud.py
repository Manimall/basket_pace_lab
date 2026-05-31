from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Match, PeriodType, QuarterStats, Team

# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------

async def get_or_create_team(
    session: AsyncSession,
    external_id: str,
    name: str,
    abbreviation: str,
    conference: str | None = None,
    division: str | None = None,
) -> Team:
    """Fetch a team by external_id or create it if absent.

    Args:
        session: Active async DB session.
        external_id: Source-unique team identifier.
        name: Full team name.
        abbreviation: Short team code.
        conference: Optional conference label.
        division: Optional division label.

    Returns:
        The existing or newly-created ``Team``.
    """
    result = await session.execute(
        select(Team).where(Team.external_id == external_id)
    )
    team = result.scalar_one_or_none()

    if team is None:
        team = Team(
            external_id=external_id,
            name=name,
            abbreviation=abbreviation,
            conference=conference,
            division=division,
        )
        session.add(team)
        await session.flush()  # получаем team.id не закрывая транзакцию

    return team


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

async def upsert_match(
    session: AsyncSession,
    external_id: str,
    home_team_id: int,
    away_team_id: int,
    scheduled_at: datetime,
    season: str,
    **kwargs: Any,
) -> Match:
    """
    INSERT ... ON CONFLICT (external_id) DO UPDATE.

    kwargs принимает любые необязательные поля Match:
    status, season_type, home_score_final, away_score_final,
    home_score_regulation, away_score_regulation,
    went_to_overtime, overtime_periods_count,
    total_line, first_half_line, second_half_line,
    arena, attendance, home_is_b2b, away_is_b2b.
    """
    values: dict[str, Any] = {
        "external_id": external_id,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "scheduled_at": scheduled_at,
        "season": season,
        **kwargs,
    }

    # Поля, которые обновляем при конфликте (всё кроме immutable-ключей)
    update_fields = {
        k: v for k, v in values.items()
        if k not in ("external_id", "home_team_id", "away_team_id", "scheduled_at", "season")
    }
    update_fields["updated_at"] = datetime.utcnow()

    stmt = (
        pg_insert(Match)
        .values(**values)
        .on_conflict_do_update(
            constraint="uq_match_external_id",
            set_=update_fields,
        )
        .returning(Match.id)
    )

    result = await session.execute(stmt)
    match_id: int = result.scalar_one()
    await session.flush()

    match_result = await session.execute(select(Match).where(Match.id == match_id))
    return match_result.scalar_one()


# ---------------------------------------------------------------------------
# QuarterStats
# ---------------------------------------------------------------------------

class QuarterStatRow:
    """Typed data-transfer object for a single period row."""

    __slots__ = (
        "period_number",
        "period_type",
        "home_score",
        "away_score",
        "q4_includes_ot_points",
        "home_fga",
        "away_fga",
        "home_fta",
        "away_fta",
        "home_off_reb",
        "away_off_reb",
        "home_turnovers",
        "away_turnovers",
        "home_possessions",
        "away_possessions",
        "home_pace",
        "away_pace",
        "period_duration_seconds",
    )

    def __init__(
        self,
        period_number: int,
        period_type: PeriodType = PeriodType.QUARTER,
        home_score: int | None = None,
        away_score: int | None = None,
        q4_includes_ot_points: bool = False,
        home_fga: int | None = None,
        away_fga: int | None = None,
        home_fta: int | None = None,
        away_fta: int | None = None,
        home_off_reb: int | None = None,
        away_off_reb: int | None = None,
        home_turnovers: int | None = None,
        away_turnovers: int | None = None,
        home_possessions: float | None = None,
        away_possessions: float | None = None,
        home_pace: float | None = None,
        away_pace: float | None = None,
        period_duration_seconds: int | None = None,
    ) -> None:
        self.period_number = period_number
        self.period_type = period_type
        self.home_score = home_score
        self.away_score = away_score
        self.q4_includes_ot_points = q4_includes_ot_points
        self.home_fga = home_fga
        self.away_fga = away_fga
        self.home_fta = home_fta
        self.away_fta = away_fta
        self.home_off_reb = home_off_reb
        self.away_off_reb = away_off_reb
        self.home_turnovers = home_turnovers
        self.away_turnovers = away_turnovers
        self.home_possessions = home_possessions
        self.away_possessions = away_possessions
        self.home_pace = home_pace
        self.away_pace = away_pace
        self.period_duration_seconds = period_duration_seconds


async def save_quarter_stats(
    session: AsyncSession,
    match_id: int,
    rows: list[QuarterStatRow],
) -> list[QuarterStats]:
    """
    Полная замена почетвертной статистики для матча.

    DELETE + bulk INSERT внутри одной транзакции: безопасен при повторных
    запусках парсера (live-обновления, перезапуск после сбоя).
    """
    await session.execute(
        delete(QuarterStats).where(QuarterStats.match_id == match_id)
    )

    objects = [
        QuarterStats(
            match_id=match_id,
            period_number=row.period_number,
            period_type=row.period_type,
            home_score=row.home_score,
            away_score=row.away_score,
            q4_includes_ot_points=row.q4_includes_ot_points,
            home_fga=row.home_fga,
            away_fga=row.away_fga,
            home_fta=row.home_fta,
            away_fta=row.away_fta,
            home_off_reb=row.home_off_reb,
            away_off_reb=row.away_off_reb,
            home_turnovers=row.home_turnovers,
            away_turnovers=row.away_turnovers,
            home_possessions=row.home_possessions,
            away_possessions=row.away_possessions,
            home_pace=row.home_pace,
            away_pace=row.away_pace,
            period_duration_seconds=row.period_duration_seconds,
        )
        for row in rows
    ]

    session.add_all(objects)
    await session.flush()
    return objects
