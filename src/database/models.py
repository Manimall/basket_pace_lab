from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class MatchStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class SeasonType(str, enum.Enum):
    REGULAR = "regular"
    PLAYOFFS = "playoffs"
    PRESEASON = "preseason"


# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------

class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    abbreviation: Mapped[str] = mapped_column(String(8), nullable=False)
    conference: Mapped[Optional[str]] = mapped_column(String(32))
    division: Mapped[Optional[str]] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )

    home_matches: Mapped[list["Match"]] = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches: Mapped[list["Match"]] = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("external_id", name="uq_match_external_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)

    home_team_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    away_team_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False, index=True
    )

    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    season: Mapped[str] = mapped_column(String(128), nullable=False)
    tournament_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    tournament_name: Mapped[Optional[str]] = mapped_column(String(128))
    season_type: Mapped[SeasonType] = mapped_column(
        Enum(SeasonType, name="season_type_enum"), nullable=False, default=SeasonType.REGULAR
    )
    status: Mapped[MatchStatus] = mapped_column(
        Enum(MatchStatus, name="match_status_enum"), nullable=False, default=MatchStatus.SCHEDULED
    )

    # --- Final scores (regulation + OT combined, as reported) ---
    home_score_final: Mapped[Optional[int]] = mapped_column(SmallInteger)
    away_score_final: Mapped[Optional[int]] = mapped_column(SmallInteger)

    # --- Regulation-only scores (Q1+Q2+Q3+Q4, excluding all OT periods) ---
    # Critical for total-line ML: use these to avoid OT score contamination.
    home_score_regulation: Mapped[Optional[int]] = mapped_column(SmallInteger)
    away_score_regulation: Mapped[Optional[int]] = mapped_column(SmallInteger)

    # --- OT metadata ---
    went_to_overtime: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Number of OT periods played (0 = no OT)
    overtime_periods_count: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)

    # --- Betting lines (pre-match) ---
    total_line: Mapped[Optional[float]] = mapped_column(Float)        # match total
    first_half_line: Mapped[Optional[float]] = mapped_column(Float)   # H1 total
    second_half_line: Mapped[Optional[float]] = mapped_column(Float)  # H2 total

    # --- Venue / context ---
    arena: Mapped[Optional[str]] = mapped_column(String(128))
    attendance: Mapped[Optional[int]] = mapped_column(Integer)
    # Back-to-back flags affect pace/fatigue significantly
    home_is_b2b: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    away_is_b2b: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), onupdate=datetime.utcnow
    )

    home_team: Mapped["Team"] = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team: Mapped["Team"] = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    quarter_stats: Mapped[list["QuarterStats"]] = relationship(
        "QuarterStats", back_populates="match", cascade="all, delete-orphan", order_by="QuarterStats.period_number"
    )


# ---------------------------------------------------------------------------
# QuarterStats
# ---------------------------------------------------------------------------

class PeriodType(str, enum.Enum):
    QUARTER = "quarter"  # Q1–Q4 (regulation)
    OVERTIME = "overtime"  # OT1, OT2, …


class QuarterStats(Base):
    """
    One row per team per period (Q1-Q4 + OT periods).

    Design rationale for OT isolation:
      - period_type = OVERTIME  →  period is an overtime period; never mix
        these rows into Q4 totals when building ML features.
      - For Q4 specifically: home_score / away_score contain ONLY regulation
        Q4 points. If raw provider data lumps Q4+OT together, strip OT points
        out before inserting (use the separate OT rows for that delta).
      - q4_includes_ot_points flag is a data-quality sentinel: True means the
        source mixed Q4 and OT, and the row still needs a manual/automated
        correction pass. ML pipelines must filter rows where this is True.
    """

    __tablename__ = "quarter_stats"
    __table_args__ = (
        UniqueConstraint("match_id", "period_number", "period_type", name="uq_period_per_match"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    match_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("matches.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # 1-4 for regulation quarters; 1, 2, … for OT periods
    period_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    period_type: Mapped[PeriodType] = mapped_column(
        Enum(PeriodType, name="period_type_enum"), nullable=False, default=PeriodType.QUARTER
    )

    home_score: Mapped[Optional[int]] = mapped_column(SmallInteger)
    away_score: Mapped[Optional[int]] = mapped_column(SmallInteger)

    # Data-quality sentinel: True if provider merged Q4+OT into this row.
    # ML pipeline must exclude rows where q4_includes_ot_points = True.
    q4_includes_ot_points: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # --- Pace / box-score metrics (optional, populated from box-score feeds) ---
    # Shooting
    home_fga: Mapped[Optional[int]] = mapped_column(SmallInteger)
    away_fga: Mapped[Optional[int]] = mapped_column(SmallInteger)
    home_fta: Mapped[Optional[int]] = mapped_column(SmallInteger)
    away_fta: Mapped[Optional[int]] = mapped_column(SmallInteger)
    # Rebounds
    home_off_reb: Mapped[Optional[int]] = mapped_column(SmallInteger)
    away_off_reb: Mapped[Optional[int]] = mapped_column(SmallInteger)
    # Turnovers
    home_turnovers: Mapped[Optional[int]] = mapped_column(SmallInteger)
    away_turnovers: Mapped[Optional[int]] = mapped_column(SmallInteger)
    # Derived pace metrics (Poss = FGA - OffReb + TO + 0.44*FTA)
    home_possessions: Mapped[Optional[float]] = mapped_column(Float)
    away_possessions: Mapped[Optional[float]] = mapped_column(Float)
    home_pace: Mapped[Optional[float]] = mapped_column(Float)
    away_pace: Mapped[Optional[float]] = mapped_column(Float)
    # Duration in seconds (useful for OT periods that may be shortened)
    period_duration_seconds: Mapped[Optional[int]] = mapped_column(SmallInteger)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )

    match: Mapped["Match"] = relationship("Match", back_populates="quarter_stats")
