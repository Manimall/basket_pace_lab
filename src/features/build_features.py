"""
Feature engineering pipeline for basketball totals prediction.

BasketballFeatureBuilder turns raw DB rows (matches + quarter_stats) into
a training-ready DataFrame for CatBoost.

Pipeline:
  1. Load matches + quarter_stats from Postgres
  2. Compute per-match, per-team box-score metrics (pace, rates) [match_stats.py]
  3. Add context features: is_playoff, days_rest            [match_stats.py]
  4. Build per-team chronological timelines
  5. Apply rolling averages L3 / L5 / L10 — shift(1) prevents leakage
  6. Join everything back to match-level and return

Available rolling stats:
  pace, opp_pace, ft_rate, to_rate, oreb_rate,
  pts_per_poss, pts_scored, pts_allowed, win
"""
from __future__ import annotations

import asyncio
import logging
from typing import Sequence

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import settings
from src.data_collection.sofascore_client import FTA_TO_POSS_FACTOR  # noqa: F401 — re-exported for notebooks
from src.database.engine import get_session_factory
from src.features.match_stats import add_context, calc_match_stats
from src.features.rolling_utils import MATCHUP_COLS, compute_matchup_features

log = logging.getLogger(__name__)

_WINDOWS: tuple[int, ...] = settings.features.pace_roll_windows

_ROLL_STATS: tuple[str, ...] = (
    "pace",
    "opp_pace",
    "ft_rate",
    "to_rate",
    "oreb_rate",
    "pts_per_poss",
    "pts_scored",
    "pts_allowed",
    "win",
)

_SQL_MATCHES = """
    SELECT
        m.id            AS match_id,
        m.scheduled_at,
        m.home_team_id,
        m.away_team_id,
        ht.name         AS home_team,
        at.name         AS away_team,
        m.home_score_final,
        m.away_score_final,
        m.went_to_overtime,
        m.season,
        m.has_quarter_breakdown,
        COALESCE(m.tournament_name, 'NBA') AS tournament_name
    FROM matches m
    JOIN teams ht ON ht.id = m.home_team_id
    JOIN teams at ON at.id = m.away_team_id
    WHERE m.home_score_final IS NOT NULL
      AND m.away_score_final IS NOT NULL
    ORDER BY m.scheduled_at
"""

_SQL_QS = """
    SELECT
        match_id, period_number, period_type::text AS period_type,
        home_score,     away_score,
        home_fga,       away_fga,
        home_fta,       away_fta,
        home_off_reb,   away_off_reb,
        home_turnovers, away_turnovers,
        home_possessions, away_possessions,
        home_pace,      away_pace
    FROM quarter_stats
    WHERE period_type::text IN ('QUARTER', 'GAME')
    ORDER BY match_id, period_number
"""


class BasketballFeatureBuilder:
    """
    Builds the feature matrix from the Postgres database.

    Usage:
        builder = BasketballFeatureBuilder()
        df = builder.build()                          # sync
        df = await builder.generate_training_dataset()  # async
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._sf = session_factory or get_session_factory()

    # ── 1. Data loading ───────────────────────────────────────────────

    async def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        async with self._sf() as db:
            matches_rows = (await db.execute(text(_SQL_MATCHES))).mappings().all()
            qs_rows      = (await db.execute(text(_SQL_QS))).mappings().all()
        matches = pd.DataFrame(matches_rows)
        qs      = pd.DataFrame(qs_rows)
        matches["scheduled_at"] = pd.to_datetime(matches["scheduled_at"], utc=True)
        log.info("Loaded %d matches, %d quarter_stats rows.", len(matches), len(qs))
        return matches, qs

    # ── 2. Per-team timelines ─────────────────────────────────────────

    @staticmethod
    def _build_timelines(df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for side, _ in (("home", "away"), ("away", "home")):
            t = pd.DataFrame({
                "match_id":     df["match_id"].values,
                "team_id":      df[f"{side}_team_id"].values,
                "scheduled_at": df["scheduled_at"].values,
                "is_home":      1 if side == "home" else 0,
                **{stat: df[f"{side}_{stat}"].values for stat in _ROLL_STATS},
            })
            parts.append(t)
        return pd.concat(parts, ignore_index=True)

    # ── 3. Rolling averages (no leakage) ──────────────────────────────

    @staticmethod
    def _calc_rolling(
        timeline: pd.DataFrame,
        windows: Sequence[int] = _WINDOWS,
    ) -> pd.DataFrame:
        parts = []
        for _, grp in timeline.groupby("team_id"):
            grp = grp.sort_values("scheduled_at").copy()
            for col in _ROLL_STATS:
                shifted = grp[col].shift(1)
                for w in windows:
                    grp[f"{col}_L{w}"] = shifted.rolling(w, min_periods=1).mean()
            parts.append(grp)
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _join_rolling(df: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
        roll_cols = [c for c in rolling.columns if "_L" in c]
        keep = ["match_id", "team_id"] + roll_cols
        for side in ("home", "away"):
            is_home_val = 1 if side == "home" else 0
            sub = (
                rolling[rolling["is_home"] == is_home_val][keep]
                .rename(columns={c: f"{side}_{c}" for c in roll_cols})
            )
            df = df.merge(
                sub,
                left_on=["match_id", f"{side}_team_id"],
                right_on=["match_id", "team_id"],
                how="left",
            ).drop(columns="team_id")
        return df

    # ── 4. Public API ─────────────────────────────────────────────────

    async def generate_training_dataset(self) -> pd.DataFrame:
        """Full async pipeline. Returns clean DataFrame ready for CatBoost."""
        matches, qs = await self._load_data()
        df = calc_match_stats(matches, qs)
        df = add_context(df)
        timeline = self._build_timelines(df)
        rolling  = self._calc_rolling(timeline)
        df       = self._join_rolling(df, rolling)

        league_avg_pace = df.groupby("tournament_name")["home_pace"].transform("mean")
        df = compute_matchup_features(
            df,
            scored_col="pts_scored",
            allowed_col="pts_allowed",
            league_col="tournament_name",
            league_avg_pace=league_avg_pace,
        )
        return df

    def build(self) -> pd.DataFrame:
        """Synchronous wrapper — call from scripts or notebooks."""
        return asyncio.run(self.generate_training_dataset())

    # ── 5. Column helpers ─────────────────────────────────────────────

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list[str]:
        """All input feature columns (rolling + matchup + context)."""
        roll    = [c for c in df.columns if "_L3" in c or "_L5" in c or "_L10" in c]
        matchup = [c for c in MATCHUP_COLS if c in df.columns]
        ctx     = ["is_playoff", "home_days_rest", "away_days_rest", "has_quarter_breakdown"]
        return roll + matchup + [c for c in ctx if c in df.columns]

    @staticmethod
    def get_target_columns() -> dict[str, list[str]]:
        """Returns target columns grouped by prediction task."""
        return {
            "game": ["game_total", "game_avg_pace"],
            "quarters": [
                f"q{p}_{t}"
                for p in range(1, 5)
                for t in ("total", "avg_pace")
            ],
        }
