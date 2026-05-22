"""
Feature engineering pipeline for basketball totals prediction.

BasketballFeatureBuilder turns raw DB rows (matches + quarter_stats) into
a training-ready DataFrame for CatBoost.

Pipeline:
  1. Load matches + quarter_stats from Postgres
  2. Compute per-match, per-team box-score metrics (pace, rates)
  3. Build per-team chronological timelines
  4. Apply rolling averages L3 / L5 / L10 — shift(1) prevents leakage
  5. Add context features: is_playoff, days_rest
  6. Join everything back to match-level and return

Available rolling stats (limited by what the DB stores):
  pace        — team's own pace that game (Poss * 48/40)
  opp_pace    — opponent's pace that game
  ft_rate     — FTA / possessions  (proxy for getting fouled)
  to_rate     — TO / possessions
  oreb_rate   — OREB / FGA  (2nd-chance offense proxy)
  pts_per_poss — points scored per possession (offensive efficiency)

NOT yet available (schema extension needed): 3PT%, FT%.
"""
from __future__ import annotations

import asyncio
from functools import reduce
from typing import Sequence

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.data_collection.sofascore_client import FTA_TO_POSS_FACTOR  # noqa: F401 — re-exported for notebooks
from src.database.engine import get_session_factory

# Rolling windows (last N games before the match)
_WINDOWS: tuple[int, ...] = (3, 5, 10)

# Stats we'll compute rolling averages for (per team per game)
_ROLL_STATS: tuple[str, ...] = (
    "pace",
    "opp_pace",
    "ft_rate",
    "to_rate",
    "oreb_rate",
    "pts_per_poss",
)

# NBA 25/26 playoff start date (used to derive is_playoff since
# season_type was written as REGULAR for all games in the collector)
_PLAYOFF_START = pd.Timestamp("2026-04-19", tz="UTC")


class BasketballFeatureBuilder:
    """
    Builds the feature matrix from the Postgres database.

    Usage:
        builder = BasketballFeatureBuilder()
        df = builder.build()                   # sync
        df = await builder.generate_training_dataset()  # async
    """

    def __init__(self, session_factory=None) -> None:
        self._sf = session_factory or get_session_factory()

    # ──────────────────────────────────────────────────────────────────
    # 1. Data loading
    # ──────────────────────────────────────────────────────────────────

    async def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        async with self._sf() as db:
            matches_rows = (
                await db.execute(text("""
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
                        m.has_quarter_breakdown
                    FROM matches m
                    JOIN teams ht ON ht.id = m.home_team_id
                    JOIN teams at ON at.id = m.away_team_id
                    WHERE m.home_score_final IS NOT NULL
                      AND m.away_score_final IS NOT NULL
                    ORDER BY m.scheduled_at
                """))
            ).mappings().all()

            qs_rows = (
                await db.execute(text("""
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
                """))
            ).mappings().all()

        matches = pd.DataFrame(matches_rows)
        qs = pd.DataFrame(qs_rows)
        matches["scheduled_at"] = pd.to_datetime(matches["scheduled_at"], utc=True)
        return matches, qs

    # ──────────────────────────────────────────────────────────────────
    # 2. Match-level metrics
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_match_stats(matches: pd.DataFrame, qs: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot quarter_stats to wide format and compute game-level metrics.

        Handles two period types:
          - QUARTER rows: per-quarter pivoting + q-level targets
          - GAME rows: game-level fallback (e.g. EuroLeague); pace used as proxy for q1_avg_pace

        Returns one row per match with:
          - q{1-4}_home_score, q{1-4}_away_score   (quarter scores → targets)
          - q{1-4}_avg_pace, q{1-4}_total           (quarter pace/total → targets)
          - game_total, game_avg_pace                (game-level targets)
          - home/away pace, ft_rate, to_rate, oreb_rate, pts_per_poss (for rolling)
        """
        qs_q = qs[qs["period_type"] == "QUARTER"].copy()
        qs_g = qs[qs["period_type"] == "GAME"].copy()

        # ── per-quarter wide columns (from QUARTER rows) ──────────────
        q_dfs = []
        for p in range(1, 5):
            q = (
                qs_q[qs_q["period_number"] == p][
                    ["match_id", "home_score", "away_score", "home_pace", "away_pace"]
                ]
                .copy()
                .rename(columns={c: f"q{p}_{c}" for c in ["home_score", "away_score", "home_pace", "away_pace"]})
            )
            q_dfs.append(q)
        q_wide = reduce(lambda a, b: a.merge(b, on="match_id", how="outer"), q_dfs)

        for p in range(1, 5):
            q_wide[f"q{p}_total"] = (
                q_wide.get(f"q{p}_home_score", np.nan)
                + q_wide.get(f"q{p}_away_score", np.nan)
            )
            q_wide[f"q{p}_avg_pace"] = (
                q_wide.get(f"q{p}_home_pace", np.nan)
                + q_wide.get(f"q{p}_away_pace", np.nan)
            ) / 2

        # ── game-level aggregates from QUARTER rows ───────────────────
        agg = (
            qs_q.groupby("match_id")
            .agg(
                home_fga=("home_fga", "sum"),
                away_fga=("away_fga", "sum"),
                home_fta=("home_fta", "sum"),
                away_fta=("away_fta", "sum"),
                home_off_reb=("home_off_reb", "sum"),
                away_off_reb=("away_off_reb", "sum"),
                home_to=("home_turnovers", "sum"),
                away_to=("away_turnovers", "sum"),
                home_poss=("home_possessions", "sum"),
                away_poss=("away_possessions", "sum"),
            )
            .reset_index()
        )

        # ── game-level fallback from GAME rows (e.g. EuroLeague) ──────
        if not qs_g.empty:
            agg_fb = (
                qs_g[["match_id", "home_fga", "away_fga", "home_fta", "away_fta",
                       "home_off_reb", "away_off_reb", "home_turnovers", "away_turnovers",
                       "home_possessions", "away_possessions"]]
                .rename(columns={"home_turnovers": "home_to", "away_turnovers": "away_to",
                                 "home_possessions": "home_poss", "away_possessions": "away_poss"})
            )
            agg = agg.merge(agg_fb, on="match_id", how="outer", suffixes=("", "_fb"))
            for col in ["home_fga", "away_fga", "home_fta", "away_fta",
                        "home_off_reb", "away_off_reb", "home_to", "away_to",
                        "home_poss", "away_poss"]:
                fb = f"{col}_fb"
                if fb in agg.columns:
                    agg[col] = agg[col].fillna(agg[fb])
                    agg.drop(columns=[fb], inplace=True)

        df = matches.merge(q_wide, on="match_id", how="left").merge(agg, on="match_id", how="left")

        # ── game targets ──────────────────────────────────────────────
        df["game_total"] = df["home_score_final"] + df["away_score_final"]
        pace_cols = [f"q{p}_avg_pace" for p in range(1, 5)]
        df["game_avg_pace"] = df[pace_cols].mean(axis=1)

        # For GAME-type matches: fill game_avg_pace and q1_avg_pace from game-level pace
        if not qs_g.empty:
            game_pace = (
                qs_g[["match_id", "home_pace", "away_pace"]]
                .rename(columns={"home_pace": "home_pace_fb", "away_pace": "away_pace_fb"})
            )
            game_pace["game_avg_pace_fb"] = (game_pace["home_pace_fb"] + game_pace["away_pace_fb"]) / 2
            df = df.merge(game_pace, on="match_id", how="left")
            df["game_avg_pace"] = df["game_avg_pace"].fillna(df["game_avg_pace_fb"])
            df["q1_avg_pace"] = df["q1_avg_pace"].fillna(df["game_avg_pace"])

        # ── per-team stats for rolling features ───────────────────────
        for side in ("home", "away"):
            poss = df[f"{side}_poss"].replace(0, np.nan)
            fga  = df[f"{side}_fga"].replace(0, np.nan)

            q_avg_pace = df[[f"q{p}_{side}_pace" for p in range(1, 5)]].mean(axis=1)
            if not qs_g.empty and f"{side}_pace_fb" in df.columns:
                df[f"{side}_pace"] = q_avg_pace.fillna(df[f"{side}_pace_fb"])
            else:
                df[f"{side}_pace"] = q_avg_pace

            df[f"{side}_ft_rate"]      = df[f"{side}_fta"]     / poss
            df[f"{side}_to_rate"]      = df[f"{side}_to"]      / poss
            df[f"{side}_oreb_rate"]    = df[f"{side}_off_reb"]  / fga
            df[f"{side}_pts_per_poss"] = df[f"{side}_score_final"] / poss

        # opponent pace (needed as a rolling stat)
        df["home_opp_pace"] = df["away_pace"]
        df["away_opp_pace"] = df["home_pace"]

        # Drop temporary fallback columns
        fb_cols = [c for c in df.columns if c.endswith("_fb")]
        if fb_cols:
            df.drop(columns=fb_cols, inplace=True)

        return df

    # ──────────────────────────────────────────────────────────────────
    # 3. Context features
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _add_context(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("scheduled_at").reset_index(drop=True)

        df["is_playoff"] = (df["scheduled_at"] >= _PLAYOFF_START).astype(int)

        # All appearances per team (home + away combined) → days_rest
        home_apps = df[["match_id", "scheduled_at", "home_team_id"]].rename(
            columns={"home_team_id": "team_id"}
        )
        away_apps = df[["match_id", "scheduled_at", "away_team_id"]].rename(
            columns={"away_team_id": "team_id"}
        )
        all_apps = pd.concat([home_apps, away_apps]).sort_values(
            ["team_id", "scheduled_at"]
        )
        all_apps["prev_date"] = all_apps.groupby("team_id")["scheduled_at"].shift(1)
        all_apps["days_rest"] = (
            all_apps["scheduled_at"] - all_apps["prev_date"]
        ).dt.days

        for side in ("home", "away"):
            team_col = f"{side}_team_id"
            side_rest = (
                all_apps.merge(
                    df[["match_id", team_col]].rename(columns={team_col: "team_id"}),
                    on=["match_id", "team_id"],
                )[["match_id", "days_rest"]]
                .rename(columns={"days_rest": f"{side}_days_rest"})
            )
            df = df.merge(side_rest, on="match_id", how="left")

        return df

    # ──────────────────────────────────────────────────────────────────
    # 4. Rolling features (no leakage)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_timelines(df: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape match-level data into one row per (team, game).
        Each row carries that team's own stats + opponent_pace for that game.
        """
        parts = []
        for side, opp in (("home", "away"), ("away", "home")):
            t = pd.DataFrame({
                "match_id":     df["match_id"].values,
                "team_id":      df[f"{side}_team_id"].values,
                "scheduled_at": df["scheduled_at"].values,
                "is_home":      1 if side == "home" else 0,
                **{stat: df[f"{side}_{stat}"].values for stat in _ROLL_STATS},
            })
            parts.append(t)
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _calc_rolling(
        timeline: pd.DataFrame,
        windows: Sequence[int] = _WINDOWS,
    ) -> pd.DataFrame:
        """
        For each team, sort by date, then compute rolling means.
        shift(1) ensures game N only uses games 1..N-1 (no leakage).
        min_periods=1 handles teams with < window games (season openers).
        """
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
        """Attach home/away rolling columns to the match DataFrame."""
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

    # ──────────────────────────────────────────────────────────────────
    # 5. Public API
    # ──────────────────────────────────────────────────────────────────

    async def generate_training_dataset(self) -> pd.DataFrame:
        """
        Full async pipeline. Returns clean DataFrame ready for CatBoost.

        Feature columns   → get_feature_columns()
        Target columns    → get_target_columns()
        """
        matches, qs = await self._load_data()
        df = self._calc_match_stats(matches, qs)
        df = self._add_context(df)
        timeline = self._build_timelines(df)
        rolling = self._calc_rolling(timeline)
        df = self._join_rolling(df, rolling)
        return df

    def build(self) -> pd.DataFrame:
        """Synchronous wrapper — call from scripts / notebooks."""
        return asyncio.run(self.generate_training_dataset())

    # ──────────────────────────────────────────────────────────────────
    # Helpers for model training
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list[str]:
        """All input feature columns (rolling + context)."""
        roll = [c for c in df.columns if "_L3" in c or "_L5" in c or "_L10" in c]
        ctx  = ["is_playoff", "home_days_rest", "away_days_rest", "has_quarter_breakdown"]
        return roll + [c for c in ctx if c in df.columns]

    @staticmethod
    def get_target_columns() -> dict[str, list[str]]:
        """
        Returns a dict grouping targets by prediction task.

          'game'       → overall game targets
          'quarters'   → per-quarter targets
        """
        return {
            "game": [
                "game_total",
                "game_avg_pace",
            ],
            "quarters": [
                f"q{p}_{t}"
                for p in range(1, 5)
                for t in ("total", "avg_pace")
            ],
        }
