"""
Score-based feature engineering for the per-league validation pipeline.

Loads matches + quarter_stats and produces rolling L3/L5/EMA-5 score features
plus matchup features. Used exclusively by validate_by_league.py.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date

import pandas as pd
from sqlalchemy import text

from src.config import settings
from src.database.engine import dispose_engine, get_session_factory
from src.features.fatigue import FATIGUE_FEATURE_COLS, add_fatigue_features
from src.features.rolling_utils import (
    EMA_SPAN,
    MATCHUP_COLS,
    ROLL_WINDOWS,
    SCORE_STAT_COLS,
    compute_matchup_features,
    compute_rolling_ema,
    fill_feature_nans,
)

log = logging.getLogger(__name__)

# Extra per-team stats needed to derive matchup features
_EXTRA_TIMELINE_COLS: tuple[str, ...] = ("win", "pace")
_TIMELINE_STAT_COLS: tuple[str, ...] = SCORE_STAT_COLS + _EXTRA_TIMELINE_COLS

TARGET = "game_total"

# ── Feature column lists ──────────────────────────────────────────────────────

ROLL_FEAT_COLS: list[str] = [
    f"{side}_{stat}_{sfx}"
    for side in ("home", "away")
    for stat in SCORE_STAT_COLS
    for sfx in ("L3", "L5", f"EMA{EMA_SPAN}")
]
CTX_COLS: list[str]  = ["home_days_rest", "away_days_rest", "is_playoff"]
CAT_COLS: list[str]  = ["league"]
# Bookmaker signal features (populated when total_line is available).
# line_movement dropped: Flashscore API doesn't expose opening line,
# so total_line_open is always NULL → feature would be 100% NaN.
BM_COLS: list[str]   = ["bookmaker_total_closing", "market_vs_history_delta"]
ALL_FEAT: list[str]  = (
    ROLL_FEAT_COLS + list(MATCHUP_COLS) + CTX_COLS + FATIGUE_FEATURE_COLS + BM_COLS + CAT_COLS
)

# ── SQL ───────────────────────────────────────────────────────────────────────

# Current-season hard gate. Past seasons are dropped at SQL load time to keep
# the entire pipeline (build_features, prepare_dataset, validate_by_league,
# backtester*) on a single chronological cohort. See FeatureConfig docstring.
_SQL_MATCHES = """
    SELECT
        m.id           AS match_id,
        m.scheduled_at,
        m.home_team_id,
        m.away_team_id,
        m.home_score_final,
        m.away_score_final,
        m.season_type::text AS season_type,
        COALESCE(m.tournament_name, 'NBA') AS league,
        m.total_line,
        m.total_line_open
    FROM matches m
    WHERE m.home_score_final IS NOT NULL
      AND m.away_score_final IS NOT NULL
      AND m.has_quarter_breakdown = TRUE
      AND m.scheduled_at >= :current_season_start
    ORDER BY m.scheduled_at
"""

_SQL_QS = """
    SELECT qs.match_id, qs.period_number,
           qs.home_score, qs.away_score,
           qs.home_pace,  qs.away_pace
    FROM quarter_stats qs
    JOIN matches m ON m.id = qs.match_id
    WHERE qs.period_type::text = 'QUARTER'
      AND qs.period_number IN (1, 2, 3, 4)
      AND m.scheduled_at >= :current_season_start
    ORDER BY qs.match_id, qs.period_number
"""

# ── Data loading ──────────────────────────────────────────────────────────────


async def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load current-season matches + quarter_stats from PostgreSQL.

    Filters both queries by ``settings.features.current_season_start`` so that
    every downstream consumer (feature build, backtester, validate_by_league)
    sees the same single-season cohort. Past seasons are treated as
    information poison and excluded at the source.

    Returns:
        Tuple ``(matches, qs)``:
            * matches: one row per finished match in the current season with
              quarter breakdown available.
            * qs: four rows per match (Q1-Q4) joined on the same season filter.
    """
    sf            = get_session_factory()
    season_start  = date.fromisoformat(settings.features.current_season_start)
    params        = {"current_season_start": season_start}
    async with sf() as db:
        m_rows = (await db.execute(text(_SQL_MATCHES), params)).mappings().all()
        q_rows = (await db.execute(text(_SQL_QS),      params)).mappings().all()
    await dispose_engine()
    matches = pd.DataFrame(m_rows)
    qs      = pd.DataFrame(q_rows)
    matches["scheduled_at"] = pd.to_datetime(matches["scheduled_at"], utc=True)
    log.info(
        "Loaded %d matches, %d QS rows (season ≥ %s).",
        len(matches), len(qs), season_start.isoformat(),
    )
    return matches, qs


# ── Feature engineering ───────────────────────────────────────────────────────


def build_features(matches: pd.DataFrame, qs: pd.DataFrame) -> pd.DataFrame:
    """
    Score-based rolling L3/L5/EMA-5 + matchup features, shift-1 (no leakage).
    NaN in first-game rows filled with column mean.
    """
    rest_default  = settings.features.days_rest_default
    rest_clip_max = settings.features.days_rest_clip_max

    # ── quarter scores pivot ──────────────────────────────────────────
    wide = qs.pivot_table(
        index="match_id",
        columns="period_number",
        values=["home_score", "away_score"],
        aggfunc="first",
    )
    wide.columns = [
        f"{'h' if v == 'home_score' else 'a'}_q{p}" for v, p in wide.columns
    ]
    wide = wide.reset_index()

    pace_agg = (
        qs.groupby("match_id")
        .agg(match_home_pace=("home_pace", "mean"), match_away_pace=("away_pace", "mean"))
        .reset_index()
    )

    df = matches.merge(wide, on="match_id", how="left")
    df = df.merge(pace_agg, on="match_id", how="left")

    df["game_total"] = df["home_score_final"] + df["away_score_final"]
    df["q1_total"]   = df["h_q1"] + df["a_q1"]
    df["h1_total"]   = (
        df["h_q1"].fillna(0) + df["h_q2"].fillna(0) +
        df["a_q1"].fillna(0) + df["a_q2"].fillna(0)
    )

    # ── per-team timeline ─────────────────────────────────────────────
    home_win = (df["home_score_final"] > df["away_score_final"]).astype(float)
    away_win = (df["away_score_final"] > df["home_score_final"]).astype(float)

    home_tl = pd.DataFrame({
        "match_id":         df["match_id"],
        "team_id":          df["home_team_id"],
        "scheduled_at":     df["scheduled_at"],
        "pts_scored_q1":    df["h_q1"],
        "pts_allowed_q1":   df["a_q1"],
        "pts_scored_h1":    df["h_q1"].fillna(0) + df["h_q2"].fillna(0),
        "pts_allowed_h1":   df["a_q1"].fillna(0) + df["a_q2"].fillna(0),
        "pts_scored_game":  df["home_score_final"],
        "pts_allowed_game": df["away_score_final"],
        "win":              home_win,
        "pace":             df["match_home_pace"],
    })
    away_tl = pd.DataFrame({
        "match_id":         df["match_id"],
        "team_id":          df["away_team_id"],
        "scheduled_at":     df["scheduled_at"],
        "pts_scored_q1":    df["a_q1"],
        "pts_allowed_q1":   df["h_q1"],
        "pts_scored_h1":    df["a_q1"].fillna(0) + df["a_q2"].fillna(0),
        "pts_allowed_h1":   df["h_q1"].fillna(0) + df["h_q2"].fillna(0),
        "pts_scored_game":  df["away_score_final"],
        "pts_allowed_game": df["home_score_final"],
        "win":              away_win,
        "pace":             df["match_away_pace"],
    })

    timeline = pd.concat([home_tl, away_tl], ignore_index=True)
    rolling  = compute_rolling_ema(timeline, stat_cols=_TIMELINE_STAT_COLS)

    roll_cols = [
        f"{col}_{sfx}"
        for col in _TIMELINE_STAT_COLS
        for sfx in ("L3", "L5", f"EMA{EMA_SPAN}")
    ]
    keep = ["match_id", "team_id"] + roll_cols

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        side_roll = (
            rolling
            .merge(df[["match_id", team_col]], left_on=["match_id", "team_id"],
                   right_on=["match_id", team_col], how="inner")[keep]
            .rename(columns={c: f"{side}_{c}" for c in roll_cols})
            .drop(columns="team_id")
        )
        df = df.merge(side_roll, on="match_id", how="left")

    # ── matchup features ──────────────────────────────────────────────
    league_avg_pace = df.groupby("league")["match_home_pace"].transform("mean")
    df = compute_matchup_features(
        df,
        scored_col="pts_scored_game",
        allowed_col="pts_allowed_game",
        league_col="league",
        league_avg_pace=league_avg_pace,
    )

    # ── NaN fill ──────────────────────────────────────────────────────
    score_feat_cols = [f"{side}_{c}" for side in ("home", "away") for c in roll_cols]
    fill_feature_nans(df, score_feat_cols + list(MATCHUP_COLS))

    # ── days rest ─────────────────────────────────────────────────────
    all_apps = pd.concat([
        df[["match_id", "scheduled_at", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
        df[["match_id", "scheduled_at", "away_team_id"]].rename(columns={"away_team_id": "team_id"}),
    ]).sort_values(["team_id", "scheduled_at"])
    all_apps["days_rest"] = all_apps.groupby("team_id")["scheduled_at"].diff().dt.days
    all_apps["days_rest"] = all_apps["days_rest"].fillna(rest_default).clip(0, rest_clip_max)

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        rest = (
            all_apps
            .merge(df[["match_id", team_col]].rename(columns={team_col: "team_id"}),
                   on=["match_id", "team_id"])[["match_id", "days_rest"]]
            .rename(columns={"days_rest": f"{side}_days_rest"})
        )
        df = df.merge(rest, on="match_id", how="left")

    df["is_playoff"]     = (df["season_type"] == "playoffs").astype(int)
    df["home_days_rest"] = df["home_days_rest"].fillna(rest_default).clip(0, rest_clip_max)
    df["away_days_rest"] = df["away_days_rest"].fillna(rest_default).clip(0, rest_clip_max)

    # ── Schedule-fatigue features ─────────────────────────────────────
    # B2B flag, density windows (4d/7d), road streak, rest_diff. Off-season
    # gaps reset all indicators — see src.features.fatigue for details.
    df = add_fatigue_features(df)

    # ── Bookmaker signal features ─────────────────────────────────────
    # total_line = closing O/U line (NaN for matches without odds data)
    df["bookmaker_total_closing"] = df["total_line"]

    # line_movement > 0 means market moved the total UP (sharp money on Over)
    # NaN when opening line is unavailable (older data or source didn't provide it)
    df["line_movement"] = df["total_line"] - df["total_line_open"]

    # market_vs_history_delta: form-based expected total minus market expectation.
    # Positive = teams are scoring more than the market expects (Over lean).
    # Negative = teams are scoring less (Under lean).
    h_scored_l5 = df.get("home_pts_scored_game_L5")
    a_scored_l5 = df.get("away_pts_scored_game_L5")
    if h_scored_l5 is not None and a_scored_l5 is not None:
        df["market_vs_history_delta"] = (h_scored_l5 + a_scored_l5) - df["total_line"]
    else:
        df["market_vs_history_delta"] = float("nan")

    return df
