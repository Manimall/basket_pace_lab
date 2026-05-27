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
from src.features.advanced_metrics import (
    ADVANCED_FEAT_COLS,
    add_advanced_metrics,
    aggregate_box_score,
)
from src.features.arena_context import ARENA_FEAT_COLS, add_arena_context
from src.features.context_features import add_bookmaker_signals, add_rest_and_playoff
from src.features.fatigue import FATIGUE_FEATURE_COLS, add_fatigue_features
from src.features.rolling_utils import (
    EMA_SPAN,
    MATCHUP_COLS,
    SCORE_STAT_COLS,
    compute_matchup_features,
    compute_rolling_ema,
    fill_feature_nans,
    merge_rolling_by_side,
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
    ROLL_FEAT_COLS + list(MATCHUP_COLS) + CTX_COLS + FATIGUE_FEATURE_COLS
    + ADVANCED_FEAT_COLS + ARENA_FEAT_COLS + BM_COLS + CAT_COLS
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
           qs.home_score,        qs.away_score,
           qs.home_pace,         qs.away_pace,
           qs.home_possessions,  qs.away_possessions,
           qs.home_turnovers,    qs.away_turnovers
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


def _build_team_timeline(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Build one side's per-appearance timeline of score-based stats.

    Args:
        df: Match-level frame with quarter pivot columns and final scores.
        side: ``"home"`` or ``"away"`` — selects scored/allowed orientation.

    Returns:
        One row per match for the given side, with the score timeline columns
        (q1 / h1 / h2 / game, scored & allowed) plus ``win`` and ``pace``.
        ``h2 = game - h1`` is the universal second-half stat.
    """
    own, opp = ("h", "a") if side == "home" else ("a", "h")
    own_final = df[f"{side}_score_final"]
    opp_final = df["away_score_final" if side == "home" else "home_score_final"]
    own_h1 = df[f"{own}_q1"].fillna(0) + df[f"{own}_q2"].fillna(0)
    opp_h1 = df[f"{opp}_q1"].fillna(0) + df[f"{opp}_q2"].fillna(0)
    return pd.DataFrame({
        "match_id":         df["match_id"],
        "team_id":          df[f"{side}_team_id"],
        "scheduled_at":     df["scheduled_at"],
        "pts_scored_q1":    df[f"{own}_q1"],
        "pts_allowed_q1":   df[f"{opp}_q1"],
        "pts_scored_h1":    own_h1,
        "pts_allowed_h1":   opp_h1,
        "pts_scored_h2":    own_final - own_h1,
        "pts_allowed_h2":   opp_final - opp_h1,
        "pts_scored_game":  own_final,
        "pts_allowed_game": opp_final,
        "win":              (own_final > opp_final).astype(float),
        "pace":             df[f"match_{side}_pace"],
    })


def build_features(matches: pd.DataFrame, qs: pd.DataFrame) -> pd.DataFrame:
    """
    Score-based rolling L3/L5/EMA-5 + matchup + fatigue + advanced features.

    shift-1 on every rolling stat guarantees no leakage. NaN in first-game
    rows of score features filled with column mean; advanced metrics left NaN
    for score-only leagues (no cross-league imputation).
    """
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

    # ── per-team timeline + rolling (shift-1, no leakage) ─────────────
    timeline = pd.concat(
        [_build_team_timeline(df, "home"), _build_team_timeline(df, "away")],
        ignore_index=True,
    )
    rolling  = compute_rolling_ema(timeline, stat_cols=_TIMELINE_STAT_COLS)
    roll_cols = [
        f"{col}_{sfx}"
        for col in _TIMELINE_STAT_COLS
        for sfx in ("L3", "L5", f"EMA{EMA_SPAN}")
    ]
    df = merge_rolling_by_side(df, rolling, roll_cols)

    # ── matchup features ──────────────────────────────────────────────
    league_avg_pace = df.groupby("league")["match_home_pace"].transform("mean")
    df = compute_matchup_features(
        df,
        scored_col="pts_scored_game",
        allowed_col="pts_allowed_game",
        league_col="league",
        league_avg_pace=league_avg_pace,
    )

    # ── NaN fill (score features only; advanced left NaN deliberately) ─
    score_feat_cols = [f"{side}_{c}" for side in ("home", "away") for c in roll_cols]
    fill_feature_nans(df, score_feat_cols + list(MATCHUP_COLS))

    # ── context, fatigue, advanced, arena, bookmaker ──────────────────
    df = add_rest_and_playoff(df)
    df = add_fatigue_features(df)
    df = add_advanced_metrics(df, aggregate_box_score(qs))
    df = add_arena_context(df)
    df = add_bookmaker_signals(df)

    return df
