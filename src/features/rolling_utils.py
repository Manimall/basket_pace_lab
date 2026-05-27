"""
Shared rolling and EMA feature computation for basketball stats.

Used by validate_by_league and build_features to avoid duplication.
shift(1) on every stat column guarantees zero leakage: game N
only sees the team's history up to game N-1.
"""
from __future__ import annotations

import pandas as pd

from src.config import settings

# Stats used in score-based rolling (validate_by_league context).
# h2 = game - h1 (second half) is universal: derivable from scores alone,
# so it works for every league regardless of box-score availability.
SCORE_STAT_COLS: tuple[str, ...] = (
    "pts_scored_q1",
    "pts_allowed_q1",
    "pts_scored_h1",
    "pts_allowed_h1",
    "pts_scored_h2",
    "pts_allowed_h2",
    "pts_scored_game",
    "pts_allowed_game",
)

# Rolling windows and EMA span — driven by config, exported for callers
ROLL_WINDOWS: tuple[int, ...] = settings.features.score_roll_windows
EMA_SPAN: int = settings.features.ema_span
_MIN_PERIODS: int = settings.features.min_periods


def compute_rolling_ema(
    timeline: pd.DataFrame,
    stat_cols: tuple[str, ...] = SCORE_STAT_COLS,
    windows: tuple[int, ...] = ROLL_WINDOWS,
    ema_span: int = EMA_SPAN,
) -> pd.DataFrame:
    """
    Per-team rolling means and EMA with shift(1) to prevent leakage.

    Input:  timeline with [team_id, scheduled_at, match_id, *stat_cols, ...]
    Output: same DataFrame + {col}_L{w} and {col}_EMA{span} columns.
    """
    parts = []
    for _, grp in timeline.groupby("team_id", sort=False):
        grp = grp.sort_values("scheduled_at").copy()
        for col in stat_cols:
            shifted = grp[col].shift(1)
            for w in windows:
                grp[f"{col}_L{w}"] = shifted.rolling(w, min_periods=_MIN_PERIODS).mean()
            grp[f"{col}_EMA{ema_span}"] = shifted.ewm(
                span=ema_span, min_periods=_MIN_PERIODS
            ).mean()
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def merge_rolling_by_side(
    df: pd.DataFrame, rolling: pd.DataFrame, roll_cols: list[str],
) -> pd.DataFrame:
    """Attach per-team rolled columns onto a match frame as home_*/away_* pairs.

    Shared by score-based and advanced-metric pipelines to avoid duplicating
    the home/away merge dance.

    Args:
        df: Match-level frame; must contain ``match_id``, ``home_team_id``,
            ``away_team_id``.
        rolling: Per-appearance rows from ``compute_rolling_ema`` (one row per
            team-match) carrying ``team_id`` and the columns in ``roll_cols``.
        roll_cols: Names of the rolled columns to attach for each side.

    Returns:
        ``df`` extended with ``home_<col>`` and ``away_<col>`` for every
        column in ``roll_cols``.
    """
    keep = ["match_id", "team_id"] + roll_cols
    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        side_roll = (
            rolling
            .merge(
                df[["match_id", team_col]],
                left_on=["match_id", "team_id"],
                right_on=["match_id", team_col],
                how="inner",
            )[keep]
            .rename(columns={c: f"{side}_{c}" for c in roll_cols})
            .drop(columns="team_id")
        )
        df = df.merge(side_roll, on="match_id", how="left")
    return df


def fill_feature_nans(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """
    Fill NaN in rolling/EMA columns with the column's global mean.

    Only a team's very first game produces NaN (no prior history after shift(1)).
    Column mean is a safe fill — these rows are < 1% of the dataset.
    Modifies df in-place; also returns it for chaining.
    """
    for col in feat_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    return df


# Matchup feature column names — single source of truth for both pipelines
MATCHUP_COLS: tuple[str, ...] = (
    "home_attack_strength",
    "away_attack_strength",
    "home_defense_strength",
    "away_defense_strength",
    "win_rate_diff_L5",
    "expected_matchup_pace",
)


def compute_matchup_features(
    df: pd.DataFrame,
    scored_col: str = "pts_scored_game",
    allowed_col: str = "pts_allowed_game",
    league_col: str = "league",
    league_avg_pace: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Add opponent-aware matchup features to a match-level DataFrame.

    Requires columns already present in df (after rolling join, shift-1 applied):
      home/away_{scored_col}_L5    — rolling pts scored (attack proxy)
      home/away_{allowed_col}_L5   — rolling pts allowed (defense proxy)
      home_score_final, away_score_final  — used to compute league avg scoring
      home_win_L5, away_win_L5     — optional; enables win_rate_diff_L5
      home_pace_L5, away_pace_L5   — optional; enables expected_matchup_pace

    Args:
        league_avg_pace: pre-computed per-row league average pace (from raw pace
            column, not rolled). Pass None to skip expected_matchup_pace.
    """
    pts = pd.concat([
        df[[league_col, "home_score_final"]].rename(columns={"home_score_final": "pts"}),
        df[[league_col, "away_score_final"]].rename(columns={"away_score_final": "pts"}),
    ])
    avg_pts = df[league_col].map(pts.groupby(league_col)["pts"].mean())

    df["home_attack_strength"]  = df[f"home_{scored_col}_L5"]  / avg_pts
    df["away_attack_strength"]  = df[f"away_{scored_col}_L5"]  / avg_pts
    df["home_defense_strength"] = df[f"home_{allowed_col}_L5"] / avg_pts
    df["away_defense_strength"] = df[f"away_{allowed_col}_L5"] / avg_pts

    if {"home_win_L5", "away_win_L5"}.issubset(df.columns):
        df["win_rate_diff_L5"] = df["home_win_L5"] - df["away_win_L5"]

    if (
        league_avg_pace is not None
        and {"home_pace_L5", "away_pace_L5"}.issubset(df.columns)
    ):
        df["expected_matchup_pace"] = (
            df["home_pace_L5"] + df["away_pace_L5"] - league_avg_pace
        )

    return df
