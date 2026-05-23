"""
Shared rolling and EMA feature computation for basketball stats.

Used by validate_by_league and build_features to avoid duplication.
shift(1) on every stat column guarantees zero leakage: game N
only sees the team's history up to game N-1.
"""
from __future__ import annotations

import pandas as pd

# Stats used in score-based rolling (validate_by_league context)
SCORE_STAT_COLS: tuple[str, ...] = (
    "pts_scored_q1",
    "pts_allowed_q1",
    "pts_scored_h1",
    "pts_allowed_h1",
    "pts_scored_game",
    "pts_allowed_game",
)

# Rolling windows and EMA span — single source of truth
ROLL_WINDOWS: tuple[int, ...] = (3, 5)
EMA_SPAN: int = 5
# min_periods=1 → NaN only on a team's very first game; all later games have a value
_MIN_PERIODS: int = 1


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
