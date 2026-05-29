"""Box-score advanced metrics sourced from ``team_match_advanced`` (Go scout).

Unlike :mod:`src.features.advanced_metrics` (which derives ORtg/DRtg from
``quarter_stats`` possessions and is therefore all-NaN for score-only leagues
like EuroLeague), this module reads the dedicated ``team_match_advanced`` table
the Go ETL service fills from Sofascore box scores. For EuroLeague the current
season is fully populated, so these become *real* efficiency signals the model
never had before.

Per-team match-level metrics → shift(1) rolling L5 + EMA (no leakage), merged
back as ``home_box_*`` / ``away_box_*`` columns. The ``box_`` prefix keeps them
distinct from advanced_metrics' ``home_ortg_L3`` etc.

Fallback policy (per spec):
  * possessions / true_pace → ``team_match_advanced`` value, else the
    ``quarter_stats``-derived possessions (NBA box score), else league mean.
  * ORtg / DRtg / 3PA / 3PA-rate → ``team_match_advanced`` value, else
    league-mean fill (then global mean for leagues with no data at all), so
    CatBoost never sees NaN in this group.

3PA-rate note: the table has no total FGA, so "3-point share of shots" is
approximated as 3PA per 100 possessions (pace-adjusted long-range reliance).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import settings
from src.features.rolling_utils import compute_rolling_ema, merge_rolling_by_side

log = logging.getLogger(__name__)

_PER_100: float = 100.0

# Per-team box-score stats (already ``box_``-prefixed so rolled columns never
# collide with advanced_metrics' ortg/drtg columns).
TEAM_ADV_STATS: tuple[str, ...] = (
    "box_ortg",
    "box_drtg",
    "box_true_pace",
    "box_tpa",        # raw 3-point attempts
    "box_tpa_rate",   # 3PA per 100 possessions (long-range reliance)
)

# L5 + EMA. The window is the longest configured score window (no hardcoded 5):
# settings.features.score_roll_windows is (3, 5) → [-1] == 5.
_TA_WINDOWS:  tuple[int, ...] = (settings.features.score_roll_windows[-1],)
_TA_EMA_SPAN: int            = settings.features.ema_span
_TA_SUFFIXES: tuple[str, ...] = tuple(f"L{w}" for w in _TA_WINDOWS) + (f"EMA{_TA_EMA_SPAN}",)

_ROLL_COLS: list[str] = [f"{stat}_{sfx}" for stat in TEAM_ADV_STATS for sfx in _TA_SUFFIXES]

TEAM_ADV_FEAT_COLS: list[str] = [
    f"{side}_{col}" for side in ("home", "away") for col in _ROLL_COLS
]


def _side_timeline(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Build one side's per-appearance raw box metrics (pre-rolling).

    Reads the ``{side}_*`` box-score columns already on the match frame (joined
    in ``load_data``). Possessions/pace fall back to the quarter_stats-derived
    ``{side}_poss`` (added by ``add_advanced_metrics``) where the box row is
    missing. 3PA-rate ≈ 3PA per 100 possessions (no FGA available).
    """
    poss = df[f"{side}_possessions"]
    fallback_col = f"{side}_poss"  # quarter_stats possessions (NBA), if present
    if fallback_col in df.columns:
        poss = poss.fillna(df[fallback_col])

    tpa = df[f"{side}_three_pointers_attempted"]
    tpa_rate = tpa / poss.replace(0, np.nan) * _PER_100

    return pd.DataFrame({
        "match_id":      df["match_id"],
        "team_id":       df[f"{side}_team_id"],
        "scheduled_at":  df["scheduled_at"],
        "box_ortg":      df[f"{side}_offensive_rating"],
        "box_drtg":      df[f"{side}_defensive_rating"],
        "box_true_pace": poss,
        "box_tpa":       tpa,
        "box_tpa_rate":  tpa_rate,
    })


def _fill_by_league_mean(df: pd.DataFrame, cols: list[str], league_col: str) -> None:
    """Fill NaN with per-league mean, then global mean for all-NaN leagues."""
    for col in cols:
        if col not in df.columns or not df[col].isna().any():
            continue
        df[col] = df[col].fillna(df.groupby(league_col)[col].transform("mean"))
        if df[col].isna().any():  # league had no data at all
            df[col] = df[col].fillna(df[col].mean())


def add_team_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Attach rolled box-score features from the joined ``team_match_advanced`` cols.

    Args:
        df: Match-level frame with the ``home_*`` / ``away_*`` box-score columns
            from ``load_data`` (offensive_rating, defensive_rating, possessions,
            three_pointers_attempted), plus ``match_id``, ``home_team_id``,
            ``away_team_id``, ``scheduled_at``, ``league``.

    Returns:
        ``df`` extended with the columns in ``TEAM_ADV_FEAT_COLS`` (shift-1
        rolled L5+EMA; no NaN — first-game and missing rows are league-mean
        filled, then global mean for leagues with no box-score at all).
    """
    required = {"home_offensive_rating", "home_possessions", "home_three_pointers_attempted"}
    if not required.issubset(df.columns):
        log.warning("Box-score columns absent — team-advanced features all-NaN before fill.")
        for col in TEAM_ADV_FEAT_COLS:
            df[col] = np.nan
        _fill_by_league_mean(df, TEAM_ADV_FEAT_COLS, "league")
        return df

    timeline = pd.concat(
        [_side_timeline(df, "home"), _side_timeline(df, "away")],
        ignore_index=True,
    )
    rolling = compute_rolling_ema(
        timeline, stat_cols=TEAM_ADV_STATS, windows=_TA_WINDOWS, ema_span=_TA_EMA_SPAN,
    )
    df = merge_rolling_by_side(df, rolling, _ROLL_COLS)
    _fill_by_league_mean(df, TEAM_ADV_FEAT_COLS, "league")

    n_cov = int(df["home_offensive_rating"].notna().sum())
    log.info("Team-advanced features: %d matches carry box-score ORtg/DRtg.", n_cov)
    return df
