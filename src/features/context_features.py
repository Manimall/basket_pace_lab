"""Context and market-signal features for the score pipeline.

Extracted from ``score_features`` to keep that module focused on score-based
rolling and under the file-size limit. Two responsibilities:

* ``add_rest_and_playoff`` — per-team days of rest + a playoff flag.
* ``add_bookmaker_signals`` — the closing-line columns and the
  market-vs-history delta (line is used only as target/benchmark, never as a
  direct model feature — see the V4 invariant in ``model.py``).
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import settings

log = logging.getLogger(__name__)


def add_rest_and_playoff(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``home_days_rest`` / ``away_days_rest`` and ``is_playoff``.

    Days of rest is the per-team gap since the previous appearance, clipped to
    ``[0, days_rest_clip_max]`` and defaulted (season opener) to
    ``days_rest_default``.

    Args:
        df: Match-level frame with ``match_id``, ``scheduled_at``,
            ``home_team_id``, ``away_team_id``, ``season_type``.

    Returns:
        ``df`` with the three context columns added.
    """
    rest_default  = settings.features.days_rest_default
    rest_clip_max = settings.features.days_rest_clip_max

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
    return df


def add_bookmaker_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add closing-line columns and the market-vs-history delta.

    ``bookmaker_total_closing`` mirrors ``total_line``; ``line_movement`` is the
    close-minus-open delta (NaN when the opening line is unavailable);
    ``market_vs_history_delta`` is the form-based expected total minus the
    market expectation (positive = Over lean).

    Args:
        df: Match-level frame with ``total_line`` / ``total_line_open`` and the
            rolling ``home/away_pts_scored_game_L5`` columns.

    Returns:
        ``df`` with the three signal columns added.
    """
    df["bookmaker_total_closing"] = df["total_line"]
    df["line_movement"]           = df["total_line"] - df["total_line_open"]

    h_scored_l5 = df.get("home_pts_scored_game_L5")
    a_scored_l5 = df.get("away_pts_scored_game_L5")
    if h_scored_l5 is not None and a_scored_l5 is not None:
        df["market_vs_history_delta"] = (h_scored_l5 + a_scored_l5) - df["total_line"]
    else:
        df["market_vs_history_delta"] = float("nan")
    return df
