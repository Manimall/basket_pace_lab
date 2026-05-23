"""
Match-level and context feature engineering for basketball.

Functions extracted from BasketballFeatureBuilder to keep build_features.py
under 250 lines while preserving identical behavior.
"""
from __future__ import annotations

import logging
from functools import reduce

import numpy as np
import pandas as pd

from src.config import settings

log = logging.getLogger(__name__)

_PLAYOFF_START = pd.Timestamp(settings.features.nba_playoff_start, tz="UTC")


def calc_match_stats(matches: pd.DataFrame, qs: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot quarter_stats to wide format and compute game-level metrics.

    Handles two period types:
      - QUARTER rows: per-quarter pivoting + q-level targets
      - GAME rows: game-level fallback (e.g. EuroLeague); pace used as proxy for q1_avg_pace

    Returns one row per match with quarter scores/pace, game total/pace,
    and per-team rolling-stat inputs (pace, ft_rate, to_rate, oreb_rate,
    pts_per_poss, pts_scored, pts_allowed, win).
    """
    qs_q = qs[qs["period_type"] == "QUARTER"].copy()
    qs_g = qs[qs["period_type"] == "GAME"].copy()

    # ── per-quarter wide columns ──────────────────────────────────────
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

    # ── game-level aggregates from QUARTER rows ───────────────────────
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

    # ── game-level fallback from GAME rows (e.g. EuroLeague) ─────────
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

    # ── game targets ──────────────────────────────────────────────────
    df["game_total"] = df["home_score_final"] + df["away_score_final"]
    pace_cols = [f"q{p}_avg_pace" for p in range(1, 5)]
    df["game_avg_pace"] = df[pace_cols].mean(axis=1)

    if not qs_g.empty:
        game_pace = (
            qs_g[["match_id", "home_pace", "away_pace"]]
            .rename(columns={"home_pace": "home_pace_fb", "away_pace": "away_pace_fb"})
        )
        game_pace["game_avg_pace_fb"] = (game_pace["home_pace_fb"] + game_pace["away_pace_fb"]) / 2
        df = df.merge(game_pace, on="match_id", how="left")
        df["game_avg_pace"] = df["game_avg_pace"].fillna(df["game_avg_pace_fb"])
        df["q1_avg_pace"]   = df["q1_avg_pace"].fillna(df["game_avg_pace"])

    # ── per-team stats for rolling ────────────────────────────────────
    for side in ("home", "away"):
        poss = df[f"{side}_poss"].replace(0, np.nan)
        fga  = df[f"{side}_fga"].replace(0, np.nan)

        q_avg_pace = df[[f"q{p}_{side}_pace" for p in range(1, 5)]].mean(axis=1)
        if not qs_g.empty and f"{side}_pace_fb" in df.columns:
            df[f"{side}_pace"] = q_avg_pace.fillna(df[f"{side}_pace_fb"])
        else:
            df[f"{side}_pace"] = q_avg_pace

        df[f"{side}_ft_rate"]      = df[f"{side}_fta"]        / poss
        df[f"{side}_to_rate"]      = df[f"{side}_to"]         / poss
        df[f"{side}_oreb_rate"]    = df[f"{side}_off_reb"]    / fga
        df[f"{side}_pts_per_poss"] = df[f"{side}_score_final"] / poss

    df["home_opp_pace"] = df["away_pace"]
    df["away_opp_pace"] = df["home_pace"]

    df["home_pts_scored"]  = df["home_score_final"]
    df["away_pts_scored"]  = df["away_score_final"]
    df["home_pts_allowed"] = df["away_score_final"]
    df["away_pts_allowed"] = df["home_score_final"]
    df["home_win"] = (df["home_score_final"] > df["away_score_final"]).astype(float)
    df["away_win"] = (df["away_score_final"] > df["home_score_final"]).astype(float)

    fb_cols = [c for c in df.columns if c.endswith("_fb")]
    if fb_cols:
        df.drop(columns=fb_cols, inplace=True)

    return df


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_playoff flag and per-team days_rest columns."""
    df = df.copy().sort_values("scheduled_at").reset_index(drop=True)

    df["is_playoff"] = (df["scheduled_at"] >= _PLAYOFF_START).astype(int)

    home_apps = df[["match_id", "scheduled_at", "home_team_id"]].rename(
        columns={"home_team_id": "team_id"}
    )
    away_apps = df[["match_id", "scheduled_at", "away_team_id"]].rename(
        columns={"away_team_id": "team_id"}
    )
    all_apps = pd.concat([home_apps, away_apps]).sort_values(["team_id", "scheduled_at"])
    all_apps["prev_date"] = all_apps.groupby("team_id")["scheduled_at"].shift(1)
    all_apps["days_rest"] = (all_apps["scheduled_at"] - all_apps["prev_date"]).dt.days

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
