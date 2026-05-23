"""
Tests for rolling feature computation and matchup features.

Covers:
  - shift(1) no-leakage guarantee
  - L3 rolling mean correctness
  - EMA-5 correctness
  - NaN fill behavior (fill_feature_nans)
  - compute_matchup_features (attack/defense strength, win_rate_diff_L5)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.rolling_utils import (
    EMA_SPAN,
    compute_matchup_features,
    compute_rolling_ema,
    fill_feature_nans,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_timeline(n: int = 8) -> pd.DataFrame:
    """Single team, n sequential games with known pts_scored_game values."""
    return pd.DataFrame({
        "team_id":          [1] * n,
        "match_id":         list(range(n)),
        "scheduled_at":     pd.date_range("2025-01-01", periods=n, freq="2D", tz="UTC"),
        "pts_scored_game":  [100.0 + i * 5 for i in range(n)],
        "pts_allowed_game": [95.0  - i * 2 for i in range(n)],
        "pts_scored_q1":    [25.0] * n,
        "pts_allowed_q1":   [22.0] * n,
        "pts_scored_h1":    [50.0] * n,
        "pts_allowed_h1":   [44.0] * n,
    })


# ── shift-1 no-leakage ────────────────────────────────────────────────────────


def test_no_leakage_first_game_is_nan():
    """Game 0 has no prior history — all rolling values must be NaN."""
    tl = _make_timeline(5)
    out = compute_rolling_ema(tl)
    first = out.sort_values("scheduled_at").iloc[0]
    assert np.isnan(first["pts_scored_game_L3"]), "First game L3 must be NaN"
    assert np.isnan(first[f"pts_scored_game_EMA{EMA_SPAN}"]), "First game EMA must be NaN"


def test_no_leakage_value_uses_only_prior_games():
    """Game 1 L3 should equal game 0's pts_scored_game (only 1 prior game)."""
    tl = _make_timeline(5)
    out = compute_rolling_ema(tl).sort_values("scheduled_at").reset_index(drop=True)
    # Game 1 (index 1) shifted value is game 0 → L3 with min_periods=1 = game0 score
    expected = tl["pts_scored_game"].iloc[0]
    actual   = out.loc[1, "pts_scored_game_L3"]
    assert abs(actual - expected) < 1e-9, f"Expected {expected}, got {actual}"


# ── L3 rolling mean ───────────────────────────────────────────────────────────


def test_l3_rolling_mean_correctness():
    """
    Game 3 L3 = mean(games 0,1,2) = mean(100, 105, 110) = 105.
    shift(1) means game 3 sees games 0-2.
    """
    tl  = _make_timeline(6)
    out = compute_rolling_ema(tl).sort_values("scheduled_at").reset_index(drop=True)
    # game 3 → shifted window is [game0, game1, game2] = [100, 105, 110]
    expected = (100.0 + 105.0 + 110.0) / 3
    actual   = out.loc[3, "pts_scored_game_L3"]
    assert abs(actual - expected) < 1e-9, f"Expected {expected}, got {actual}"


def test_l5_rolling_mean_correctness():
    """Game 5 L5 = mean(games 0..4) = mean(100,105,110,115,120) = 110."""
    tl  = _make_timeline(7)
    out = compute_rolling_ema(tl).sort_values("scheduled_at").reset_index(drop=True)
    expected = (100.0 + 105.0 + 110.0 + 115.0 + 120.0) / 5
    actual   = out.loc[5, "pts_scored_game_L5"]
    assert abs(actual - expected) < 1e-9, f"Expected {expected}, got {actual}"


# ── EMA ───────────────────────────────────────────────────────────────────────


def test_ema_span_matches_pandas():
    """compute_rolling_ema EMA output must match pd.Series.ewm(span=EMA_SPAN)."""
    tl  = _make_timeline(8)
    out = compute_rolling_ema(tl).sort_values("scheduled_at").reset_index(drop=True)

    pts = pd.Series([100.0 + i * 5 for i in range(8)])
    expected_ema = pts.shift(1).ewm(span=EMA_SPAN, min_periods=1).mean()

    for i in range(1, 8):
        actual   = out.loc[i, f"pts_scored_game_EMA{EMA_SPAN}"]
        expected = expected_ema.iloc[i]
        assert abs(actual - expected) < 1e-6, f"Row {i}: EMA mismatch ({actual} vs {expected})"


# ── fill_feature_nans ─────────────────────────────────────────────────────────


def test_fill_feature_nans_fills_with_column_mean():
    df = pd.DataFrame({"x": [np.nan, 10.0, 20.0, 30.0]})
    fill_feature_nans(df, ["x"])
    col_mean = (10.0 + 20.0 + 30.0) / 3
    assert abs(df.loc[0, "x"] - col_mean) < 1e-9


def test_fill_feature_nans_leaves_non_nan_unchanged():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    fill_feature_nans(df, ["x"])
    assert list(df["x"]) == [1.0, 2.0, 3.0]


def test_fill_feature_nans_skips_missing_columns():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    fill_feature_nans(df, ["x", "does_not_exist"])  # must not raise


# ── compute_matchup_features ──────────────────────────────────────────────────


def _make_match_df() -> pd.DataFrame:
    """Minimal match-level DataFrame with precomputed rolling _L5 columns."""
    avg_pts = 110.0
    return pd.DataFrame({
        "league":                  ["NBA", "NBA"],
        "home_score_final":        [110.0, 105.0],
        "away_score_final":        [100.0, 115.0],
        "home_pts_scored_game_L5": [115.0, 108.0],
        "away_pts_scored_game_L5": [102.0, 112.0],
        "home_pts_allowed_game_L5":[100.0, 106.0],
        "away_pts_allowed_game_L5":[108.0, 111.0],
        "home_win_L5":             [0.6,   0.4],
        "away_win_L5":             [0.4,   0.6],
        "home_pace_L5":            [98.0,  96.0],
        "away_pace_L5":            [97.0,  99.0],
    })


def test_attack_strength_ratio():
    """home_attack_strength = home_pts_scored_L5 / league_avg_pts."""
    df = _make_match_df()
    pts = pd.concat([
        df[["league", "home_score_final"]].rename(columns={"home_score_final": "pts"}),
        df[["league", "away_score_final"]].rename(columns={"away_score_final": "pts"}),
    ])
    avg_pts = pts.groupby("league")["pts"].mean().iloc[0]

    df = compute_matchup_features(
        df,
        scored_col="pts_scored_game",
        allowed_col="pts_allowed_game",
        league_col="league",
    )
    expected = df["home_pts_scored_game_L5"].iloc[0] / avg_pts
    assert abs(df["home_attack_strength"].iloc[0] - expected) < 1e-9


def test_win_rate_diff_computed():
    df = _make_match_df()
    df = compute_matchup_features(df, scored_col="pts_scored_game",
                                  allowed_col="pts_allowed_game", league_col="league")
    assert "win_rate_diff_L5" in df.columns
    assert abs(df["win_rate_diff_L5"].iloc[0] - (0.6 - 0.4)) < 1e-9


def test_expected_matchup_pace():
    df = _make_match_df()
    league_avg_pace = pd.Series([97.5, 97.5])  # precomputed avg
    df = compute_matchup_features(
        df, scored_col="pts_scored_game", allowed_col="pts_allowed_game",
        league_col="league", league_avg_pace=league_avg_pace,
    )
    assert "expected_matchup_pace" in df.columns
    expected = 98.0 + 97.0 - 97.5
    assert abs(df["expected_matchup_pace"].iloc[0] - expected) < 1e-9
