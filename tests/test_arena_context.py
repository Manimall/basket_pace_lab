"""Tests for V8 arena-context features (home fortress / away vulnerability).

Covers:
  * home/away win-rate gap uses only prior, venue-split games (no leakage)
  * gap is NaN until both a home and an away history exist
  * home_fortress_index maps to the home team's gap; away_vulnerability_index
    maps to the away team's gap
  * declared feature columns are present after the call
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.features.arena_context import (
    ARENA_FEAT_COLS,
    AWAY_VULNERABILITY_COL,
    HOME_FORTRESS_COL,
    _team_split_winrate_gap,
    add_arena_context,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


def _arena_df() -> pd.DataFrame:
    """Team 10: wins at home (m1, m3), loses away (m2, m4); m5 is the target.

    Layout (home_team beats away_team in every row by construction):
        m1: 10 vs 20  → team10 home win
        m2: 30 vs 10  → team10 away loss
        m3: 10 vs 21  → team10 home win
        m4: 31 vs 10  → team10 away loss
        m5: 10 vs 22  → team10 home (fortress target)
    """
    rows = [
        (1, 10, 20, 110, 100, "2025-10-01"),
        (2, 30, 10, 110, 100, "2025-10-03"),
        (3, 10, 21, 115,  95, "2025-10-05"),
        (4, 31, 10, 120,  90, "2025-10-07"),
        (5, 10, 22, 100,  99, "2025-10-09"),
    ]
    df = pd.DataFrame(
        rows,
        columns=["match_id", "home_team_id", "away_team_id",
                 "home_score_final", "away_score_final", "scheduled_at"],
    )
    df["scheduled_at"] = pd.to_datetime(df["scheduled_at"], utc=True)
    return df


# ── _team_split_winrate_gap (unit) ────────────────────────────────────────────


def test_split_gap_records_only_prior_games():
    games = pd.DataFrame({
        "is_road": [False, True, False, True, False],   # H, A, H, A, H
        "win":     [1.0,   0.0,  1.0,   0.0,  1.0],
    })
    gaps = _team_split_winrate_gap(games, window=5)
    # g0 (first home): no history at all → NaN
    assert math.isnan(gaps[0])
    # g1 (away): home hist [win]=1.0, away hist empty → NaN (gap undefined)
    assert math.isnan(gaps[1])
    # g2 (home): home [win]=1.0, away [loss]=0.0 → gap 1.0
    assert gaps[2] == pytest.approx(1.0)
    # g4: home [win,win]=1.0, away [loss,loss]=0.0 → gap 1.0
    assert gaps[4] == pytest.approx(1.0)


def test_split_gap_window_caps_history():
    # window 1: only the single most recent same-venue game counts
    games = pd.DataFrame({
        "is_road": [False, True, False, True],
        "win":     [1.0,   1.0,  0.0,   0.0],
    })
    gaps = _team_split_winrate_gap(games, window=1)
    # last row (away): home hist last1 = [loss]=0.0 (from g2), away last1 = [win]=1.0 (g1) → gap -1.0
    assert gaps[3] == pytest.approx(0.0 - 1.0)


# ── add_arena_context (integration) ───────────────────────────────────────────


def test_home_fortress_index_strong_home_team():
    out = add_arena_context(_arena_df())
    # m5: team10 home; prior home wins (m1,m3)=1.0, away losses (m2,m4)=0.0 → +1.0
    assert out.loc[out.match_id == 5, HOME_FORTRESS_COL].iloc[0] == pytest.approx(1.0)


def test_fortress_nan_on_first_appearance():
    out = add_arena_context(_arena_df())
    # m1: team10's first game → no split history → NaN
    assert math.isnan(out.loc[out.match_id == 1, HOME_FORTRESS_COL].iloc[0])


def test_away_vulnerability_maps_to_away_team_gap():
    out = add_arena_context(_arena_df())
    # m4: away team is 10; prior home wins (m1,m3)=1.0, away loss (m2)=0.0 → +1.0
    assert out.loc[out.match_id == 4, AWAY_VULNERABILITY_COL].iloc[0] == pytest.approx(1.0)


def test_away_vulnerability_nan_when_away_team_lacks_split():
    out = add_arena_context(_arena_df())
    # m5: away team is 22, first ever appearance → NaN
    assert math.isnan(out.loc[out.match_id == 5, AWAY_VULNERABILITY_COL].iloc[0])


def test_arena_feature_columns_present():
    out = add_arena_context(_arena_df())
    for col in ARENA_FEAT_COLS:
        assert col in out.columns, f"missing arena column: {col}"
