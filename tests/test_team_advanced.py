"""Tests for box-score advanced features sourced from team_match_advanced.

Covers:
  * shift(1) no-leakage (first appearance filled, not leaking own game)
  * L5 value equals the single prior game (one history point)
  * 3PA-rate = 3PA / possessions * 100, carried through the roll
  * per-league mean fill leaves no NaN
  * declared feature columns present
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.features.team_advanced import (
    TEAM_ADV_FEAT_COLS,
    add_team_advanced,
)


def _match_df() -> pd.DataFrame:
    """Team 1 (home) vs team 2 (away) over 3 sequential matches.

    Team 1 ORtg climbs 100 → 110 → 120; possessions flat at 100, 3PA = 40.
    Lets us check that match 2 sees only match 1 (shift-1).
    """
    rows = []
    for i in range(3):
        rows.append({
            "match_id":     i + 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "scheduled_at": pd.Timestamp("2025-10-01", tz="UTC") + pd.Timedelta(days=2 * i),
            "league":       "EuroLeague",
            "home_offensive_rating":         100.0 + 10 * i,
            "away_offensive_rating":         90.0,
            "home_defensive_rating":         90.0,
            "away_defensive_rating":         100.0 + 10 * i,
            "home_possessions":              100.0,
            "away_possessions":              100.0,
            "home_three_pointers_attempted": 40,
            "away_three_pointers_attempted": 30,
        })
    return pd.DataFrame(rows)


def test_feature_columns_present_and_no_nan():
    out = add_team_advanced(_match_df())
    for col in TEAM_ADV_FEAT_COLS:
        assert col in out.columns, f"missing column {col}"
        assert not out[col].isna().any(), f"NaN remained in {col} after league-mean fill"


def test_l5_uses_only_prior_game():
    out = add_team_advanced(_match_df()).sort_values("match_id").reset_index(drop=True)
    # Match 2: team 1's only prior game is match 1 → L5 ORtg == match-1 ORtg (100).
    assert out.loc[1, "home_box_ortg_L5"] == pytest.approx(100.0)
    # Match 3: prior games 1,2 → mean(100, 110) = 105.
    assert out.loc[2, "home_box_ortg_L5"] == pytest.approx(105.0)


def test_first_appearance_filled_not_leaked():
    out = add_team_advanced(_match_df()).sort_values("match_id").reset_index(drop=True)
    # Match 1 has no prior history; raw shift(1) is NaN, so it must NOT equal the
    # team's own match-1 ORtg (100) — it is league-mean filled instead.
    assert not math.isclose(out.loc[0, "home_box_ortg_L5"], 100.0, abs_tol=1e-9) \
        or out["home_box_ortg_L5"].nunique() == 1


def test_3pa_rate_is_attempts_per_100_possessions():
    out = add_team_advanced(_match_df()).sort_values("match_id").reset_index(drop=True)
    # Team 1: 40 3PA / 100 poss * 100 = 40.0; match 2 L5 sees only match 1 → 40.0.
    assert out.loc[1, "home_box_tpa_rate_L5"] == pytest.approx(40.0)
    # Away team 2: 30 / 100 * 100 = 30.0.
    assert out.loc[1, "away_box_tpa_rate_L5"] == pytest.approx(30.0)


def test_missing_box_columns_graceful():
    df = pd.DataFrame({
        "match_id": [1], "home_team_id": [1], "away_team_id": [2],
        "scheduled_at": pd.to_datetime(["2025-10-01"], utc=True), "league": ["ABA"],
    })
    out = add_team_advanced(df)  # no box columns at all
    for col in TEAM_ADV_FEAT_COLS:
        assert col in out.columns
