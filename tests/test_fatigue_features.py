"""Tests for schedule-fatigue feature engineering in src.features.fatigue.

I/O-free: pure pandas computation, no DB, no CatBoost. Covers the contract
of add_fatigue_features and the supporting helpers, including off-season
reset behaviour.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.fatigue import (
    FATIGUE_FEATURE_COLS,
    add_fatigue_features,
    build_team_appearances,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_matches(rows: list[dict]) -> pd.DataFrame:
    """Build a tiny matches frame for tests with UTC-typed scheduled_at."""
    df = pd.DataFrame(rows)
    df["scheduled_at"] = pd.to_datetime(df["scheduled_at"], utc=True)
    return df


def _row(match_id: int, date: str, home: int, away: int,
         home_rest: int = 7, away_rest: int = 7) -> dict:
    """One match row with sensible defaults for the existing days_rest columns."""
    return {
        "match_id":       match_id,
        "scheduled_at":   date,
        "home_team_id":   home,
        "away_team_id":   away,
        "home_days_rest": home_rest,
        "away_days_rest": away_rest,
    }


# ── build_team_appearances ────────────────────────────────────────────────────


def test_build_team_appearances_stacks_home_and_away():
    matches = _make_matches([_row(1, "2025-10-01", home=1, away=2)])
    apps = build_team_appearances(matches)
    assert len(apps) == 2
    assert set(apps["team_id"]) == {1, 2}
    # is_road: team 1 is home → False; team 2 is away → True
    assert (apps[apps["team_id"] == 1]["is_road"] == False).all()  # noqa: E712
    assert (apps[apps["team_id"] == 2]["is_road"] == True).all()   # noqa: E712


# ── B2B flag ──────────────────────────────────────────────────────────────────


def test_b2b_detected_when_previous_game_yesterday():
    """Two games on consecutive days for team 1 → is_b2b=1 on the 2nd."""
    matches = _make_matches([
        _row(1, "2025-10-01", home=1, away=2),
        _row(2, "2025-10-02", home=1, away=3),
    ])
    out = add_fatigue_features(matches)
    assert out.loc[out["match_id"] == 2, "home_is_b2b"].iloc[0] == 1
    assert out.loc[out["match_id"] == 1, "home_is_b2b"].iloc[0] == 0


def test_b2b_zero_when_gap_is_two_days():
    matches = _make_matches([
        _row(1, "2025-10-01", home=1, away=2),
        _row(2, "2025-10-03", home=1, away=3),
    ])
    out = add_fatigue_features(matches)
    assert out.loc[out["match_id"] == 2, "home_is_b2b"].iloc[0] == 0


# ── density windows ───────────────────────────────────────────────────────────


def test_three_in_four_days_triggers_high_density():
    """Games on Oct 1, 3, 4 → 3rd game has games_last_short=3 and is_high_density=1."""
    matches = _make_matches([
        _row(1, "2025-10-01", home=1, away=2),
        _row(2, "2025-10-03", home=1, away=3),
        _row(3, "2025-10-04", home=1, away=4),
    ])
    out = add_fatigue_features(matches)
    assert out.loc[out["match_id"] == 3, "home_games_last_short"].iloc[0] == 3
    assert out.loc[out["match_id"] == 3, "home_is_high_density"].iloc[0] == 1


def test_high_density_zero_when_games_spread_out():
    """One game per week → games_last_short stays at 1."""
    matches = _make_matches([
        _row(1, "2025-10-01", home=1, away=2),
        _row(2, "2025-10-08", home=1, away=3),
        _row(3, "2025-10-15", home=1, away=4),
    ])
    out = add_fatigue_features(matches)
    assert out.loc[out["match_id"] == 3, "home_games_last_short"].iloc[0] == 1
    assert out.loc[out["match_id"] == 3, "home_is_high_density"].iloc[0] == 0


# ── off-season reset ──────────────────────────────────────────────────────────


def test_off_season_gap_resets_all_indicators():
    """Gap > days_rest_clip_max (21d) → next game looks like season opener."""
    matches = _make_matches([
        _row(1, "2025-05-01", home=1, away=2),   # last game of prev season
        _row(2, "2025-10-01", home=1, away=3),   # new season opener (~153 days later)
    ])
    out = add_fatigue_features(matches)
    assert out.loc[out["match_id"] == 2, "home_is_b2b"].iloc[0] == 0
    assert out.loc[out["match_id"] == 2, "home_games_last_short"].iloc[0] == 1
    assert out.loc[out["match_id"] == 2, "home_is_high_density"].iloc[0] == 0


# ── consecutive road streak ──────────────────────────────────────────────────


def test_consecutive_road_increments_across_away_games_then_resets_on_home():
    """Team 1 plays 3 away, then home → road streak 1, 2, 3, then 0 on home."""
    matches = _make_matches([
        _row(1, "2025-10-01", home=2, away=1),   # team 1 away
        _row(2, "2025-10-03", home=3, away=1),   # team 1 away
        _row(3, "2025-10-05", home=4, away=1),   # team 1 away
        _row(4, "2025-10-07", home=1, away=5),   # team 1 home → reset
    ])
    out = add_fatigue_features(matches)
    assert out.loc[out["match_id"] == 1, "away_consecutive_road"].iloc[0] == 1
    assert out.loc[out["match_id"] == 2, "away_consecutive_road"].iloc[0] == 2
    assert out.loc[out["match_id"] == 3, "away_consecutive_road"].iloc[0] == 3
    # Match 4: away is team 5 (first game ever for team 5) → streak = 1
    assert out.loc[out["match_id"] == 4, "away_consecutive_road"].iloc[0] == 1


def test_consecutive_road_resets_after_off_season_gap():
    matches = _make_matches([
        _row(1, "2025-04-30", home=2, away=1),   # team 1 away, late season
        _row(2, "2025-05-02", home=3, away=1),   # team 1 away, streak=2
        _row(3, "2025-10-01", home=4, away=1),   # new season → reset to 1
    ])
    out = add_fatigue_features(matches)
    assert out.loc[out["match_id"] == 2, "away_consecutive_road"].iloc[0] == 2
    assert out.loc[out["match_id"] == 3, "away_consecutive_road"].iloc[0] == 1


# ── rest_diff ─────────────────────────────────────────────────────────────────


def test_rest_diff_uses_existing_days_rest_columns():
    matches = _make_matches([
        _row(1, "2025-10-01", home=1, away=2, home_rest=7, away_rest=2),
    ])
    out = add_fatigue_features(matches)
    assert out.loc[out["match_id"] == 1, "rest_diff"].iloc[0] == 5


def test_rest_diff_can_be_negative_when_away_more_rested():
    matches = _make_matches([
        _row(1, "2025-10-01", home=1, away=2, home_rest=1, away_rest=4),
    ])
    out = add_fatigue_features(matches)
    assert out.loc[out["match_id"] == 1, "rest_diff"].iloc[0] == -3


# ── contract ──────────────────────────────────────────────────────────────────


def test_all_declared_feature_columns_present_after_call():
    """add_fatigue_features must populate every column listed in the contract."""
    matches = _make_matches([_row(1, "2025-10-01", home=1, away=2)])
    out = add_fatigue_features(matches)
    for col in FATIGUE_FEATURE_COLS:
        assert col in out.columns, f"missing fatigue feature: {col}"


def test_home_consecutive_road_is_not_emitted():
    """The host team's road streak is structurally zero; column must not exist."""
    matches = _make_matches([_row(1, "2025-10-01", home=1, away=2)])
    out = add_fatigue_features(matches)
    assert "home_consecutive_road" not in out.columns
