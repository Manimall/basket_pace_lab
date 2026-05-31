"""Tests for the H2 universal feature and NULL-safe advanced (Four Factors) metrics.

Covers:
  * H2 derivation (game - h1) in score_features._build_team_timeline
  * aggregate_box_score NULL handling (all-NULL group → NaN, not 0)
  * ORtg/DRtg/TOV% correctness and NaN-safety on missing/zero possessions
  * has_boxscore flag
  * leakage protection: advanced rolling uses shift(1) (first appearance NaN)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.features.advanced_metrics import (
    ADVANCED_FEAT_COLS,
    HAS_BOXSCORE_COL,
    add_advanced_metrics,
    aggregate_box_score,
)
from src.features.score_features import ALL_FEAT, _build_team_timeline

# ── H2 universal feature ──────────────────────────────────────────────────────


def _one_match_df() -> pd.DataFrame:
    """Single match: home 12+13 in H1 (=25), final 30 → H2=5; away 10+11=21, final 28 → H2=7."""
    return pd.DataFrame({
        "match_id":         [1],
        "home_team_id":     [10],
        "away_team_id":     [20],
        "scheduled_at":     pd.to_datetime(["2025-10-01"], utc=True),
        "h_q1":             [12.0],
        "h_q2":             [13.0],
        "a_q1":             [10.0],
        "a_q2":             [11.0],
        "home_score_final": [30.0],
        "away_score_final": [28.0],
        "match_home_pace":  [100.0],
        "match_away_pace":  [99.0],
    })


def test_h2_equals_game_minus_h1_home_side():
    tl = _build_team_timeline(_one_match_df(), "home")
    assert tl["pts_scored_h1"].iloc[0]  == 25.0
    assert tl["pts_scored_h2"].iloc[0]  == 5.0    # 30 − 25
    assert tl["pts_allowed_h1"].iloc[0] == 21.0
    assert tl["pts_allowed_h2"].iloc[0] == 7.0    # 28 − 21


def test_h2_away_side_mirrors_orientation():
    tl = _build_team_timeline(_one_match_df(), "away")
    # away team's own scoring = away_final 28, own h1 = 21 → h2 = 7
    assert tl["pts_scored_h1"].iloc[0] == 21.0
    assert tl["pts_scored_h2"].iloc[0] == 7.0
    # away team allows home: home final 30, home h1 = 25 → allowed h2 = 5
    assert tl["pts_allowed_h2"].iloc[0] == 5.0


def test_h2_rolling_features_registered_in_all_feat():
    assert "home_pts_scored_h2_L3" in ALL_FEAT
    assert "away_pts_allowed_h2_EMA5" in ALL_FEAT


# ── aggregate_box_score NULL handling ─────────────────────────────────────────


def _qs(possessions: list[float | None], turnovers: list[float | None]) -> pd.DataFrame:
    return pd.DataFrame({
        "match_id":         [1, 1, 1, 1],
        "period_number":    [1, 2, 3, 4],
        "home_possessions": possessions,
        "away_possessions": possessions,
        "home_turnovers":   turnovers,
        "away_turnovers":   turnovers,
    })


def test_aggregate_box_score_all_null_yields_nan_not_zero():
    agg = aggregate_box_score(_qs([None] * 4, [None] * 4))
    assert math.isnan(agg["home_poss"].iloc[0])
    assert math.isnan(agg["home_tov"].iloc[0])


def test_aggregate_box_score_sums_quarters():
    agg = aggregate_box_score(_qs([25.0, 24.0, 26.0, 25.0], [3.0, 2.0, 4.0, 1.0]))
    assert agg["home_poss"].iloc[0] == 100.0
    assert agg["home_tov"].iloc[0]  == 10.0


def test_aggregate_box_score_missing_columns_is_graceful():
    qs = pd.DataFrame({"match_id": [1, 1], "period_number": [1, 2]})
    agg = aggregate_box_score(qs)
    assert "match_id" in agg.columns
    assert "home_poss" not in agg.columns  # nothing to aggregate


# ── advanced metrics computation + leakage ────────────────────────────────────


def _two_match_df() -> pd.DataFrame:
    """Team 10 plays match 1 then match 2 — lets us check shift(1) rolling."""
    return pd.DataFrame({
        "match_id":         [1, 2],
        "home_team_id":     [10, 10],
        "away_team_id":     [20, 30],
        "scheduled_at":     pd.to_datetime(["2025-10-01", "2025-10-03"], utc=True),
        "home_score_final": [110.0, 120.0],
        "away_score_final": [100.0, 90.0],
    })


def _box(poss: float | None, tov: float | None) -> pd.DataFrame:
    return pd.DataFrame({
        "match_id":  [1, 2],
        "home_poss": [poss, poss],
        "away_poss": [poss, poss],
        "home_tov":  [tov, tov],
        "away_tov":  [tov, tov],
    })


def test_ortg_rolling_correct_and_shifted():
    out = add_advanced_metrics(_two_match_df(), _box(100.0, 10.0))
    # match 1 home ORtg = 110/100*100 = 110; match 2's L3 rolling = prior game = 110
    assert out.loc[out.match_id == 2, "home_ortg_L3"].iloc[0] == pytest.approx(110.0)
    # match 1 is team 10's first appearance → rolling NaN (no leakage from own game)
    assert math.isnan(out.loc[out.match_id == 1, "home_ortg_L3"].iloc[0])


def test_has_boxscore_flag_set_when_present():
    out = add_advanced_metrics(_two_match_df(), _box(100.0, 10.0))
    assert (out[HAS_BOXSCORE_COL] == 1).all()


def test_advanced_metrics_nan_safe_on_missing_possessions():
    out = add_advanced_metrics(_two_match_df(), _box(None, None))
    assert (out[HAS_BOXSCORE_COL] == 0).all()
    assert out["home_ortg_L3"].isna().all()       # no box → NaN, not exception


def test_zero_possessions_treated_as_missing_no_inf():
    out = add_advanced_metrics(_two_match_df(), _box(0.0, 0.0))
    assert (out[HAS_BOXSCORE_COL] == 0).all()
    assert not np.isinf(out["home_ortg_L3"].fillna(0.0)).any()  # no div-by-zero inf


def test_all_advanced_feature_columns_present():
    out = add_advanced_metrics(_two_match_df(), _box(100.0, 10.0))
    for col in ADVANCED_FEAT_COLS:
        assert col in out.columns, f"missing advanced feature column: {col}"
