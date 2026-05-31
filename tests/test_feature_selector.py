"""Tests for per-league feature-selection strategy (V9 grid-search config).

I/O-free: pure dict lookups, no DB, no CatBoost. Covers:
  - V4 invariant: BM_COLS always excluded
  - Per-league group combos from grid-search validation
  - UNPROFITABLE_LEAGUES warning path
  - Default fallback for unknown / None leagues
  - Return-type immutability
"""
from __future__ import annotations

import logging

import pytest

from src.evaluation.feature_selector import (
    DEFAULT_GROUPS,
    FEATURES_BY_LEAGUE,
    UNPROFITABLE_LEAGUES,
    FeatureGroup,
    get_excluded_features,
    is_league_unprofitable,
)
from src.features.advanced_metrics import ADVANCED_FEAT_COLS
from src.features.fatigue import FATIGUE_FEATURE_COLS
from src.features.score_features import BM_COLS
from src.features.team_advanced import TEAM_ADV_FEAT_COLS


# ── V4 invariant: bookmaker columns always hidden ─────────────────────────────

def test_bm_cols_always_excluded_regardless_of_league():
    """Bookmaker-derived columns must be hidden for every league + fallback."""
    for league in ("NBA", "LegaA", "EuroLeague", "CBA", "ACB", "Israel", None, "Unknown"):
        excluded = get_excluded_features(league)
        for col in BM_COLS:
            assert col in excluded, f"BM col {col!r} not excluded for league={league!r}"


# ── BASE+FATIGUE+TEAM_ADV leagues (LegaA, EuroLeague) ────────────────────────

def test_legaa_enables_fatigue_and_team_adv():
    """LegaA: grid search winner +25.2% — fatigue AND team_adv must be visible."""
    excluded = get_excluded_features("LegaA")
    for col in FATIGUE_FEATURE_COLS:
        assert col not in excluded, f"fatigue col {col!r} wrongly excluded for LegaA"
    for col in TEAM_ADV_FEAT_COLS:
        assert col not in excluded, f"team_adv col {col!r} wrongly excluded for LegaA"


def test_euroleague_enables_fatigue_and_team_adv():
    """EuroLeague: same combo as LegaA per V9 grid search."""
    excluded = get_excluded_features("EuroLeague")
    for col in FATIGUE_FEATURE_COLS:
        assert col not in excluded, f"fatigue col {col!r} wrongly excluded for EuroLeague"
    for col in TEAM_ADV_FEAT_COLS:
        assert col not in excluded, f"team_adv col {col!r} wrongly excluded for EuroLeague"


# ── BASE+FATIGUE leagues (NBA, BBL) ──────────────────────────────────────────

def test_nba_keeps_fatigue_excludes_team_adv():
    """NBA: BASE+FATIGUE — fatigue on, TEAM_ADV off (grid search winner)."""
    excluded = get_excluded_features("NBA")
    for col in FATIGUE_FEATURE_COLS:
        assert col not in excluded, f"fatigue col {col!r} wrongly excluded for NBA"
    for col in TEAM_ADV_FEAT_COLS:
        assert col in excluded, f"team_adv col {col!r} should be excluded for NBA"


def test_bbl_keeps_fatigue_excludes_team_adv():
    excluded = get_excluded_features("BBL")
    for col in FATIGUE_FEATURE_COLS:
        assert col not in excluded
    for col in TEAM_ADV_FEAT_COLS:
        assert col in excluded


# ── BASE+TEAM_ADV leagues (LNB) ──────────────────────────────────────────────

def test_lnb_enables_team_adv_excludes_fatigue():
    """LNB: BASE+TEAM_ADV — box-score on, schedule-fatigue off."""
    excluded = get_excluded_features("LNB")
    for col in TEAM_ADV_FEAT_COLS:
        assert col not in excluded, f"team_adv col {col!r} wrongly excluded for LNB"
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded, f"fatigue col {col!r} should be excluded for LNB"


# ── BASE-only leagues ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("league", ["Israel", "BLeague", "NBL", "ABA"])
def test_base_only_leagues_exclude_fatigue_and_team_adv(league: str):
    """Grid search: adding FATIGUE or TEAM_ADV hurt — BASE is the optimum."""
    excluded = get_excluded_features(league)
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded, f"fatigue col {col!r} should be excluded for {league}"
    for col in TEAM_ADV_FEAT_COLS:
        assert col in excluded, f"team_adv col {col!r} should be excluded for {league}"


# ── UNPROFITABLE leagues (CBA, ACB) ──────────────────────────────────────────

@pytest.mark.parametrize("league", ["CBA", "ACB"])
def test_unprofitable_flag_set(league: str):
    assert is_league_unprofitable(league), f"{league} must be in UNPROFITABLE_LEAGUES"


@pytest.mark.parametrize("league", ["NBA", "LegaA", "Israel", "EuroLeague", None, "NewLeague"])
def test_profitable_leagues_not_flagged(league: str):
    assert not is_league_unprofitable(league)


def test_unprofitable_leagues_emit_warning(caplog):
    """CBA and ACB must log a WARNING when feature exclusion is requested."""
    with caplog.at_level(logging.WARNING, logger="src.evaluation.feature_selector"):
        get_excluded_features("CBA")
    assert any("UNPROFITABLE" in r.message for r in caplog.records), \
        "Expected WARNING containing 'UNPROFITABLE' for CBA"


@pytest.mark.parametrize("league", ["CBA", "ACB"])
def test_unprofitable_leagues_still_return_valid_base_exclusions(league: str):
    """Unprofitable leagues fall back to BASE — BM_COLS always excluded."""
    excluded = get_excluded_features(league)
    for col in BM_COLS:
        assert col in excluded
    # BASE fallback → fatigue and TEAM_ADV should be excluded
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded
    for col in TEAM_ADV_FEAT_COLS:
        assert col in excluded


# ── Default fallback ──────────────────────────────────────────────────────────

def test_unknown_league_uses_base_only_fallback():
    excluded = get_excluded_features("CompletelyNewLeague")
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded
    for col in TEAM_ADV_FEAT_COLS:
        assert col in excluded


def test_none_league_uses_base_only_fallback():
    excluded = get_excluded_features(None)
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded


# ── ADVANCED group: always excluded (no league uses it) ─────────────────────

@pytest.mark.parametrize("league", ["NBA", "LegaA", "EuroLeague", "CBA", None])
def test_advanced_always_excluded(league: str | None):
    """ADVANCED (quarter_stats Four-Factors) disabled for all leagues (V6 finding)."""
    excluded = get_excluded_features(league)
    for col in ADVANCED_FEAT_COLS:
        assert col in excluded, f"advanced col {col!r} should always be excluded"


# ── Return type / immutability ────────────────────────────────────────────────

def test_returned_set_is_frozen():
    assert isinstance(get_excluded_features("NBA"), frozenset)
    assert isinstance(get_excluded_features("LegaA"), frozenset)
    assert isinstance(get_excluded_features(None), frozenset)


# ── Config sanity ─────────────────────────────────────────────────────────────

def test_default_groups_is_base_only():
    assert FeatureGroup.BASE in DEFAULT_GROUPS
    assert FeatureGroup.FATIGUE not in DEFAULT_GROUPS
    assert FeatureGroup.TEAM_ADV not in DEFAULT_GROUPS


def test_legaa_has_all_three_groups():
    groups = FEATURES_BY_LEAGUE["LegaA"]
    assert FeatureGroup.BASE in groups
    assert FeatureGroup.FATIGUE in groups
    assert FeatureGroup.TEAM_ADV in groups
    assert FeatureGroup.ADVANCED not in groups


def test_nba_has_fatigue_not_team_adv():
    groups = FEATURES_BY_LEAGUE["NBA"]
    assert FeatureGroup.BASE in groups
    assert FeatureGroup.FATIGUE in groups
    assert FeatureGroup.TEAM_ADV not in groups


def test_unprofitable_leagues_constant_contains_cba_acb():
    assert "CBA" in UNPROFITABLE_LEAGUES
    assert "ACB" in UNPROFITABLE_LEAGUES


def test_nba_base_feature_not_excluded():
    """H2 rolling stat is a BASE feature — must survive for every league."""
    excluded = get_excluded_features("NBA")
    assert "home_pts_scored_h2_L3" not in excluded
    assert "away_pts_allowed_h2_EMA5" not in excluded


# ── Inference layer: run_league_backtest blocks unprofitable leagues ──────────

def test_run_league_backtest_skips_unprofitable_league(caplog):
    """run_league_backtest must return None and log SKIP for CBA/ACB."""
    from unittest.mock import patch
    from src.evaluation.backtest_league import run_league_backtest

    # Patch prepare_dataset so the test never hits the DB
    with patch("src.evaluation.backtest_league.prepare_dataset") as mock_ds, \
         caplog.at_level(logging.WARNING, logger="src.evaluation.backtest_league"):
        result = run_league_backtest("CBA")

    assert result is None, "Unprofitable league must return None, not a BetReport list"
    mock_ds.assert_not_called(), "prepare_dataset must NOT be called for unprofitable leagues"
    assert any("SKIP" in r.message and "UNPROFITABLE" in r.message for r in caplog.records), \
        "Expected WARNING with 'SKIP' and 'UNPROFITABLE' in message"


def test_run_league_backtest_skips_acb(caplog):
    from unittest.mock import patch
    from src.evaluation.backtest_league import run_league_backtest

    with patch("src.evaluation.backtest_league.prepare_dataset") as mock_ds, \
         caplog.at_level(logging.WARNING, logger="src.evaluation.backtest_league"):
        result = run_league_backtest("ACB")

    assert result is None
    mock_ds.assert_not_called()
    assert any("SKIP" in r.message for r in caplog.records)
