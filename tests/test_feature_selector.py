"""Tests for per-league feature-selection strategy (V9 + ACB strict-threshold).

I/O-free: pure dict lookups, no DB, no CatBoost. Covers:
  - V4 invariant: BM_COLS always excluded
  - Per-league group combos from grid-search validation
  - ACB strict threshold (0.60) — NOT in UNPROFITABLE, uses BASE+FATIGUE+TEAM_ADV
  - UNPROFITABLE_LEAGUES warning path (CBA only now)
  - Default fallback for unknown / None leagues
  - Return-type immutability
"""
from __future__ import annotations

import logging

import pytest

from src.evaluation.feature_selector import (
    DEFAULT_GROUPS,
    DEFAULT_MIN_THRESHOLD,
    FEATURES_BY_LEAGUE,
    STRICT_THRESHOLD_LEAGUES,
    UNPROFITABLE_LEAGUES,
    FeatureGroup,
    get_excluded_features,
    get_league_min_threshold,
    is_league_unprofitable,
)
from src.features.advanced_metrics import ADVANCED_FEAT_COLS
from src.features.fatigue import FATIGUE_FEATURE_COLS
from src.features.score_features import BM_COLS
from src.features.team_advanced import TEAM_ADV_FEAT_COLS

# ── V4 invariant: bookmaker columns always hidden ─────────────────────────────

def test_bm_cols_always_excluded_regardless_of_league() -> None:
    """Bookmaker-derived columns must be hidden for every league + fallback."""
    for league in ("NBA", "LegaA", "EuroLeague", "CBA", "ACB", "Israel", None, "Unknown"):
        excluded = get_excluded_features(league)
        for col in BM_COLS:
            assert col in excluded, f"BM col {col!r} not excluded for league={league!r}"


# ── BASE+FATIGUE+TEAM_ADV leagues ─────────────────────────────────────────────

@pytest.mark.parametrize("league", ["LegaA", "EuroLeague", "ACB"])
def test_base_fatigue_team_adv_leagues(league: str) -> None:
    """LegaA, EuroLeague, ACB: fatigue AND team_adv must be visible."""
    excluded = get_excluded_features(league)
    for col in FATIGUE_FEATURE_COLS:
        assert col not in excluded, f"fatigue col {col!r} wrongly excluded for {league}"
    for col in TEAM_ADV_FEAT_COLS:
        assert col not in excluded, f"team_adv col {col!r} wrongly excluded for {league}"


# ── ACB strict threshold ───────────────────────────────────────────────────────

def test_acb_not_in_unprofitable_leagues() -> None:
    """ACB has AUC=0.589 — it is NOT banned, just uses a strict threshold."""
    assert "ACB" not in UNPROFITABLE_LEAGUES


def test_acb_in_strict_threshold_leagues() -> None:
    assert "ACB" in STRICT_THRESHOLD_LEAGUES


def test_acb_strict_threshold_is_above_default() -> None:
    """ACB threshold must be higher than the default 0.54."""
    assert STRICT_THRESHOLD_LEAGUES["ACB"] > DEFAULT_MIN_THRESHOLD


def test_get_league_min_threshold_acb() -> None:
    threshold = get_league_min_threshold("ACB")
    assert threshold == STRICT_THRESHOLD_LEAGUES["ACB"]
    assert threshold >= 0.60, "ACB threshold должен быть ≥ 0.60"


@pytest.mark.parametrize("league", ["NBA", "LegaA", "Israel", "BLeague", None, "NewLeague"])
def test_get_league_min_threshold_default(league: str | None) -> None:
    """Все лиги без жёсткого порога получают DEFAULT_MIN_THRESHOLD."""
    assert get_league_min_threshold(league) == DEFAULT_MIN_THRESHOLD


def test_acb_features_are_best_auc_combo() -> None:
    """ACB использует BASE+FATIGUE+TEAM_ADV — лучший combo по AUC (0.589)."""
    groups = FEATURES_BY_LEAGUE["ACB"]
    assert FeatureGroup.BASE in groups
    assert FeatureGroup.FATIGUE in groups
    assert FeatureGroup.TEAM_ADV in groups


# ── BASE+FATIGUE leagues (NBA, BBL) ──────────────────────────────────────────

def test_nba_keeps_fatigue_excludes_team_adv() -> None:
    excluded = get_excluded_features("NBA")
    for col in FATIGUE_FEATURE_COLS:
        assert col not in excluded
    for col in TEAM_ADV_FEAT_COLS:
        assert col in excluded


def test_bbl_keeps_fatigue_excludes_team_adv() -> None:
    excluded = get_excluded_features("BBL")
    for col in FATIGUE_FEATURE_COLS:
        assert col not in excluded
    for col in TEAM_ADV_FEAT_COLS:
        assert col in excluded


# ── BASE+TEAM_ADV leagues (LNB) ──────────────────────────────────────────────

def test_lnb_enables_team_adv_excludes_fatigue() -> None:
    excluded = get_excluded_features("LNB")
    for col in TEAM_ADV_FEAT_COLS:
        assert col not in excluded
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded


# ── BASE-only leagues ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("league", ["Israel", "BLeague", "NBL", "ABA"])
def test_base_only_leagues_exclude_fatigue_and_team_adv(league: str) -> None:
    excluded = get_excluded_features(league)
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded
    for col in TEAM_ADV_FEAT_COLS:
        assert col in excluded


# ── UNPROFITABLE leagues (CBA only) ──────────────────────────────────────────

def test_unprofitable_leagues_contains_only_cba() -> None:
    """CBA (AUC<0.5) — единственная полностью забаненная лига. ACB убрана."""
    assert "CBA" in UNPROFITABLE_LEAGUES
    assert "ACB" not in UNPROFITABLE_LEAGUES


def test_cba_is_unprofitable() -> None:
    assert is_league_unprofitable("CBA")


@pytest.mark.parametrize("league", ["ACB", "NBA", "LegaA", "Israel", None, "NewLeague"])
def test_acb_and_others_not_unprofitable(league: str | None) -> None:
    assert not is_league_unprofitable(league)


def test_cba_emits_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="src.evaluation.feature_selector"):
        get_excluded_features("CBA")
    assert any("УБЫТОЧНАЯ" in r.message for r in caplog.records)


def test_cba_returns_valid_base_exclusions() -> None:
    excluded = get_excluded_features("CBA")
    for col in BM_COLS:
        assert col in excluded
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded
    for col in TEAM_ADV_FEAT_COLS:
        assert col in excluded


# ── Default fallback ──────────────────────────────────────────────────────────

def test_unknown_league_uses_base_only_fallback() -> None:
    excluded = get_excluded_features("CompletelyNewLeague")
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded
    for col in TEAM_ADV_FEAT_COLS:
        assert col in excluded


def test_none_league_uses_base_only_fallback() -> None:
    excluded = get_excluded_features(None)
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded


# ── ADVANCED group: always excluded ──────────────────────────────────────────

@pytest.mark.parametrize("league", ["NBA", "LegaA", "ACB", "EuroLeague", "CBA", None])
def test_advanced_always_excluded(league: str | None) -> None:
    excluded = get_excluded_features(league)
    for col in ADVANCED_FEAT_COLS:
        assert col in excluded, f"advanced col {col!r} should always be excluded"


# ── Return type / immutability ────────────────────────────────────────────────

def test_returned_set_is_frozen() -> None:
    assert isinstance(get_excluded_features("NBA"), frozenset)
    assert isinstance(get_excluded_features("ACB"), frozenset)
    assert isinstance(get_excluded_features(None), frozenset)


# ── Config sanity ─────────────────────────────────────────────────────────────

def test_default_groups_is_base_only() -> None:
    assert FeatureGroup.BASE in DEFAULT_GROUPS
    assert FeatureGroup.FATIGUE not in DEFAULT_GROUPS
    assert FeatureGroup.TEAM_ADV not in DEFAULT_GROUPS


def test_nba_base_feature_not_excluded() -> None:
    excluded = get_excluded_features("NBA")
    assert "home_pts_scored_h2_L3" not in excluded
    assert "away_pts_allowed_h2_EMA5" not in excluded


# ── Inference layer: run_league_backtest blocks unprofitable leagues ──────────

def test_run_league_backtest_skips_cba(caplog: pytest.LogCaptureFixture) -> None:
    """run_league_backtest должен вернуть None и залогировать ПРОПУСК для CBA."""
    from unittest.mock import patch

    from src.evaluation.backtest_league import run_league_backtest

    with patch("src.evaluation.backtest_league.prepare_dataset") as mock_ds, \
         caplog.at_level(logging.WARNING, logger="src.evaluation.backtest_league"):
        result = run_league_backtest("CBA")

    assert result is None
    mock_ds.assert_not_called()
    assert any("ПРОПУСК" in r.message and "УБЫТОЧНАЯ" in r.message for r in caplog.records)


def test_run_league_backtest_allows_acb() -> None:
    """ACB НЕ должна блокироваться — у неё жёсткий порог, а не бан."""
    from unittest.mock import MagicMock, patch

    from src.evaluation.backtest_league import run_league_backtest

    # Мокаем всю цепочку кроме проверки блокировки
    with patch("src.evaluation.backtest_league.prepare_dataset", return_value=MagicMock()), \
         patch("src.evaluation.backtest_league.run_pipeline", return_value=[]):
        result = run_league_backtest("ACB")

    # ACB не в UNPROFITABLE_LEAGUES → run_pipeline должен вызваться
    assert result is not None or result == []  # не упало с блокировкой
