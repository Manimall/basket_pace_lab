"""Tests for per-league feature-selection strategy in src.evaluation.feature_selector.

I/O-free: pure dict lookups, no DB, no CatBoost. Covers the contract of
``get_excluded_features`` across the per-league config, default fallback,
and the V4 invariant (BM_COLS always excluded).
"""
from __future__ import annotations

from src.evaluation.feature_selector import (
    DEFAULT_GROUPS,
    FEATURES_BY_LEAGUE,
    FeatureGroup,
    get_excluded_features,
)
from src.features.advanced_metrics import ADVANCED_FEAT_COLS
from src.features.fatigue import FATIGUE_FEATURE_COLS
from src.features.score_features import BM_COLS


# ── always-excluded invariant ─────────────────────────────────────────────────


def test_bm_cols_always_excluded_regardless_of_league():
    """Bookmaker-derived columns must be hidden for every league + fallback."""
    for league in ("NBA", "ABA", "Israel", "EuroLeague", None, "UnknownXYZ"):
        excluded = get_excluded_features(league)
        for col in BM_COLS:
            assert col in excluded, f"BM col {col!r} not excluded for league={league!r}"


# ── NBA: fatigue group enabled ────────────────────────────────────────────────


def test_nba_keeps_fatigue_features():
    """NBA pipeline must see fatigue columns (densely-scheduled, B2B real)."""
    excluded = get_excluded_features("NBA")
    for col in FATIGUE_FEATURE_COLS:
        assert col not in excluded, f"fatigue col {col!r} unexpectedly excluded for NBA"


# ── sparsely-scheduled leagues: fatigue dropped ───────────────────────────────


def test_aba_drops_fatigue_features():
    excluded = get_excluded_features("ABA")
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded, f"fatigue col {col!r} should be dropped for ABA"


def test_israel_drops_fatigue_features():
    excluded = get_excluded_features("Israel")
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded, f"fatigue col {col!r} should be dropped for Israel"


def test_euroleague_drops_fatigue_features():
    excluded = get_excluded_features("EuroLeague")
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded, f"fatigue col {col!r} should be dropped for EuroLeague"


# ── default fallback ──────────────────────────────────────────────────────────


def test_unknown_league_uses_default_fallback():
    """Any league not in FEATURES_BY_LEAGUE gets DEFAULT_GROUPS (BASE only)."""
    excluded = get_excluded_features("CompletelyNewLeague")
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded


def test_none_league_uses_default_fallback():
    """``None`` (mixed-league pipeline) drops fatigue same as unknown league."""
    excluded = get_excluded_features(None)
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded


# ── return type / immutability ────────────────────────────────────────────────


def test_returned_set_is_frozen():
    """Caller must not be able to mutate the cached config indirectly."""
    excluded = get_excluded_features("NBA")
    assert isinstance(excluded, frozenset)


def test_different_calls_can_return_different_sets():
    """NBA's excluded set must differ from Israel's (fatigue toggle)."""
    nba       = get_excluded_features("NBA")
    israel    = get_excluded_features("Israel")
    diff      = israel - nba
    # Israel excludes everything NBA does, plus fatigue cols on top.
    assert diff == frozenset(FATIGUE_FEATURE_COLS)


# ── config sanity ─────────────────────────────────────────────────────────────


def test_default_groups_contains_base_only():
    """DEFAULT_GROUPS must keep BASE (otherwise no features remain)."""
    assert FeatureGroup.BASE in DEFAULT_GROUPS
    assert FeatureGroup.FATIGUE not in DEFAULT_GROUPS


def test_nba_config_contains_base_and_fatigue_only():
    assert FeatureGroup.BASE in FEATURES_BY_LEAGUE["NBA"]
    assert FeatureGroup.FATIGUE in FEATURES_BY_LEAGUE["NBA"]
    # ADVANCED deliberately OFF for NBA (isolation result: redundant noise).
    assert FeatureGroup.ADVANCED not in FEATURES_BY_LEAGUE["NBA"]


# ── advanced group (Four Factors) ─────────────────────────────────────────────


def test_advanced_excluded_for_nba_isolation():
    """NBA must NOT see advanced metrics — they collapse to raw-points noise."""
    excluded = get_excluded_features("NBA")
    for col in ADVANCED_FEAT_COLS:
        assert col in excluded, f"advanced col {col!r} should be dropped for NBA"


def test_advanced_excluded_for_default_fallback():
    for col in ADVANCED_FEAT_COLS:
        assert col in get_excluded_features("EuroLeague")
        assert col in get_excluded_features(None)


def test_nba_still_keeps_h2_base_feature():
    """H2 is a BASE feature (rolling score), must survive for NBA."""
    excluded = get_excluded_features("NBA")
    assert "home_pts_scored_h2_L3" not in excluded
    assert "away_pts_allowed_h2_EMA5" not in excluded
