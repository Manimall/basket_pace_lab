"""Per-league feature-selection strategy.

This module is the single source of truth for "which feature groups should
the model see for league X?" — both training and inference layers read here.

V9 Grid-Search Sprint (2026-05-31)
-----------------------------------
Per-league configs validated empirically by ``scripts/grid_search_leagues.py``
across 4 feature combos (BASE / BASE+FATIGUE / BASE+TEAM_ADV /
BASE+FATIGUE+TEAM_ADV) on 2 340 box-score-enriched matches.

Winners summary (working thresholds 0.54–0.58):

    League      │ Combo                   │ Best ROI │ AUC
    ────────────┼─────────────────────────┼──────────┼──────
    LegaA       │ BASE+FATIGUE+TEAM_ADV   │  +25.2%  │ 0.669
    EuroLeague  │ BASE+FATIGUE+TEAM_ADV   │   +1.2%  │ 0.515
    NBA         │ BASE+FATIGUE             │  +10.8%  │ 0.557
    BBL         │ BASE+FATIGUE             │   +5.6%  │ 0.542
    LNB         │ BASE+TEAM_ADV           │  +10.0%  │ 0.516
    Israel      │ BASE                    │  +17.4%  │ 0.528
    BLeague     │ BASE                    │  +15.7%  │ 0.583
    NBL         │ BASE                    │  +11.8%  │ 0.650
    ABA         │ BASE                    │   +6.2%  │ 0.584
    CBA         │ BASE (fallback)         │   -6.2%  │ 0.459  ← UNPROFITABLE
    ACB         │ BASE (fallback)         │   -6.8%  │ 0.589  ← UNPROFITABLE

UNPROFITABLE_LEAGUES (CBA, ACB) return BASE exclusions but emit a WARNING so
inference pipelines can gate on ``is_league_unprofitable()`` before staking.

Design choice
-------------
Lookup-based strategy: a ``dict[league_key, frozenset[FeatureGroup]]`` mapping
declares which groups each league uses. Unlisted leagues fall back to
``DEFAULT_GROUPS`` (BASE only) — safe for new or data-sparse leagues.
"""
from __future__ import annotations

import logging
from enum import StrEnum

from src.features.advanced_metrics import ADVANCED_FEAT_COLS
from src.features.arena_context import ARENA_FEAT_COLS
from src.features.fatigue import FATIGUE_FEATURE_COLS
from src.features.score_features import BM_COLS
from src.features.team_advanced import TEAM_ADV_FEAT_COLS

log = logging.getLogger(__name__)


class FeatureGroup(StrEnum):
    """Logical buckets of features for per-league selection.

    Members:
        BASE: Always-on features — rolling score stats (incl. universal H2),
            matchup deltas, context (``home_days_rest`` etc.). Never dropped.
        FATIGUE: Schedule-fatigue columns (10 cols). Densely-scheduled leagues
            benefit (NBA, BBL); sparsely-scheduled ones get noise.
        ADVANCED: Four-Factors metrics derived from quarter_stats possessions
            (ORtg/DRtg/TOV%/true_pace). Disabled everywhere — at homogeneous
            pace they collapse to scaled copies of raw points (V6 finding).
        ARENA: Home-court fortress / away vulnerability indices (V8). Not
            included in V9 grid combos; kept for future experimentation.
        TEAM_ADV: Box-score advanced metrics (ORtg/DRtg/true_pace/3PA/3PA-rate,
            rolled L5+EMA) from ``team_match_advanced`` (Go scout). Real signal
            where box scores were scraped — LegaA +25.2% ROI proof (V9).
    """
    BASE     = "base"
    FATIGUE  = "fatigue"
    ADVANCED = "advanced"
    ARENA    = "arena"
    TEAM_ADV = "team_adv"


# ── Grid-search validated per-league configs (V9) ────────────────────────────

# Leagues confirmed unprofitable: best ROI negative across all 4 combos.
# Inference layer should call ``is_league_unprofitable()`` before staking.
UNPROFITABLE_LEAGUES: frozenset[str] = frozenset({"CBA", "ACB"})

# Per-league enabled feature groups, validated by grid search 2026-05-31.
# Unlisted leagues fall back to DEFAULT_GROUPS (BASE only).
FEATURES_BY_LEAGUE: dict[str, frozenset[FeatureGroup]] = {
    # ── BASE + FATIGUE + TEAM_ADV ──────────────────────────────────────────
    # LegaA: box-score data now fully backfilled → TEAM_ADV delivers +25.2%
    # ROI and jumps AUC from 0.50 to 0.67. Biggest V9 discovery.
    "LegaA":      frozenset({FeatureGroup.BASE, FeatureGroup.FATIGUE, FeatureGroup.TEAM_ADV}),
    # EuroLeague: partially backfilled → modest +1.2% ROI; will improve as
    # scout finishes remaining 327 pending matches.
    "EuroLeague": frozenset({FeatureGroup.BASE, FeatureGroup.FATIGUE, FeatureGroup.TEAM_ADV}),

    # ── BASE + FATIGUE ─────────────────────────────────────────────────────
    # NBA: dense schedule (B2B, 3-in-4), fatigue is real signal (+10.8% ROI).
    "NBA":        frozenset({FeatureGroup.BASE, FeatureGroup.FATIGUE}),
    # BBL: British Basketball League, similarly dense European calendar.
    "BBL":        frozenset({FeatureGroup.BASE, FeatureGroup.FATIGUE}),

    # ── BASE + TEAM_ADV ────────────────────────────────────────────────────
    # LNB (French Pro A): box-score adds signal (+10% vs +5.1% for BASE).
    "LNB":        frozenset({FeatureGroup.BASE, FeatureGroup.TEAM_ADV}),

    # ── BASE only ──────────────────────────────────────────────────────────
    # Adding FATIGUE or TEAM_ADV hurt in grid search; BASE is the optimum.
    "Israel":     frozenset({FeatureGroup.BASE}),
    "BLeague":    frozenset({FeatureGroup.BASE}),
    "NBL":        frozenset({FeatureGroup.BASE}),
    "ABA":        frozenset({FeatureGroup.BASE}),

    # ── UNPROFITABLE — BASE fallback with WARNING ──────────────────────────
    # All combos returned negative ROI. Model cannot beat the line here.
    # CBA: AUC=0.459 (< 0.5) — model predicts opposite of reality.
    # ACB: AUC=0.589 but ROI still -6.8% — line is too efficient.
    "CBA":        frozenset({FeatureGroup.BASE}),
    "ACB":        frozenset({FeatureGroup.BASE}),
}

# Fallback for any unlisted league or mixed-league pipelines (None key).
DEFAULT_GROUPS: frozenset[FeatureGroup] = frozenset({FeatureGroup.BASE})

# Group → column names it controls. BASE is implicit (all columns not listed
# in another group). Only non-base groups need an entry here.
_GROUP_COLUMNS: dict[FeatureGroup, frozenset[str]] = {
    FeatureGroup.FATIGUE:  frozenset(FATIGUE_FEATURE_COLS),
    FeatureGroup.ADVANCED: frozenset(ADVANCED_FEAT_COLS),
    FeatureGroup.ARENA:    frozenset(ARENA_FEAT_COLS),
    FeatureGroup.TEAM_ADV: frozenset(TEAM_ADV_FEAT_COLS),
}

# Columns always hidden from the model (V4 invariant — line-derived leakage).
_ALWAYS_EXCLUDED: frozenset[str] = frozenset(BM_COLS)


# ── Public API ────────────────────────────────────────────────────────────────

def is_league_unprofitable(league_key: str | None) -> bool:
    """Return True when grid-search confirmed the league is unprofitable.

    Callers (inference pipelines, bet-sizers) should gate on this before
    placing a stake. The feature selector still runs normally — unprofitable
    leagues can be trained/evaluated for research, just not bet on.

    Args:
        league_key: ``matches.tournament_name`` value, or ``None``.

    Returns:
        ``True`` only for leagues in ``UNPROFITABLE_LEAGUES``.
    """
    return league_key in UNPROFITABLE_LEAGUES


def get_excluded_features(league_key: str | None) -> frozenset[str]:
    """Return the column names to hide from the model for one league.

    Combines the V4 invariant (``BM_COLS``, always excluded) with the
    grid-search-validated per-league group rules. For each ``FeatureGroup``
    NOT in the league's allow-list, its columns are added to the excluded set.

    Unprofitable leagues (CBA, ACB) fall back to BASE and emit a WARNING so
    monitoring can detect when an inference request targets them.

    Args:
        league_key: ``matches.tournament_name`` (e.g. ``"NBA"``, ``"LegaA"``).
            ``None`` triggers ``DEFAULT_GROUPS`` fallback (mixed-league pipelines).

    Returns:
        Immutable frozenset of column names that ``get_x`` must drop.
    """
    if is_league_unprofitable(league_key):
        log.warning(
            "Селектор фичей: лига %s помечена как УБЫТОЧНАЯ "
            "(Grid Search: все комбинации фичей дали отрицательный ROI). "
            "Применяется BASE fallback — ставки по этой лиге запрещены.",
            league_key,
        )

    enabled = (
        FEATURES_BY_LEAGUE.get(league_key, DEFAULT_GROUPS)
        if league_key is not None
        else DEFAULT_GROUPS
    )
    excluded: frozenset[str] = _ALWAYS_EXCLUDED
    for group, cols in _GROUP_COLUMNS.items():
        if group not in enabled:
            excluded = excluded | cols

    log.debug(
        "Feature selector: league=%s → enabled=%s, excluded=%d cols",
        league_key, sorted(g.value for g in enabled), len(excluded),
    )
    return excluded
