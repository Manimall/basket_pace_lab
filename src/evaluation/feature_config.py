"""Per-league feature-selection *data* (grid-search validated config).

Pure data extracted from ``feature_selector.py`` to keep that module focused on
logic and under the 250-line limit, and to centralise the "which feature groups
does league X use?" tables in one place (zero hardcode inside the logic layer).

The logic layer (:mod:`src.evaluation.feature_selector`) re-exports every public
name here, so it remains the single import façade for callers.

V9 Grid-Search Sprint (2026-05-31)
-----------------------------------
Per-league configs validated empirically by ``scripts/grid_search_leagues.py``
across 4 feature combos (BASE / BASE+FATIGUE / BASE+TEAM_ADV /
BASE+FATIGUE+TEAM_ADV) on 2 340 box-score-enriched matches.

Winners summary (working thresholds 0.54–0.58):

    League      │ Combo                   │ Best ROI │ AUC   │ Notes
    ────────────┼─────────────────────────┼──────────┼───────┼─────────────────
    LegaA       │ BASE+FATIGUE+TEAM_ADV   │  +25.2%  │ 0.669 │
    EuroLeague  │ BASE+FATIGUE+TEAM_ADV   │   +1.2%  │ 0.515 │
    NBA         │ BASE+FATIGUE            │  +10.8%  │ 0.557 │
    BBL         │ BASE+FATIGUE            │   +5.6%  │ 0.542 │
    LNB         │ BASE+TEAM_ADV           │  +10.0%  │ 0.516 │
    Israel      │ BASE                    │  +17.4%  │ 0.528 │
    BLeague     │ BASE                    │  +15.7%  │ 0.583 │
    NBL         │ BASE                    │  +11.8%  │ 0.650 │
    ABA         │ BASE                    │   +6.2%  │ 0.584 │
    CBA         │ BASE (fallback)         │   -6.2%  │ 0.459 │ UNPROFITABLE
    ACB         │ BASE+FATIGUE+TEAM_ADV   │   -6.8%  │ 0.589 │ STRICT threshold

ACB разбор
----------
AUC=0.589 подтверждает предсказательную силу модели — испанский рынок физически
понят. Проблема в рыночном барьере: букмекеры систематически завышают тоталы ACB,
уничтожая валуй при стандартных порогах (0.54–0.58).

Решение: ACB использует ``BASE+FATIGUE+TEAM_ADV`` (лучший AUC), но активируется
только при уверенности ≥ ``STRICT_THRESHOLD_LEAGUES["ACB"]`` = 0.60. Такой порог
отсекает инфляционные маркет-ситуации и оставляет только железобетонные матчи.

CBA (Китай): AUC=0.459 < 0.5 — модель предсказывает наоборот. Полный бан.

Design choice
-------------
Lookup-based strategy: a ``dict[league_key, frozenset[FeatureGroup]]`` mapping
declares which groups each league uses. Unlisted leagues fall back to
``DEFAULT_GROUPS`` (BASE only) — safe for new or data-sparse leagues. The four
high-pace 2025/26 leagues (BSN/KBL/WNBA/LNB_DR) are intentionally NOT listed:
their per-league grid search produced no cross-season-confirmed edge, so they
ride the safe BASE default until a walk-forward validation says otherwise.
"""
from __future__ import annotations

from enum import StrEnum

from src.features.advanced_metrics import ADVANCED_FEAT_COLS
from src.features.arena_context import ARENA_FEAT_COLS
from src.features.fatigue import FATIGUE_FEATURE_COLS
from src.features.score_features import BM_COLS
from src.features.team_advanced import TEAM_ADV_FEAT_COLS


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

# Leagues where model cannot beat the bookmaker at any confidence level.
# AUC < 0.5 means the model predicts the opposite of reality — full ban.
# Inference layer must call ``is_league_unprofitable()`` before staking.
UNPROFITABLE_LEAGUES: frozenset[str] = frozenset({"CBA"})

# Leagues with genuine predictive power (AUC > 0.5) but inflated bookmaker
# lines that require higher confidence to overcome the market barrier.
# Key = tournament_name, Value = minimum probability threshold for bet entry.
# Use ``get_league_min_threshold()`` instead of reading this dict directly.
STRICT_THRESHOLD_LEAGUES: dict[str, float] = {
    # ACB (Spain): AUC=0.589 — модель понимает физику лиги, но букмекеры
    # систематически завышают тоталы. Порог 0.60 отсекает инфляционные ситуации.
    "ACB": 0.60,
}

# Default minimum probability threshold applied to all unlisted leagues.
DEFAULT_MIN_THRESHOLD: float = 0.54

# Per-league enabled feature groups, validated by grid search 2026-05-31.
# Unlisted leagues fall back to DEFAULT_GROUPS (BASE only).
FEATURES_BY_LEAGUE: dict[str, frozenset[FeatureGroup]] = {
    # ── BASE + FATIGUE + TEAM_ADV ──────────────────────────────────────────
    "LegaA":      frozenset({FeatureGroup.BASE, FeatureGroup.FATIGUE, FeatureGroup.TEAM_ADV}),
    "EuroLeague": frozenset({FeatureGroup.BASE, FeatureGroup.FATIGUE, FeatureGroup.TEAM_ADV}),
    # ACB: AUC=0.589 — используем лучший combo по AUC; порог жёсткий (0.60).
    "ACB":        frozenset({FeatureGroup.BASE, FeatureGroup.FATIGUE, FeatureGroup.TEAM_ADV}),

    # ── BASE + FATIGUE ─────────────────────────────────────────────────────
    "NBA":        frozenset({FeatureGroup.BASE, FeatureGroup.FATIGUE}),
    "BBL":        frozenset({FeatureGroup.BASE, FeatureGroup.FATIGUE}),

    # ── BASE + TEAM_ADV ────────────────────────────────────────────────────
    "LNB":        frozenset({FeatureGroup.BASE, FeatureGroup.TEAM_ADV}),

    # ── BASE only ──────────────────────────────────────────────────────────
    "Israel":     frozenset({FeatureGroup.BASE}),
    "BLeague":    frozenset({FeatureGroup.BASE}),
    "NBL":        frozenset({FeatureGroup.BASE}),
    "ABA":        frozenset({FeatureGroup.BASE}),

    # ── UNPROFITABLE — BASE fallback + WARNING ─────────────────────────────
    # CBA: AUC=0.459 — модель предсказывает наоборот, полный бан.
    "CBA":        frozenset({FeatureGroup.BASE}),
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
