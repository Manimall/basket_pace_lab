"""Per-league feature-selection strategy (logic layer).

This module is the single source of truth for "which feature groups should the
model see for league X?" — both training and inference layers read here.

The per-league *data* (the ``FeatureGroup`` enum, ``FEATURES_BY_LEAGUE`` map,
thresholds and column groupings) lives in :mod:`src.evaluation.feature_config`
and is re-exported below, so callers keep importing everything from this one
façade. See that module's docstring for the grid-search validation and the
rationale behind each league's config.

This file holds only the resolution logic:

    * :func:`is_league_unprofitable` — betting gate for banned leagues.
    * :func:`get_league_min_threshold` — per-league bet-entry probability.
    * :func:`get_excluded_features` — columns to hide for a given league.
"""
from __future__ import annotations

import logging

from src.evaluation.feature_config import (
    _ALWAYS_EXCLUDED,
    _GROUP_COLUMNS,
    DEFAULT_GROUPS,
    DEFAULT_MIN_THRESHOLD,
    FEATURES_BY_LEAGUE,
    STRICT_THRESHOLD_LEAGUES,
    UNPROFITABLE_LEAGUES,
    FeatureGroup,
)

log = logging.getLogger(__name__)

# Re-export the config data as this module's public surface (single façade).
__all__ = [
    "DEFAULT_GROUPS",
    "DEFAULT_MIN_THRESHOLD",
    "FEATURES_BY_LEAGUE",
    "STRICT_THRESHOLD_LEAGUES",
    "UNPROFITABLE_LEAGUES",
    "FeatureGroup",
    "get_excluded_features",
    "get_league_min_threshold",
    "is_league_unprofitable",
]


# ── Public API ────────────────────────────────────────────────────────────────

def is_league_unprofitable(league_key: str | None) -> bool:
    """Return True when grid-search confirmed the league is unprofitable.

    Callers (inference pipelines, bet-sizers) should gate on this before
    placing a stake. The feature selector still runs normally — unprofitable
    leagues can be trained/evaluated for research, just not bet on.

    Note: ACB is NOT in UNPROFITABLE_LEAGUES. It has a strict threshold
    instead — use ``get_league_min_threshold()`` to enforce it.

    Args:
        league_key: ``matches.tournament_name`` value, or ``None``.

    Returns:
        ``True`` only for leagues in ``UNPROFITABLE_LEAGUES``.
    """
    return league_key in UNPROFITABLE_LEAGUES


def get_league_min_threshold(league_key: str | None) -> float:
    """Return the minimum probability threshold for bet placement.

    Most leagues use ``DEFAULT_MIN_THRESHOLD`` (0.54). Leagues in
    ``STRICT_THRESHOLD_LEAGUES`` (e.g. ACB) require higher confidence to
    overcome inflated bookmaker lines — use the returned threshold instead of
    the global default when filtering bet candidates.

    Args:
        league_key: ``matches.tournament_name`` value, or ``None``.

    Returns:
        Minimum probability threshold in [0.5, 1.0). Always ≥ 0.54.
    """
    if league_key is None:
        return DEFAULT_MIN_THRESHOLD
    threshold = STRICT_THRESHOLD_LEAGUES.get(league_key, DEFAULT_MIN_THRESHOLD)
    if threshold > DEFAULT_MIN_THRESHOLD:
        log.debug(
            "Feature selector: league=%s использует жёсткий порог %.2f "
            "(стандартный %.2f) — рыночный барьер завышенных линий.",
            league_key, threshold, DEFAULT_MIN_THRESHOLD,
        )
    return threshold


def get_excluded_features(league_key: str | None) -> frozenset[str]:
    """Return the column names to hide from the model for one league.

    Combines the V4 invariant (``BM_COLS``, always excluded) with the
    grid-search-validated per-league group rules. For each ``FeatureGroup``
    NOT in the league's allow-list, its columns are added to the excluded set.

    Unprofitable leagues (CBA) fall back to BASE and emit a WARNING so
    monitoring can detect when an inference request targets them.

    Args:
        league_key: ``matches.tournament_name`` (e.g. ``"NBA"``, ``"LegaA"``).
            ``None`` triggers ``DEFAULT_GROUPS`` fallback (mixed-league pipelines).

    Returns:
        Immutable frozenset of column names that ``get_x`` must drop.
    """
    if is_league_unprofitable(league_key):
        log.warning(
            "ПРОПУСК: лига %s помечена как УБЫТОЧНАЯ "
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
