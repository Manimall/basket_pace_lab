"""Per-league feature-selection strategy.

Different leagues have different schedule density patterns. Fatigue features
(B2B, density windows, road streak, rest_diff) provide strong signal in
densely-scheduled NBA but degenerate to noise in sparsely-scheduled European
leagues (documented in ``docs/postmortem_v1_v6.md`` V7 section).

This module owns the single source of truth that answers the question
"which columns should the model see for league X?" — both training and
inference layers read from here.

Design choice
-------------
Lookup-based strategy: a ``dict[league_key, frozenset[FeatureGroup]]``
mapping declares which feature groups each league uses. Unlisted leagues
fall back to ``DEFAULT_GROUPS`` (BASE only). This keeps the rule extensible
(new groups go in the enum + ``_GROUP_COLUMNS``) without touching call-sites.
"""
from __future__ import annotations

import logging
from enum import StrEnum

from src.features.fatigue import FATIGUE_FEATURE_COLS
from src.features.score_features import BM_COLS

log = logging.getLogger(__name__)


class FeatureGroup(StrEnum):
    """Logical buckets of features for per-league selection.

    Members:
        BASE: Always-on features — rolling stats, matchup deltas, context
            (``home_days_rest`` etc.). These never get dropped.
        FATIGUE: V7 schedule-fatigue columns (10 cols). Densely-scheduled
            leagues benefit; sparsely-scheduled ones get noise.
    """
    BASE    = "base"
    FATIGUE = "fatigue"


# Per-league enabled groups. Unlisted leagues fall back to DEFAULT_GROUPS.
# Documented empirically in postmortem V7 / V7.1.
FEATURES_BY_LEAGUE: dict[str, frozenset[FeatureGroup]] = {
    "NBA": frozenset({FeatureGroup.BASE, FeatureGroup.FATIGUE}),
}

# Fallback for any unlisted league or for mixed-league pipelines where no
# single league key applies (e.g. the OTHER control group).
DEFAULT_GROUPS: frozenset[FeatureGroup] = frozenset({FeatureGroup.BASE})

# Group → the column names it controls. BASE is implicit (everything not
# listed in another group). Only non-base groups need an entry here.
_GROUP_COLUMNS: dict[FeatureGroup, frozenset[str]] = {
    FeatureGroup.FATIGUE: frozenset(FATIGUE_FEATURE_COLS),
}

# Columns always hidden from the model (V4 invariant — line-derived).
_ALWAYS_EXCLUDED: frozenset[str] = frozenset(BM_COLS)


def get_excluded_features(league_key: str | None) -> frozenset[str]:
    """Return the set of column names to hide from the model for one league.

    Combines the V4 invariant (``BM_COLS``, always excluded) with the
    league-specific feature-group rules. For each ``FeatureGroup`` NOT in
    the league's allow-list, its columns are added to the excluded set.

    Args:
        league_key: Value of ``matches.tournament_name`` (e.g. ``"NBA"``,
            ``"ABA"``, ``"Israel"``). ``None`` triggers ``DEFAULT_GROUPS``
            fallback (used for mixed-league pipelines).

    Returns:
        Immutable frozenset of column names that ``get_x`` must drop.
    """
    enabled = (
        FEATURES_BY_LEAGUE.get(league_key, DEFAULT_GROUPS)
        if league_key is not None
        else DEFAULT_GROUPS
    )
    excluded = _ALWAYS_EXCLUDED
    for group, cols in _GROUP_COLUMNS.items():
        if group not in enabled:
            excluded = excluded | cols
    log.debug(
        "Feature selector: league=%s → enabled=%s, excluded=%d cols",
        league_key, sorted(g.value for g in enabled), len(excluded),
    )
    return excluded
