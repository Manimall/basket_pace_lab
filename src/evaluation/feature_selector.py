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
        FATIGUE: V7 schedule-fatigue columns (10 cols). Densely-scheduled
            leagues benefit; sparsely-scheduled ones get noise.
        ADVANCED: Four-Factors metrics (ORtg/DRtg/TOV%/true_pace + has_boxscore).
            Only meaningful where box-score exists (Sofascore). At homogeneous
            NBA pace these collapse to scaled copies of raw points
            (multicollinearity) and add noise — disabled by default.
        ARENA: V8 home-court fortress / away vulnerability indices. European
            leagues have strong home-court regimes; enabled there, off for NBA.
        TEAM_ADV: Box-score advanced metrics (ORtg/DRtg/true_pace/3PA/3PA-rate,
            rolled L5+EMA) sourced from team_match_advanced (Go scout). Real
            efficiency signal for leagues the box score was scraped for —
            currently EuroLeague (current season fully backfilled).
    """
    BASE     = "base"
    FATIGUE  = "fatigue"
    ADVANCED = "advanced"
    ARENA    = "arena"
    TEAM_ADV = "team_adv"


# Leagues with structurally strong home-court advantage (European hardcore):
# ARENA group is enabled for these. tournament_name values as stored in DB.
_ARENA_LEAGUES: tuple[str, ...] = (
    "EuroLeague", "ABA", "ACB", "BBL", "LNB", "LegaA", "VTB", "Israel",
)

# Per-league enabled groups. Unlisted leagues fall back to DEFAULT_GROUPS.
# Documented empirically in postmortem V7 / V7.1 / V8.
#
# NBA: fatigue helps (dense calendar). ADVANCED off — the isolation test showed
# Four-Factors metrics are redundant at NBA's uniform pace. ARENA off pending
# separate validation.
FEATURES_BY_LEAGUE: dict[str, frozenset[FeatureGroup]] = {
    "NBA": frozenset({FeatureGroup.BASE, FeatureGroup.FATIGUE}),
    # EuroLeague: box-score advanced backfilled (current season) → TEAM_ADV on.
    "EuroLeague": frozenset({FeatureGroup.BASE, FeatureGroup.ARENA, FeatureGroup.TEAM_ADV}),
    **{
        league: frozenset({FeatureGroup.BASE, FeatureGroup.ARENA})
        for league in _ARENA_LEAGUES
        if league != "EuroLeague"
    },
}

# Fallback for any unlisted league or for mixed-league pipelines where no
# single league key applies (e.g. the OTHER control group).
DEFAULT_GROUPS: frozenset[FeatureGroup] = frozenset({FeatureGroup.BASE})

# Group → the column names it controls. BASE is implicit (everything not
# listed in another group). Only non-base groups need an entry here.
_GROUP_COLUMNS: dict[FeatureGroup, frozenset[str]] = {
    FeatureGroup.FATIGUE:  frozenset(FATIGUE_FEATURE_COLS),
    FeatureGroup.ADVANCED: frozenset(ADVANCED_FEAT_COLS),
    FeatureGroup.ARENA:    frozenset(ARENA_FEAT_COLS),
    FeatureGroup.TEAM_ADV: frozenset(TEAM_ADV_FEAT_COLS),
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
