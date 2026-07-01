"""Structural invariants for the league registries.

Guards the two league configs against the mistakes that silently break ingest:
slug typos, tournament_name/key drift (which breaks the Sofascore id-finder,
since it loads DB rows by ``tournament_name == catalog key``), and duplicate
Sofascore tournament IDs (e.g. accidentally reusing France's LNB id for the
Dominican LNB).
"""
from __future__ import annotations

import pytest

from src.data_collection.flashscore.collector_config import COLLECTOR_LEAGUES
from src.data_collection.sofascore.catalog import LEAGUE_CATALOG

# High-pace regional / summer leagues added for the 2025/2026 push.
_NEW_LEAGUES: tuple[str, ...] = ("KBL", "LNB_DR", "BSN", "WNBA")


@pytest.mark.parametrize("key", _NEW_LEAGUES)
def test_new_league_in_both_registries(key: str) -> None:
    assert key in COLLECTOR_LEAGUES, f"{key} missing from COLLECTOR_LEAGUES"
    assert key in LEAGUE_CATALOG, f"{key} missing from LEAGUE_CATALOG"


@pytest.mark.parametrize("key", _NEW_LEAGUES)
def test_new_league_tournament_name_matches_key(key: str) -> None:
    # The id-finder resolves DB rows via tournament_name == catalog key when no
    # --db-name override is given, so the two MUST agree for enrichment to work.
    assert COLLECTOR_LEAGUES[key]["tournament_name"] == key


@pytest.mark.parametrize("key", _NEW_LEAGUES)
def test_new_league_season_is_current(key: str) -> None:
    seasons = LEAGUE_CATALOG[key]["seasons"]
    assert seasons, f"{key} has no seasons"
    assert seasons[0]["code"] == "2526", f"{key} newest season is not 2025/2026"


def test_all_collector_slugs_well_formed() -> None:
    for key, cfg in COLLECTOR_LEAGUES.items():
        path = cfg["path"]
        assert path.startswith("/basketball/") and path.endswith("/"), f"bad slug for {key}: {path}"


def test_no_duplicate_sofascore_tournament_ids() -> None:
    ids = [entry["tournament_id"] for entry in LEAGUE_CATALOG.values()]
    assert len(ids) == len(set(ids)), "duplicate Sofascore tournament_id in LEAGUE_CATALOG"


def test_dominican_lnb_distinct_from_france_lnb() -> None:
    """The DR league must not shadow France's LNB (tournament 156)."""
    assert LEAGUE_CATALOG["LNB"]["tournament_id"] == 156
    assert LEAGUE_CATALOG["LNB_DR"]["tournament_id"] != LEAGUE_CATALOG["LNB"]["tournament_id"]
