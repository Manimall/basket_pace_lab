"""Static per-league configuration for the Flashscore results-page collector.

Pure data extracted from ``collector_page.py`` to keep that module under the
250-line limit and to centralise all hardcoded league paths / season labels
in one place (zero hardcode inside the scraping logic).
"""
from __future__ import annotations

from typing import TypedDict

from src.database.models import SeasonType


class CollectorLeague(TypedDict):
    """Static config for one Flashscore-collected league.

    Attributes:
        path: Flashscore results-page path fragment (without the trailing
            ``results/``).
        tournament_name: Value stored in ``matches.tournament_name``.
        season: Human-readable season label stored on each match.
        season_type: Regular / playoff classification.
    """

    path:            str
    tournament_name: str
    season:          str
    season_type:     SeasonType


# league_key → config. Several ABA seasons share tournament_name="ABA" so their
# data merges into one bucket downstream.
COLLECTOR_LEAGUES: dict[str, CollectorLeague] = {
    "LegaA": {
        "path":            "/basketball/italy/lega-a/",
        "tournament_name": "LegaA",
        "season":          "Lega A 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "PBA_PhilCup": {
        "path":            "/basketball/philippines/pba-philippine-cup/",
        "tournament_name": "PBA_PhilCup",
        "season":          "PBA Philippine Cup 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "PBA_CommCup": {
        "path":            "/basketball/philippines/pba-commissioner-s-cup/",
        "tournament_name": "PBA_CommCup",
        "season":          "PBA Commissioner's Cup 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "PBA_GovCup": {
        "path":            "/basketball/philippines/pba-governors-cup/",
        "tournament_name": "PBA_GovCup",
        "season":          "PBA Governors' Cup 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "Taiwan_PLeague": {
        "path":            "/basketball/taiwan/p-league/",
        "tournament_name": "Taiwan_PLeague",
        "season":          "P.League+ 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "Taiwan_TPBL": {
        "path":            "/basketball/taiwan/tpbl/",
        "tournament_name": "Taiwan_TPBL",
        "season":          "TPBL 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "ABA": {
        "path":            "/basketball/europe/admiralbet-aba-league/",
        "tournament_name": "ABA",
        "season":          "ABA League 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "ABA_2425": {
        "path":            "/basketball/europe/admiralbet-aba-league-2024-2025/",
        "tournament_name": "ABA",
        "season":          "ABA League 24/25",
        "season_type":     SeasonType.REGULAR,
    },
    "ABA_2324": {
        "path":            "/basketball/europe/admiralbet-aba-league-2023-2024/",
        "tournament_name": "ABA",
        "season":          "ABA League 23/24",
        "season_type":     SeasonType.REGULAR,
    },
    "Israel": {
        "path":            "/basketball/israel/super-league/",
        "tournament_name": "Israel",
        "season":          "Super League 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "VTB": {
        "path":            "/basketball/russia/vtb-united-league/",
        "tournament_name": "VTB",
        "season":          "VTB United League 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "PLK": {
        "path":            "/basketball/poland/basket-liga/",
        "tournament_name": "PLK",
        "season":          "Basket Liga 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "LKL": {
        "path":            "/basketball/lithuania/lkl/",
        "tournament_name": "LKL",
        "season":          "LKL 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "BSL": {
        "path":            "/basketball/turkey/super-lig/",
        "tournament_name": "BSL",
        "season":          "Super Lig 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    # ── High-pace regional / summer leagues (2025/2026) ──
    "KBL": {
        # South Korea — crosses the 2025→2026 calendar boundary (Oct–Apr).
        "path":            "/basketball/south-korea/kbl/",
        "tournament_name": "KBL",
        "season":          "KBL 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "LNB_DR": {
        # Dominican Republic — key disambiguated from France's "LNB".
        "path":            "/basketball/dominican-republic/lnb/",
        "tournament_name": "LNB_DR",
        "season":          "LNB Dominicana 2026",
        "season_type":     SeasonType.REGULAR,
    },
    "BSN": {
        # Puerto Rico — Baloncesto Superior Nacional (summer, high totals).
        "path":            "/basketball/puerto-rico/bsn/",
        "tournament_name": "BSN",
        "season":          "BSN 2026",
        "season_type":     SeasonType.REGULAR,
    },
    "WNBA": {
        "path":            "/basketball/usa/wnba/",
        "tournament_name": "WNBA",
        "season":          "WNBA 2026",
        "season_type":     SeasonType.REGULAR,
    },
    # ── B.League (Japan) prior-season archive, for multi-season walk-forward.
    # Same tournament_name as the current 25/26 data so they pool into one league.
    "BLeague_2425": {
        "path":            "/basketball/japan/b-league-2024-2025/",
        "tournament_name": "BLeague",
        "season":          "B.League 24/25",
        "season_type":     SeasonType.REGULAR,
    },
}
