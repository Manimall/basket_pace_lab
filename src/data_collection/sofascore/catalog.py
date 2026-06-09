"""
Sofascore league / season catalog.

LEAGUE_CATALOG maps short league keys to Sofascore tournament IDs and seasons.
Season codes ("2526", "2425", "2324") are used for CLI --seasons filtering.
"""
from __future__ import annotations

from typing import TypedDict


class SeasonEntry(TypedDict):
    """One Sofascore season within a tournament.

    Attributes:
        id: Sofascore season id (used in the events API path).
        name: Human-readable season label.
        code: Short season code ("2526" …) used for ``--seasons`` filtering.
    """

    id:   int
    name: str
    code: str


class LeagueEntry(TypedDict):
    """One league's Sofascore tournament id and its seasons.

    Attributes:
        tournament_id: Sofascore unique-tournament id.
        seasons: Ordered list of seasons (newest first by convention).
    """

    tournament_id: int
    seasons:       list[SeasonEntry]


LEAGUE_CATALOG: dict[str, LeagueEntry] = {
    "NBA": {
        "tournament_id": 132,
        "seasons": [
            {"id": 80229, "name": "NBA 25/26",  "code": "2526"},
            {"id": 65360, "name": "NBA 24/25",  "code": "2425"},
            {"id": 54105, "name": "NBA 23/24",  "code": "2324"},
        ],
    },
    "EuroLeague": {
        "tournament_id": 138,
        "seasons": [
            {"id": 78545, "name": "Euroleague 25/26", "code": "2526"},
            {"id": 63971, "name": "Euroleague 24/25", "code": "2425"},
            {"id": 53198, "name": "Euroleague 23/24", "code": "2324"},
        ],
    },
    "VTB": {
        "tournament_id": 1438,
        "seasons": [
            {"id": 80491, "name": "United League 25/26", "code": "2526"},
            {"id": 64482, "name": "United League 24/25", "code": "2425"},
            {"id": 53040, "name": "United League 23/24", "code": "2324"},
        ],
    },
    "ACB": {
        "tournament_id": 264,
        "seasons": [
            {"id": 80922, "name": "Liga ACB 25/26", "code": "2526"},
            {"id": 64689, "name": "Liga ACB 24/25", "code": "2425"},
            {"id": 53749, "name": "Liga ACB 23/24", "code": "2324"},
        ],
    },
    "LegaA": {
        "tournament_id": 262,
        "seasons": [
            {"id": 79529, "name": "Serie A 25/26", "code": "2526"},
            {"id": 64742, "name": "Serie A 24/25", "code": "2425"},
            {"id": 53536, "name": "Serie A 23/24", "code": "2324"},
        ],
    },
    "BBL": {
        "tournament_id": 227,
        "seasons": [
            {"id": 79994, "name": "BBL 25/26", "code": "2526"},
            {"id": 65031, "name": "BBL 24/25", "code": "2425"},
            {"id": 52951, "name": "BBL 23/24", "code": "2324"},
        ],
    },
    "LNB": {
        "tournament_id": 156,
        "seasons": [
            {"id": 79100, "name": "Pro A 25/26", "code": "2526"},
            {"id": 64004, "name": "Pro A 24/25", "code": "2425"},
            {"id": 53266, "name": "Pro A 23/24", "code": "2324"},
        ],
    },
    "NBL": {
        "tournament_id": 1524,
        "seasons": [
            {"id": 77205, "name": "NBL 25/26", "code": "2526"},
            {"id": 61848, "name": "NBL 24/25", "code": "2425"},
            {"id": 52012, "name": "NBL 23/24", "code": "2324"},
        ],
    },
    "CBA": {
        "tournament_id": 1566,
        "seasons": [
            {"id": 85375, "name": "CBA 25/26", "code": "2526"},
            {"id": 67166, "name": "CBA 24/25", "code": "2425"},
            {"id": 55486, "name": "CBA 23/24", "code": "2324"},
        ],
    },
    "ChinaNBL": {
        "tournament_id": 27568,
        "seasons": [
            {"id": 87684, "name": "NBL 25/26", "code": "2526"},
            {"id": 77353, "name": "NBL 2025",  "code": "2425"},
        ],
    },
    "PBA_Phil": {
        "tournament_id": 1956,
        "seasons": [
            {"id": 84100, "name": "PBA Philippine Cup 25/26", "code": "2526"},
            {"id": 74130, "name": "PBA Philippine Cup 2025",  "code": "2425"},
            {"id": 58687, "name": "PBA Philippine Cup 2024",  "code": "2324"},
        ],
    },
    "PBA_Comm": {
        "tournament_id": 1656,
        "seasons": [
            {"id": 90891, "name": "PBA Commissioner Cup 2026",  "code": "2526"},
            {"id": 69126, "name": "PBA Commissioner Cup 24/25", "code": "2425"},
            {"id": 56111, "name": "PBA Commissioner Cup 23/24", "code": "2324"},
        ],
    },
    "PBA_Gov": {
        "tournament_id": 1712,
        "seasons": [
            {"id": 65381, "name": "PBA Governors Cup 2024", "code": "2425"},
            {"id": 48362, "name": "PBA Governors Cup 22/23", "code": "2324"},
        ],
    },
    "LNBP": {
        "tournament_id": 1472,
        "seasons": [
            {"id": 75884, "name": "LNBP 2025", "code": "2526"},
            {"id": 61417, "name": "LNBP 2024", "code": "2425"},
            {"id": 52174, "name": "LNBP 2023", "code": "2324"},
        ],
    },
    "BLeague": {
        "tournament_id": 1502,
        "seasons": [
            {"id": 77915, "name": "B1 League 25/26", "code": "2526"},
            {"id": 64082, "name": "B1 League 24/25", "code": "2425"},
            {"id": 54374, "name": "B1 League 23/24", "code": "2324"},
        ],
    },
    # ── Leagues added for sofascore_id_finder (previously Flashscore-only) ──
    "ABA": {
        # AdmiralBet ABA League
        # sofascore.com/basketball/tournament/international/admiralbet-aba-league/235
        "tournament_id": 235,
        "seasons": [
            {"id": 80150, "name": "AdmiralBet ABA League 25/26", "code": "2526"},
            {"id": 61743, "name": "Liga ABA 24/25",              "code": "2425"},
            {"id": 53473, "name": "Liga ABA 23/24",              "code": "2324"},
        ],
    },
    "Israel": {
        # Israeli Basketball Super League
        # sofascore.com/basketball/tournament/israel/super-league/1197
        "tournament_id": 1197,
        "seasons": [
            {"id": 81980, "name": "Super League 25/26", "code": "2526"},
            {"id": 66111, "name": "Super League 24/25", "code": "2425"},
            {"id": 54443, "name": "Super League 23/24", "code": "2324"},
        ],
    },
    "PLK": {
        # Orlen Basket Liga (Poland top division)
        # sofascore.com/basketball/tournament/poland/orlen-basket-liga/263
        "tournament_id": 263,
        "seasons": [
            {"id": 77879, "name": "PLK 25/26", "code": "2526"},
            {"id": 65624, "name": "PLK 24/25", "code": "2425"},
            {"id": 54011, "name": "PLK 23/24", "code": "2324"},
        ],
    },
    "LKL": {
        # Betsson-LKL (Lithuania top division)
        # sofascore.com/basketball/tournament/lithuania/betsafe-lkl/975
        "tournament_id": 975,
        "seasons": [
            {"id": 80356, "name": "LKL 25/26", "code": "2526"},
            {"id": 65649, "name": "LKL 24/25", "code": "2425"},
            {"id": 54106, "name": "LKL 23/24", "code": "2324"},
        ],
    },
    "BSL": {
        # Turkish Basketball Super League (Basketbol Süper Ligi)
        # sofascore.com/basketball/tournament/turkey/super-lig/519
        "tournament_id": 519,
        "seasons": [
            {"id": 81036, "name": "Super Lig 25/26", "code": "2526"},
            {"id": 65808, "name": "Super Lig 24/25", "code": "2425"},
            {"id": 54528, "name": "Super Lig 23/24", "code": "2324"},
        ],
    },
}
