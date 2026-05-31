"""Static per-league configuration for the Flashscore ID matcher.

Maps each league to its DB filter and the Flashscore results-page URLs whose
match indices are merged for fuzzy matching. Pure data — extracted from
``id_matcher.py`` to keep that orchestration module under the 250-line limit
and to satisfy the zero-hardcode rule (no URLs inline in matching logic).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.data_collection.constants import FLASHSCORE_BASE_URL

# Minimum fuzzy similarity to accept a Flashscore↔DB match.
MATCH_THRESHOLD: float = 0.45


@dataclass(frozen=True)
class LeagueConfig:
    """One league's DB filter and Flashscore source URLs."""

    db_filter: str       # WHERE fragment; alias 'm' refers to the matches table
    fs_urls:   list[str]  # full results-page URLs (multiple = merged index)


def _results(path: str) -> str:
    return f"{FLASHSCORE_BASE_URL}{path}/results/"


LEAGUES: dict[str, LeagueConfig] = {
    "NBA": LeagueConfig(
        db_filter="m.tournament_name IS NULL",
        fs_urls=[
            _results("/basketball/usa/nba"),
            _results("/basketball/usa/nba-2024-2025"),  # most unmatched are 24/25
            _results("/basketball/usa/nba-cup"),         # In-Season Tournament
            _results("/basketball/usa/nba-playoffs"),
        ],
    ),
    "BLeague": LeagueConfig(
        db_filter="m.tournament_name = 'BLeague'",
        fs_urls=[_results("/basketball/japan/b-league")],
    ),
    "ChinaNBL": LeagueConfig(
        db_filter="m.tournament_name = 'ChinaNBL'",
        fs_urls=[  # Flashscore coverage uncertain; URLs are best-effort
            _results("/basketball/china/nbl"),
            _results("/basketball/china/nbl-2"),
        ],
    ),
    "LNBP": LeagueConfig(
        db_filter="m.tournament_name = 'LNBP'",
        fs_urls=[_results("/basketball/mexico/lnbp")],
    ),
    "ABA": LeagueConfig(
        db_filter="m.tournament_name = 'ABA'",
        fs_urls=[
            _results("/basketball/europe/admiralbet-aba-league"),
            _results("/basketball/europe/admiralbet-aba-league-2024-2025"),
            _results("/basketball/europe/admiralbet-aba-league-2023-2024"),
        ],
    ),
    "Israel": LeagueConfig(
        db_filter="m.tournament_name = 'Israel'",
        fs_urls=[_results("/basketball/israel/super-league")],
    ),
}
