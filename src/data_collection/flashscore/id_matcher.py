"""
Universal Flashscore ID matcher.

Assigns flashscore_id to DB matches collected from other sources (SofaScore, etc.)
by fuzzy-matching date ± 1 day + normalised team name similarity.

Supports: NBA, BLeague, ChinaNBL, LNBP.

Usage:
    python -m src.data_collection.flashscore.id_matcher --leagues NBA BLeague
    python -m src.data_collection.flashscore.id_matcher --leagues ChinaNBL LNBP --dry-run
    python -m src.data_collection.flashscore.id_matcher --leagues NBA --limit 50
"""
from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any

from playwright.async_api import async_playwright
from sqlalchemy import text

from src.data_collection.base import BaseCollector
from src.data_collection.constants import FLASHSCORE_BASE_URL
from src.data_collection.flashscore.browser import make_flashscore_session
from src.data_collection.flashscore.norm import FsMatch, _norm, _sim
from src.data_collection.flashscore.page import build_index_from_url, find_match
from src.database.engine import dispose_engine, get_session_factory

log = logging.getLogger(__name__)

_MATCH_THRESHOLD = 0.45


@dataclass(frozen=True)
class _LeagueConfig:
    db_filter: str      # WHERE fragment; alias 'm' refers to the matches table
    fs_urls: list[str]  # full Flashscore results-page URLs (multiple = merged index)


_LEAGUES: dict[str, _LeagueConfig] = {
    "NBA": _LeagueConfig(
        db_filter="m.tournament_name IS NULL",
        fs_urls=[
            # Current season (2025-26)
            FLASHSCORE_BASE_URL + "/basketball/usa/nba/results/",
            # Previous season — most unmatched records are from 2024-25
            FLASHSCORE_BASE_URL + "/basketball/usa/nba-2024-2025/results/",
            # NBA Cup (In-Season Tournament) is on its own page
            FLASHSCORE_BASE_URL + "/basketball/usa/nba-cup/results/",
            # Playoffs — separate page
            FLASHSCORE_BASE_URL + "/basketball/usa/nba-playoffs/results/",
        ],
    ),
    "BLeague": _LeagueConfig(
        db_filter="m.tournament_name = 'BLeague'",
        fs_urls=[FLASHSCORE_BASE_URL + "/basketball/japan/b-league/results/"],
    ),
    "ChinaNBL": _LeagueConfig(
        db_filter="m.tournament_name = 'ChinaNBL'",
        # NOTE: Flashscore may not cover this league; URL is best-effort
        fs_urls=[
            FLASHSCORE_BASE_URL + "/basketball/china/nbl/results/",
            FLASHSCORE_BASE_URL + "/basketball/china/nbl-2/results/",
        ],
    ),
    "LNBP": _LeagueConfig(
        db_filter="m.tournament_name = 'LNBP'",
        fs_urls=[FLASHSCORE_BASE_URL + "/basketball/mexico/lnbp/results/"],
    ),
    "ABA": _LeagueConfig(
        db_filter="m.tournament_name = 'ABA'",
        fs_urls=[
            # Current season (2025/26) — sponsor naming "AdmiralBet ABA League"
            FLASHSCORE_BASE_URL + "/basketball/europe/admiralbet-aba-league/results/",
            # Previous seasons — kept for historical matching when archive data is ingested
            FLASHSCORE_BASE_URL + "/basketball/europe/admiralbet-aba-league-2024-2025/results/",
            FLASHSCORE_BASE_URL + "/basketball/europe/admiralbet-aba-league-2023-2024/results/",
        ],
    ),
    "Israel": _LeagueConfig(
        db_filter="m.tournament_name = 'Israel'",
        fs_urls=[FLASHSCORE_BASE_URL + "/basketball/israel/super-league/results/"],
    ),
}


@dataclass
class _DbMatch:
    match_id:   int
    match_date: date
    home_team:  str
    away_team:  str
    home_norm:  str
    away_norm:  str


def _score(target: _DbMatch, entry: FsMatch) -> float:
    return (_sim(target.home_norm, entry.home_norm) + _sim(target.away_norm, entry.away_norm)) / 2


async def _load_db_matches(
    sf: Any, db_filter: str, limit: int | None,
) -> list[_DbMatch]:
    lim = f"LIMIT {int(limit)}" if limit else ""
    sql = f"""
        SELECT m.id, m.scheduled_at::date AS match_date,
               ht.name AS home_team, at.name AS away_team
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE {db_filter}
          AND m.flashscore_id IS NULL
          AND m.home_score_final IS NOT NULL
        ORDER BY m.scheduled_at DESC
        {lim}
    """
    async with sf() as db:
        rows = (await db.execute(text(sql))).mappings().all()
    return [
        _DbMatch(
            match_id=int(r["id"]),
            match_date=r["match_date"],
            home_team=r["home_team"],
            away_team=r["away_team"],
            home_norm=_norm(r["home_team"]),
            away_norm=_norm(r["away_team"]),
        )
        for r in rows
    ]


async def _save_flashscore_id(sf: Any, match_id: int, fs_id: str) -> None:
    async with sf() as db:
        async with db.begin():
            await db.execute(
                text("UPDATE matches SET flashscore_id = :fs_id WHERE id = :mid"),
                {"fs_id": fs_id, "mid": match_id},
            )


class FlashscoreIdMatcher(BaseCollector):
    def __init__(
        self,
        leagues: list[str],
        dry_run: bool = False,
        limit: int | None = None,
    ) -> None:
        unknown = [l for l in leagues if l not in _LEAGUES]
        if unknown:
            raise ValueError(f"Unknown leagues: {unknown}. Available: {list(_LEAGUES)}")
        self._leagues = leagues
        self._dry_run = dry_run
        self._limit   = limit
        self._sf      = get_session_factory()

    async def run(self) -> None:
        totals = {"matched": 0, "unmatched": 0}

        async with async_playwright() as pw:
            browser, page = await make_flashscore_session(pw)
            try:
                for league in self._leagues:
                    await self._run_league(page, league, _LEAGUES[league], totals)
            finally:
                await browser.close()

        await dispose_engine()
        log.info("━" * 60)
        log.info(
            "OVERALL  matched=%d  unmatched=%d%s",
            totals["matched"], totals["unmatched"],
            "  [DRY RUN]" if self._dry_run else "",
        )

    async def _run_league(
        self,
        page: Any,
        league: str,
        cfg: _LeagueConfig,
        totals: dict[str, int],
    ) -> None:
        db_matches = await _load_db_matches(self._sf, cfg.db_filter, self._limit)
        log.info("[%s] DB matches without flashscore_id: %d", league, len(db_matches))
        if not db_matches:
            return

        # Merge indices from all configured URLs, deduplicating by fs_id
        index: list[FsMatch] = []
        seen: set[str] = set()
        for url in cfg.fs_urls:
            for entry in await build_index_from_url(page, url):
                if entry.fs_id not in seen:
                    seen.add(entry.fs_id)
                    index.append(entry)
        log.info("[%s] Flashscore index: %d entries total", league, len(index))

        matched = unmatched = 0
        unmatched_examples: list[tuple[_DbMatch, str]] = []

        for target in db_matches:
            fs_match = find_match(
                index, target.match_date,
                target.home_team, target.away_team,
                threshold=_MATCH_THRESHOLD,
            )
            if fs_match is None:
                unmatched += 1
                nearby = sorted(
                    [e for e in index if e.match_date and
                     abs((e.match_date - target.match_date).days) <= 2],
                    key=lambda e: -_score(target, e),
                )
                hint = (
                    f"{nearby[0].home_raw} vs {nearby[0].away_raw}"
                    if nearby else "no candidates"
                )
                unmatched_examples.append((target, hint))
            else:
                matched += 1
                log.debug(
                    "  MATCH %s | %s vs %s → %s",
                    target.match_date, target.home_team, target.away_team, fs_match.fs_id,
                )
                if not self._dry_run:
                    await _save_flashscore_id(self._sf, target.match_id, fs_match.fs_id)

        totals["matched"] += matched
        totals["unmatched"] += unmatched
        log.info(
            "[%s] matched=%d  unmatched=%d  total=%d%s",
            league, matched, unmatched, len(db_matches),
            "  [DRY RUN]" if self._dry_run else "",
        )
        if unmatched_examples:
            log.info("[%s] Unmatched examples (DB → closest Flashscore candidate):", league)
            for target, hint in unmatched_examples[:5]:
                log.info(
                    "  %s | %s vs %s  →  closest: %s",
                    target.match_date, target.home_team, target.away_team, hint,
                )


def _parse_args() -> dict[str, Any]:
    args = sys.argv[1:]
    opts: dict[str, Any] = {"leagues": list(_LEAGUES), "dry_run": False, "limit": None}
    i = 0
    while i < len(args):
        if args[i] == "--leagues" and i + 1 < len(args):
            vals: list[str] = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                vals.append(args[i]); i += 1
            opts["leagues"] = vals
        elif args[i] == "--dry-run":
            opts["dry_run"] = True; i += 1
        elif args[i] == "--limit" and i + 1 < len(args):
            opts["limit"] = int(args[i + 1]); i += 2
        else:
            i += 1
    return opts


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        stream=sys.stdout,
    )
    opts = _parse_args()
    FlashscoreIdMatcher(
        leagues=opts["leagues"],
        dry_run=opts["dry_run"],
        limit=opts["limit"],
    ).run_sync()
