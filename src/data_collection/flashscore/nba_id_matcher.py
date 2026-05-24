"""
NBA flashscore_id matcher.

NBA matches in our DB were collected via SofaScore and have no flashscore_id.
This script loads the NBA results page from Flashscore, matches each DB record
by date + normalised team name, and updates flashscore_id in the DB.

Usage:
    python -m src.data_collection.flashscore.nba_id_matcher
    python -m src.data_collection.flashscore.nba_id_matcher --dry-run
    python -m src.data_collection.flashscore.nba_id_matcher --limit 200
"""
from __future__ import annotations

import asyncio
import logging
import random
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any

from playwright.async_api import async_playwright
from sqlalchemy import text

from src.config import settings
from src.data_collection.constants import FLASHSCORE_BASE_URL, USER_AGENTS
from src.data_collection.flashscore.norm import _norm, _sim
from src.data_collection.flashscore.odds import dismiss_overlays
from src.data_collection.flashscore.page import build_league_index, FsMatch
from src.database.engine import dispose_engine, get_session_factory

log = logging.getLogger(__name__)

_MATCH_THRESHOLD: float = 0.45
_DATE_WINDOW_DAYS: int  = 1


@dataclass
class _DbNbaMatch:
    match_id:   int
    match_date: date
    home_team:  str
    away_team:  str
    home_norm:  str
    away_norm:  str


async def _load_db_matches(sf: Any, limit: int | None) -> list[_DbNbaMatch]:
    lim = f"LIMIT {int(limit)}" if limit else ""
    sql = f"""
        SELECT m.id, m.scheduled_at::date AS match_date,
               ht.name AS home_team, at.name AS away_team
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE m.tournament_name IS NULL
          AND m.flashscore_id IS NULL
          AND m.home_score_final IS NOT NULL
        ORDER BY m.scheduled_at DESC
        {lim}
    """
    async with sf() as db:
        rows = (await db.execute(text(sql))).mappings().all()
    return [
        _DbNbaMatch(
            match_id=int(r["id"]),
            match_date=r["match_date"],
            home_team=r["home_team"],
            away_team=r["away_team"],
            home_norm=_norm(r["home_team"]),
            away_norm=_norm(r["away_team"]),
        )
        for r in rows
    ]


def _find_best(
    index: list[FsMatch],
    target: _DbNbaMatch,
) -> FsMatch | None:
    best: FsMatch | None = None
    best_score = 0.0
    for entry in index:
        if entry.match_date is None:
            continue
        if abs((entry.match_date - target.match_date).days) > _DATE_WINDOW_DAYS:
            continue
        score = (
            _sim(target.home_norm, entry.home_norm)
            + _sim(target.away_norm, entry.away_norm)
        ) / 2
        if score > best_score:
            best_score, best = score, entry
    return best if best_score >= _MATCH_THRESHOLD else None


async def _update_flashscore_id(sf: Any, match_id: int, fs_id: str) -> None:
    async with sf() as db:
        async with db.begin():
            await db.execute(
                text("UPDATE matches SET flashscore_id = :fs_id WHERE id = :mid"),
                {"fs_id": fs_id, "mid": match_id},
            )


async def run(dry_run: bool = False, limit: int | None = None) -> None:
    sf  = get_session_factory()
    log.info("DB: %s:%s/%s", settings.db.host, settings.db.port, settings.db.name)

    db_matches = await _load_db_matches(sf, limit)
    log.info("NBA matches without flashscore_id: %d", len(db_matches))
    if not db_matches:
        log.info("Nothing to match.")
        await dispose_engine()
        return

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx     = await browser.new_context(user_agent=random.choice(USER_AGENTS))
        page    = await ctx.new_page()

        log.info("Building NBA index from Flashscore…")
        await page.goto(
            FLASHSCORE_BASE_URL + "/basketball/usa/nba/",
            wait_until="domcontentloaded", timeout=25_000,
        )
        await asyncio.sleep(3)
        await dismiss_overlays(page)
        index = await build_league_index(page, "NBA")
        log.info("Flashscore NBA index: %d entries", len(index))

        await browser.close()

    matched = unmatched = 0
    unmatched_examples: list[tuple[_DbNbaMatch, str]] = []

    for target in db_matches:
        fs_match = _find_best(index, target)
        if fs_match is None:
            unmatched += 1
            # Collect examples: find closest date entries for diagnosis
            candidates = sorted(
                [e for e in index if e.match_date and
                 abs((e.match_date - target.match_date).days) <= 2],
                key=lambda e: -(
                    _sim(target.home_norm, e.home_norm)
                    + _sim(target.away_norm, e.away_norm)
                ),
            )
            hint = candidates[0].home_raw + " vs " + candidates[0].away_raw if candidates else "no candidates"
            unmatched_examples.append((target, hint))
        else:
            matched += 1
            log.debug(
                "  MATCH %s | %s vs %s → fs=%s",
                target.match_date, target.home_team, target.away_team, fs_match.fs_id,
            )
            if not dry_run:
                await _update_flashscore_id(sf, target.match_id, fs_match.fs_id)

    await dispose_engine()

    log.info("━" * 60)
    log.info(
        "DONE  matched=%d  unmatched=%d  total=%d%s",
        matched, unmatched, len(db_matches),
        "  [DRY RUN]" if dry_run else "",
    )
    if unmatched_examples:
        log.info("First unmatched examples (db → closest Flashscore candidate):")
        for target, hint in unmatched_examples[:5]:
            log.info(
                "  %s | %s vs %s  →  closest: %s",
                target.match_date, target.home_team, target.away_team, hint,
            )


def _parse_args() -> dict[str, Any]:
    args = sys.argv[1:]
    opts: dict[str, Any] = {"dry_run": False, "limit": None}
    i = 0
    while i < len(args):
        if args[i] == "--dry-run":
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
    asyncio.run(run(dry_run=opts["dry_run"], limit=opts["limit"]))
