"""Orchestration for the Sofascore ID finder.

Ties together the client (fetch), repository (DB), and matcher (fuzzy) layers:
for each league, fetch its Sofascore events across the catalog seasons, match
each Flashscore-sourced DB row, and patch ``external_id`` in place (unless dry-run).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from curl_cffi.requests import AsyncSession

from src.data_collection.sofascore.catalog import LEAGUE_CATALOG
from src.data_collection.sofascore.id_finder.config import (
    IMPERSONATE_PROFILE,
    INTER_SEASON_SLEEP,
    MAX_UNMATCHED_HINTS,
    THRESHOLD_FLOOR,
    THRESHOLD_STEP,
)
from src.data_collection.sofascore.id_finder.matcher import (
    closest_hint,
    find_best_match,
    score_pair,
)
from src.data_collection.sofascore.id_finder.models import DbRow, SofaEvent
from src.data_collection.sofascore.id_finder.repository import load_fs_rows, patch_external_id
from src.data_collection.sofascore.id_finder.sofa_client import (
    build_headers,
    fetch_events_for_season,
    load_cookie_header,
)
from src.database.engine import dispose_engine, get_session_factory

log = logging.getLogger(__name__)


@dataclass
class _Totals:
    matched:   int = 0
    unmatched: int = 0


@dataclass(frozen=True)
class FinderOptions:
    """All inputs for one ID-finder run (typed instead of loose kwargs)."""

    leagues:          list[str]
    db_name_override: str | None
    seasons_filter:   list[str] | None
    threshold:        float
    dry_run:          bool


async def _collect_season_events(
    session: AsyncSession, league_key: str, opts: FinderOptions, headers: dict[str, str],
) -> list[SofaEvent]:
    cat = LEAGUE_CATALOG[league_key]
    all_events: list[SofaEvent] = []
    seen: set[int] = set()
    for season in cat["seasons"]:
        if opts.seasons_filter and season.get("code") not in opts.seasons_filter:
            continue
        log.info("[%s] Fetching season %s (id=%d)…", league_key, season["name"], season["id"])
        await asyncio.sleep(INTER_SEASON_SLEEP)
        evs = await fetch_events_for_season(session, cat["tournament_id"], season["id"], headers)
        fresh = [e for e in evs if e.event_id not in seen]
        seen.update(e.event_id for e in fresh)
        all_events.extend(fresh)
        log.info("[%s] %s → %d events (%d unique total)",
                 league_key, season["name"], len(evs), len(all_events))
    return all_events


async def _match_and_patch(
    sf, league_key: str, db_rows: list[DbRow], events: list[SofaEvent], opts: FinderOptions,
) -> tuple[int, int]:
    matched = unmatched = 0
    hints: list[tuple[DbRow, str]] = []
    for row in db_rows:
        best = find_best_match(row, events, opts.threshold)
        if best is None:
            unmatched += 1
            hints.append((row, closest_hint(row, events)))
            continue
        if not opts.dry_run:
            await patch_external_id(sf, row.match_id, str(best.event_id))
        log.debug("  MATCH %s | %s vs %s → %d (score=%.2f)",
                  row.match_date, row.home_raw, row.away_raw, best.event_id,
                  score_pair(row, best))
        matched += 1

    suffix = "  [DRY-RUN — no writes]" if opts.dry_run else ""
    log.info("[%s] matched=%d  unmatched=%d%s", league_key, matched, unmatched, suffix)
    for row, hint in hints[:MAX_UNMATCHED_HINTS]:
        log.info("  %s | %s vs %s  →  %s", row.match_date, row.home_raw, row.away_raw, hint)
    return matched, unmatched


async def run(opts: FinderOptions) -> None:
    """Execute the ID-finder for every requested league."""
    sf = get_session_factory()
    cookie_header = load_cookie_header()
    headers = build_headers(cookie_header)
    if cookie_header:
        log.info("Loaded cookies (%d bytes)", len(cookie_header))
    else:
        log.warning("No cookies — Sofascore may return 403.")

    totals = _Totals()
    async with AsyncSession(impersonate=IMPERSONATE_PROFILE) as session:
        for league_key in opts.leagues:
            if league_key not in LEAGUE_CATALOG:
                log.error("League '%s' not in LEAGUE_CATALOG — add it first.", league_key)
                continue

            db_tname = opts.db_name_override or league_key
            log.info("━" * 60)
            log.info("[%s] tournament_id=%d  db_filter='%s'",
                     league_key, LEAGUE_CATALOG[league_key]["tournament_id"], db_tname)

            db_rows = await load_fs_rows(sf, db_tname)
            log.info("[%s] DB rows with Flashscore IDs: %d", league_key, len(db_rows))
            if not db_rows:
                log.info("[%s] Nothing to patch.", league_key)
                continue

            events = await _collect_season_events(session, league_key, opts, headers)
            if not events:
                log.warning("[%s] No Sofascore events — check tournament/season IDs.", league_key)
                continue

            matched, unmatched = await _match_and_patch(sf, league_key, db_rows, events, opts)
            totals.matched   += matched
            totals.unmatched += unmatched

    await dispose_engine()
    log.info("━" * 60)
    log.info("TOTAL  matched=%d  unmatched=%d%s",
             totals.matched, totals.unmatched, "  [DRY-RUN]" if opts.dry_run else "")
    if totals.unmatched and not opts.dry_run:
        log.info("Tip: re-run with --threshold %.2f to catch more matches.",
                 max(THRESHOLD_FLOOR, opts.threshold - THRESHOLD_STEP))
