#!/usr/bin/env python3
"""Patch external_id for Flashscore-sourced matches with Sofascore numeric IDs.

Problem
-------
Matches collected via the Flashscore collector have ``external_id = 'fs_XXXXXXXX'``.
The Go scout's SQL filter (``external_id ~ '^[0-9]+$'``) ignores them, so they
never get box-score enrichment (``team_match_advanced`` stays empty → TEAM_ADV
features are league-mean filled for these leagues in the ML pipeline).

Fix
---
This script fetches the Sofascore event list for each target league, fuzzy-matches
each event to a DB row (date ± 1 day + team-name similarity), and updates
``external_id`` to the numeric Sofascore event ID in-place.  After a successful
run, ``etl_scout --loop`` can process these rows normally.

Pre-requisites
--------------
* The league key must appear in ``src.data_collection.sofascore.catalog.LEAGUE_CATALOG``
  with a valid ``tournament_id`` and season ``id`` values.
* ``DB_HOST`` / ``DB_PORT`` / … env vars (or .env) must point to a live Postgres.
* ``cookies.json`` must exist in the project root (same as for the Go scout).

Usage
-----
    # Dry-run — no DB writes, prints what would be matched
    python scripts/sofascore_id_finder.py --leagues LegaA --dry-run

    # Run for a single league, current season only
    python scripts/sofascore_id_finder.py --leagues LegaA --seasons 2526

    # Run for multiple leagues (all seasons in catalog)
    python scripts/sofascore_id_finder.py --leagues ABA Israel LegaA

    # Lower similarity threshold (default 0.50) if many unmatched remain
    python scripts/sofascore_id_finder.py --leagues ABA --threshold 0.45

Name-mismatch reference
-----------------------
Some league keys in the catalog differ from ``tournament_name`` stored in the DB
(e.g. PBA_Phil → PBA_PhilCup).  Use ``--db-name`` to override the filter:

    python scripts/sofascore_id_finder.py --leagues PBA_Phil --db-name PBA_PhilCup
    python scripts/sofascore_id_finder.py --leagues PBA_Comm --db-name PBA_CommCup
    python scripts/sofascore_id_finder.py --leagues PBA_Gov  --db-name PBA_GovCup
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import re

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from curl_cffi.requests import AsyncSession
from sqlalchemy import text

from src.data_collection.sofascore.catalog import LEAGUE_CATALOG
from src.database.engine import dispose_engine, get_session_factory

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

# ── Sofascore client constants (mirrors sofascore/client.py) ─────────────────

_COOKIES_PATH = Path(__file__).resolve().parent.parent / "cookies.json"
_BASE_URL     = "https://api.sofascore.com/api/v1"
_HEADERS: dict[str, str] = {
    "Accept":           "application/json, text/plain, */*",
    "Accept-Language":  "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Origin":           "https://www.sofascore.com",
    "Referer":          "https://www.sofascore.com/",
    "Cache-Control":    "no-cache",
    "Pragma":           "no-cache",
}

# ── Team-name normalisation (mirrors flashscore/norm.py) ─────────────────────

_STOP = re.compile(
    r"\b(basketball|club|bc|bk|fc|sk|ak|kk|as|bc|bb|sport|bball|city|team|"
    r"real|istanbul|milan|milano|london|basket|baskets|"
    r"telekom|ewe|ratiopharm|"
    r"ea7|emporio|armani|ldlc|beko|meridianbet|mozzart|bet|admiralbet|"
    r"maccabi|hapoel|ironi|bnei|elitzur|"  # Israeli prefixes
    r"segafredo|virtus|reyer|venezia|brescia|varese|reggiana)\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^a-z0-9 ]")


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    name = _STOP.sub(" ", name)
    name = _PUNCT.sub(" ", name)
    return " ".join(name.split())


def _sim(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / max(len(ta), len(tb))
    big, small = (ta, tb) if len(ta) >= len(tb) else (tb, ta)
    if small and small.issubset(big):
        containment = 0.5 + 0.25 * (len(small) / len(big))
        overlap = max(overlap, containment)
    return overlap


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class _SofaEvent:
    event_id:   int
    event_date: date
    home_raw:   str
    away_raw:   str
    home_norm:  str
    away_norm:  str


@dataclass
class _DbRow:
    match_id:   int
    ext_id:     str            # current fs_XXXXXXXX value
    match_date: date
    home_raw:   str
    away_raw:   str
    home_norm:  str
    away_norm:  str


# ── Sofascore API fetch ───────────────────────────────────────────────────────

def _load_cookies() -> list[dict] | None:
    if not _COOKIES_PATH.exists():
        return None
    try:
        data = json.loads(_COOKIES_PATH.read_text())
        return data if isinstance(data, list) else None
    except Exception:
        return None


async def _api_get(session: AsyncSession, path: str) -> dict | None:
    url = f"{_BASE_URL}{path}"
    try:
        r = await session.get(url, headers=_HEADERS, timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception as exc:
        log.debug("API error %s: %s", path, exc)
        return None


async def _fetch_events_for_season(
    session: AsyncSession,
    tournament_id: int,
    season_id: int,
    max_pages: int = 50,
) -> list[_SofaEvent]:
    """Fetch all finished events for one season, returning parsed _SofaEvent list."""
    events: list[_SofaEvent] = []
    for page in range(max_pages):
        data = await _api_get(
            session,
            f"/unique-tournament/{tournament_id}/season/{season_id}/events/last/{page}",
        )
        if not data:
            break
        raw_events = data.get("events", [])
        for ev in raw_events:
            if ev.get("status", {}).get("type") != "finished":
                continue
            ts = ev.get("startTimestamp")
            if not ts:
                continue
            ev_date = datetime.fromtimestamp(ts, tz=timezone.utc).date()
            home_raw = ev.get("homeTeam", {}).get("name", "")
            away_raw = ev.get("awayTeam", {}).get("name", "")
            if not home_raw or not away_raw:
                continue
            events.append(_SofaEvent(
                event_id   = int(ev["id"]),
                event_date = ev_date,
                home_raw   = home_raw,
                away_raw   = away_raw,
                home_norm  = _norm(home_raw),
                away_norm  = _norm(away_raw),
            ))
        if not data.get("hasNextPage", True) or not raw_events:
            break
    return events


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _load_db_rows(sf: Any, db_tournament_name: str) -> list[_DbRow]:
    """Load matches with Flashscore IDs that still need a Sofascore event ID."""
    sql = """
        SELECT m.id, m.external_id, m.scheduled_at::date AS match_date,
               ht.name AS home_team, at.name AS away_team
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE m.tournament_name = :tname
          AND m.external_id LIKE 'fs_%'
          AND m.home_score_final IS NOT NULL
        ORDER BY m.scheduled_at DESC
    """
    async with sf() as db:
        rows = (await db.execute(text(sql), {"tname": db_tournament_name})).mappings().all()
    return [
        _DbRow(
            match_id   = int(r["id"]),
            ext_id     = r["external_id"],
            match_date = r["match_date"],
            home_raw   = r["home_team"],
            away_raw   = r["away_team"],
            home_norm  = _norm(r["home_team"]),
            away_norm  = _norm(r["away_team"]),
        )
        for r in rows
    ]


async def _patch_external_id(sf: Any, match_id: int, new_ext_id: str) -> None:
    async with sf() as db:
        async with db.begin():
            await db.execute(
                text("UPDATE matches SET external_id = :eid WHERE id = :mid"),
                {"eid": new_ext_id, "mid": match_id},
            )


# ── Matching logic ────────────────────────────────────────────────────────────

def _score_pair(db_row: _DbRow, sofa_ev: _SofaEvent) -> float:
    home_sim = _sim(db_row.home_norm, sofa_ev.home_norm)
    away_sim = _sim(db_row.away_norm, sofa_ev.away_norm)
    return (home_sim + away_sim) / 2


def _find_best_match(
    db_row: _DbRow,
    events: list[_SofaEvent],
    threshold: float,
    date_window: int = 1,
) -> _SofaEvent | None:
    """Find the Sofascore event best matching this DB row."""
    candidates = [
        ev for ev in events
        if ev.event_date is not None
        and abs((ev.event_date - db_row.match_date).days) <= date_window
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda ev: _score_pair(db_row, ev))
    if _score_pair(db_row, best) >= threshold:
        return best
    return None


# ── Orchestration ─────────────────────────────────────────────────────────────

async def run(
    leagues: list[str],
    db_name_override: str | None,
    seasons_filter: list[str] | None,
    threshold: float,
    dry_run: bool,
) -> None:
    sf = get_session_factory()
    cookies = _load_cookies()

    async with AsyncSession(impersonate="chrome124", headers=_HEADERS) as session:
        if cookies:
            for c in cookies:
                session.cookies.set(c["name"], c["value"], domain=c.get("domain", ""))
            log.info("Loaded %d cookies from %s", len(cookies), _COOKIES_PATH)

        totals = {"matched": 0, "unmatched": 0, "already_numeric": 0, "conflict": 0}

        for league_key in leagues:
            if league_key not in LEAGUE_CATALOG:
                log.error(
                    "League '%s' not in LEAGUE_CATALOG — add it to catalog.py first.",
                    league_key,
                )
                continue

            cat  = LEAGUE_CATALOG[league_key]
            tid  = cat["tournament_id"]
            db_tname = db_name_override if db_name_override else league_key

            log.info("━" * 60)
            log.info("[%s] tournament_id=%d  db_filter=tournament_name='%s'",
                     league_key, tid, db_tname)

            db_rows = await _load_db_rows(sf, db_tname)
            log.info("[%s] DB rows with Flashscore IDs: %d", league_key, len(db_rows))
            if not db_rows:
                log.info("[%s] Nothing to patch.", league_key)
                continue

            # Collect all Sofascore events across requested seasons
            all_events: list[_SofaEvent] = []
            seen_event_ids: set[int] = set()
            for season in cat["seasons"]:
                if seasons_filter and season.get("code") not in seasons_filter:
                    continue
                sid   = season["id"]
                sname = season["name"]
                log.info("[%s] Fetching Sofascore events: %s (season %d)…", league_key, sname, sid)
                evs = await _fetch_events_for_season(session, tid, sid)
                new = [e for e in evs if e.event_id not in seen_event_ids]
                seen_event_ids.update(e.event_id for e in new)
                all_events.extend(new)
                log.info("[%s] %s → %d events fetched (%d unique total)", league_key, sname, len(evs), len(all_events))

            if not all_events:
                log.warning("[%s] No Sofascore events fetched — check tournament_id / season IDs.", league_key)
                continue

            matched = unmatched = conflict = 0
            unmatched_examples: list[tuple[_DbRow, str]] = []

            for db_row in db_rows:
                best = _find_best_match(db_row, all_events, threshold)
                if best is None:
                    unmatched += 1
                    # Find closest for hint
                    nearby = sorted(
                        [ev for ev in all_events
                         if ev.event_date and abs((ev.event_date - db_row.match_date).days) <= 3],
                        key=lambda ev: -_score_pair(db_row, ev),
                    )
                    hint = (
                        f"{nearby[0].home_raw} vs {nearby[0].away_raw} "
                        f"(score={_score_pair(db_row, nearby[0]):.2f})"
                        if nearby else "no candidates in ±3 days"
                    )
                    unmatched_examples.append((db_row, hint))
                    continue

                new_ext_id = str(best.event_id)
                score = _score_pair(db_row, best)
                log.debug(
                    "  MATCH %s | %s vs %s → event_id=%s (score=%.2f)",
                    db_row.match_date, db_row.home_raw, db_row.away_raw, new_ext_id, score,
                )

                if not dry_run:
                    await _patch_external_id(sf, db_row.match_id, new_ext_id)
                matched += 1

            totals["matched"]   += matched
            totals["unmatched"] += unmatched
            totals["conflict"]  += conflict
            log.info(
                "[%s] matched=%d  unmatched=%d%s",
                league_key, matched, unmatched,
                "  [DRY-RUN — no writes]" if dry_run else "",
            )
            if unmatched_examples:
                log.info("[%s] Unmatched examples (DB → closest Sofascore candidate):", league_key)
                for row, hint in unmatched_examples[:8]:
                    log.info(
                        "  %s | %s vs %s  →  %s",
                        row.match_date, row.home_raw, row.away_raw, hint,
                    )

    await dispose_engine()
    log.info("━" * 60)
    log.info(
        "TOTAL  matched=%d  unmatched=%d%s",
        totals["matched"], totals["unmatched"],
        "  [DRY-RUN]" if dry_run else "",
    )
    if totals["unmatched"] and not dry_run:
        log.info(
            "Tip: re-run with --threshold %.2f to catch more matches, "
            "or check team names in DB vs Sofascore.",
            max(0.35, threshold - 0.10),
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--leagues", nargs="+", required=True, metavar="KEY",
        help="League keys from LEAGUE_CATALOG (e.g. LegaA ABA Israel)",
    )
    p.add_argument(
        "--seasons", nargs="+", metavar="CODE",
        help="Season codes to restrict to (e.g. 2526 2425). Default: all seasons in catalog.",
    )
    p.add_argument(
        "--db-name", metavar="NAME",
        help="Override DB tournament_name filter when it differs from the catalog key "
             "(e.g. --leagues PBA_Phil --db-name PBA_PhilCup)",
    )
    p.add_argument(
        "--threshold", type=float, default=0.50, metavar="T",
        help="Minimum fuzzy similarity score to accept a match (default: 0.50).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and match but do NOT write to DB.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.db_name and len(args.leagues) > 1:
        print("ERROR: --db-name can only be used with a single --leagues entry.", file=sys.stderr)
        sys.exit(1)
    asyncio.run(run(
        leagues        = args.leagues,
        db_name_override = args.db_name,
        seasons_filter = args.seasons,
        threshold      = args.threshold,
        dry_run        = args.dry_run,
    ))
