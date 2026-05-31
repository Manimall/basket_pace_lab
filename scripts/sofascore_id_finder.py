#!/usr/bin/env python3
"""CLI: patch external_id for Flashscore-sourced matches with Sofascore IDs.

Thin wrapper over ``src.data_collection.sofascore.id_finder``. All logic lives
in that package (client / repository / matcher / runner) so each module stays
under the 250-line limit and the pure matcher is unit-tested.

Problem
-------
Matches collected via Flashscore have ``external_id = 'fs_XXXXXXXX'``; the Go
scout's filter (``external_id ~ '^[0-9]+$'``) skips them. This tool fuzzy-matches
each to its Sofascore event (date ±1 day + team-name similarity) and rewrites
``external_id`` to the numeric event ID.

Usage:
    python scripts/sofascore_id_finder.py --leagues LegaA --dry-run
    python scripts/sofascore_id_finder.py --leagues LegaA --seasons 2526
    python scripts/sofascore_id_finder.py --leagues ABA Israel LegaA
    python scripts/sofascore_id_finder.py --leagues PBA_Phil --db-name PBA_PhilCup
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_collection.sofascore.id_finder import DEFAULT_THRESHOLD, FinderOptions, run

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--leagues", nargs="+", required=True, metavar="KEY",
                   help="League keys from LEAGUE_CATALOG (e.g. LegaA ABA Israel).")
    p.add_argument("--seasons", nargs="+", metavar="CODE",
                   help="Season codes to restrict to (e.g. 2526 2425). Default: all.")
    p.add_argument("--db-name", metavar="NAME",
                   help="Override DB tournament_name when it differs from the catalog key.")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, metavar="T",
                   help=f"Min fuzzy similarity to accept a match (default {DEFAULT_THRESHOLD}).")
    p.add_argument("--dry-run", action="store_true", help="Match but do NOT write to DB.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.db_name and len(args.leagues) > 1:
        log.error("--db-name can only be used with a single --leagues entry.")
        sys.exit(1)
    asyncio.run(run(FinderOptions(
        leagues          = args.leagues,
        db_name_override = args.db_name,
        seasons_filter   = args.seasons,
        threshold        = args.threshold,
        dry_run          = args.dry_run,
    )))


if __name__ == "__main__":
    main()
