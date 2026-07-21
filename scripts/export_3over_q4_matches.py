#!/usr/bin/env python3
"""Export the "3 quarters Over → Q4 result" match list to CSV.

Lists every match where Q1, Q2 and Q3 each went Over the proxy quarter line
(total_line / 4), with per-quarter combined points and the Q4 outcome (ТМ/ТБ
against the same proxy line). This is the raw evidence behind
docs/results_q4_under_after_3_overs.md — one row per qualifying match so the
list can be eyeballed or sorted outside the DB.

Proxy note: the DB has no real per-quarter line, so "Over" is measured against
total_line / 4 (see the results doc for why that inflates Q4-Under).

Run:
    python scripts/export_3over_q4_matches.py --league CBA
    python scripts/export_3over_q4_matches.py --league CBA --out results/x.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402

logging.basicConfig(level=settings.app.log_level, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DEFAULT_OUT = "results/cba_3over_q4_matches.csv"

_SQL = """
    SELECT
        m.id            AS match_id,
        m.scheduled_at::date AS date,
        th.name         AS home,
        ta.name         AS away,
        m.total_line,
        qs.period_number,
        (qs.home_score + qs.away_score) AS combined
    FROM matches m
    JOIN quarter_stats qs ON qs.match_id = m.id
    JOIN teams th ON th.id = m.home_team_id
    JOIN teams ta ON ta.id = m.away_team_id
    WHERE m.tournament_name = :league
      AND m.has_quarter_breakdown = TRUE
      AND m.total_line IS NOT NULL
      AND qs.period_type::text = 'QUARTER'
      AND qs.period_number IN (1, 2, 3, 4)
      AND qs.q4_includes_ot_points = FALSE
    ORDER BY m.scheduled_at
"""


def build_pattern_frame(league: str) -> pd.DataFrame:
    """Return one row per match where Q1, Q2, Q3 all cleared total_line/4.

    Columns: date, home, away, total_line, quarter_line (=total_line/4),
    q1..q4 combined points, q4_result (ТМ/ТБ vs quarter_line).
    """
    engine = create_engine(settings.db.sync_dsn)
    long_df = pd.read_sql_query(text(_SQL), engine, params={"league": league})
    engine.dispose()
    if long_df.empty:
        return long_df

    wide = long_df.pivot_table(
        index=["match_id", "date", "home", "away", "total_line"],
        columns="period_number", values="combined", aggfunc="first",
    ).dropna(subset=[1, 2, 3, 4]).reset_index()
    wide = wide.rename(columns={1: "q1", 2: "q2", 3: "q3", 4: "q4"})

    wide["quarter_line"] = (wide["total_line"] / 4.0).round(2)
    over3 = (
        (wide["q1"] > wide["quarter_line"])
        & (wide["q2"] > wide["quarter_line"])
        & (wide["q3"] > wide["quarter_line"])
    )
    out = wide[over3].copy()
    out["q4_result"] = out["q4"].lt(out["quarter_line"]).map({True: "ТМ", False: "ТБ"})
    cols = ["date", "home", "away", "total_line", "quarter_line",
            "q1", "q2", "q3", "q4", "q4_result"]
    return out[cols].sort_values("date").reset_index(drop=True)


def main() -> None:
    """Export the qualifying-match list to CSV and log a one-line summary."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--league", default="CBA")
    p.add_argument("--out", default=_DEFAULT_OUT)
    args = p.parse_args()

    df = build_pattern_frame(args.league)
    if df.empty:
        log.error("Нет матчей для league=%r.", args.league)
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    tm = int((df["q4_result"] == "ТМ").sum())
    log.info("[%s] матчей '3 четверти ТБ': %d | Q4 ТМ: %d (%.1f%%) | Q4 ТБ: %d",
             args.league, len(df), tm, tm / len(df) * 100, len(df) - tm)
    log.info("CSV → %s", out_path)


if __name__ == "__main__":
    main()
