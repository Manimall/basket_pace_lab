#!/usr/bin/env python3
"""Backtest: Q4 Under after three straight quarter Overs (proxy-line method).

Hypothesis (Maksim, 2026-07-21): after Q1, Q2 and Q3 all go Over their quarter
line, Q4 regresses → Under has an edge.

The DB has NO real per-quarter line (only ``matches.total_line``). We use a
proxy, per Gemini's spec:

    quarter line proxy  = total_line / 4                     (Q1-Q3 Over test)
    Q4 live line proxy  = total_line / 4 + shift             (book skews up on a
                                                              hot game)

CRITICAL honesty guards (this is why the script does more than Gemini asked):
  * ``shift`` is NOT one magic number. We SWEEP it and print how ROI moves with
    it — an "edge" that only exists at one hand-picked shift is not an edge.
  * We always print the UNCONDITIONAL P(Q4 Under) next to the conditional one.
    Q4 is already depressed (see docs/experiment_q4_depression.md), so the only
    real signal is the LIFT: conditional − unconditional.
  * ``--cross-season`` splits by season so you see whether the lift survives
    out of sample (see docs/nba mirage lesson).

Run (in the project env / docker):
    python scripts/test_4q_pattern.py --league CBA
    python scripts/test_4q_pattern.py --league CBA --margin 10 18 --fatigue
    python scripts/test_4q_pattern.py --league CBA --cross-season
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402

logging.basicConfig(level=settings.app.log_level, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Shift sweep (points added to the Q1-Q3 proxy line to model the book's upward
# live adjustment on a hot game). Swept, never a single magic value.
_SHIFT_MIN:  float = 0.0
_SHIFT_MAX:  float = 4.0
_SHIFT_STEP: float = 0.5
_MIN_BETS:   int   = 20     # below this, samples are too small to trust

# One row per finished match with a full quarter breakdown. total_line is the
# pre-match closing O/U (leak-free). q4_includes_ot_points rows are excluded so
# Q4 is regulation-only.
_SQL: str = """
    SELECT
        m.id            AS match_id,
        m.scheduled_at,
        m.season,
        COALESCE(m.tournament_name, 'NBA') AS league,
        m.total_line,
        qs.period_number,
        qs.home_score,
        qs.away_score
    FROM matches m
    JOIN quarter_stats qs ON qs.match_id = m.id
    WHERE m.home_score_final IS NOT NULL
      AND m.away_score_final IS NOT NULL
      AND m.has_quarter_breakdown = TRUE
      AND m.total_line IS NOT NULL
      AND qs.period_type::text = 'QUARTER'
      AND qs.period_number IN (1, 2, 3, 4)
      AND qs.q4_includes_ot_points = FALSE
    ORDER BY m.scheduled_at, qs.period_number
"""


def load_matches(league: str | None) -> pd.DataFrame:
    """Load one row per match: total_line, per-quarter combined points, margin.

    Returns a frame with q1..q4 combined points and ``margin_before_q4``
    (absolute cumulative score gap entering Q4). Matches missing any of the
    four regulation quarters are dropped.
    """
    engine = create_engine(settings.db.sync_dsn)
    long_df = pd.read_sql_query(text(_SQL), engine)
    engine.dispose()
    if league:
        long_df = long_df[long_df["league"] == league]
    if long_df.empty:
        return long_df

    long_df["combined"] = long_df["home_score"] + long_df["away_score"]
    combined = long_df.pivot_table(
        index=["match_id", "scheduled_at", "season", "league", "total_line"],
        columns="period_number", values="combined", aggfunc="first",
    )
    home = long_df.pivot_table(index="match_id", columns="period_number",
                               values="home_score", aggfunc="first")
    away = long_df.pivot_table(index="match_id", columns="period_number",
                               values="away_score", aggfunc="first")
    combined = combined.dropna(subset=[1, 2, 3, 4]).reset_index()
    combined = combined.rename(columns={1: "q1", 2: "q2", 3: "q3", 4: "q4"})

    # Absolute score gap entering Q4 = |sum(home Q1-3) − sum(away Q1-3)|.
    home_cum = home[[1, 2, 3]].sum(axis=1)
    away_cum = away[[1, 2, 3]].sum(axis=1)
    margin = (home_cum - away_cum).abs().rename("margin_before_q4")
    out = combined.merge(margin, left_on="match_id", right_index=True, how="left")
    return out.sort_values("scheduled_at").reset_index(drop=True)


def apply_filters(
    df: pd.DataFrame, margin: tuple[float, float] | None, fatigue: bool,
) -> pd.DataFrame:
    """Apply optional score-corridor and fatigue (Q3 < Q1) filters."""
    out = df
    if margin is not None:
        lo, hi = margin
        out = out[(out["margin_before_q4"] >= lo) & (out["margin_before_q4"] <= hi)]
    if fatigue:
        out = out[out["q3"] < out["q1"]]
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class PatternResult:
    """One shift's backtest outcome for the 3-Over → Q4-Under pattern."""
    shift:      float
    n_bets:     int
    win_rate:   float
    roi:        float
    roi_ci_low: float
    roi_ci_high: float
    base_under: float   # unconditional P(Q4 Under) at the same Q4 line
    lift:       float   # conditional win_rate − base_under


def _bootstrap_roi_ci(profit: np.ndarray) -> tuple[float, float]:
    """Bootstrap 95% CI for ROI (%) by resampling per-bet profit units."""
    rng = np.random.default_rng(settings.evaluation.bootstrap_seed)
    n = len(profit)
    if n == 0:
        return (float("nan"), float("nan"))
    rois = [
        profit[rng.integers(0, n, n)].mean() * 100.0
        for _ in range(settings.evaluation.bootstrap_iters)
    ]
    alpha = settings.evaluation.bootstrap_ci_alpha
    return (float(np.percentile(rois, 100 * alpha / 2)),
            float(np.percentile(rois, 100 * (1 - alpha / 2))))


def _profit_units(under_win: pd.Series, push: pd.Series, odds: float) -> np.ndarray:
    """Per-bet profit in stake units: win=+(odds−1), push=0, loss=−1."""
    profit = np.where(under_win, odds - 1.0, -1.0)
    profit = np.where(push, 0.0, profit)
    return profit.astype(float)


def evaluate_shift(df: pd.DataFrame, shift: float, odds: float) -> PatternResult:
    """Backtest Q4 Under after 3 Overs at one shift, with base-rate + lift."""
    base_q = df["total_line"] / 4.0
    line_q4 = base_q + shift

    over_123 = (df["q1"] > base_q) & (df["q2"] > base_q) & (df["q3"] > base_q)
    q4_under = df["q4"] < line_q4
    q4_push = df["q4"] == line_q4

    # Unconditional Q4 Under rate at the SAME Q4 line — the honest baseline.
    base_under = float(q4_under.mean()) if len(df) else float("nan")

    bets = df[over_123]
    n = int(len(bets))
    if n == 0:
        return PatternResult(shift, 0, float("nan"), float("nan"),
                             float("nan"), float("nan"), base_under, float("nan"))
    win = q4_under[over_123]
    push = q4_push[over_123]
    profit = _profit_units(win, push, odds)
    win_rate = float(win.mean()) * 100.0
    roi = float(profit.mean()) * 100.0
    ci_low, ci_high = _bootstrap_roi_ci(profit)
    return PatternResult(shift, n, win_rate, roi, ci_low, ci_high,
                         base_under * 100.0, win_rate - base_under * 100.0)


def sweep_shifts(df: pd.DataFrame, odds: float) -> list[PatternResult]:
    """Evaluate the pattern across the full shift sweep."""
    shifts = np.arange(_SHIFT_MIN, _SHIFT_MAX + 1e-9, _SHIFT_STEP)
    return [evaluate_shift(df, float(s), odds) for s in shifts]


def _print_table(title: str, results: list[PatternResult]) -> None:
    """Log one shift-sweep table."""
    log.info("\n=== %s ===", title)
    log.info("%-6s %6s %9s %9s %19s %10s %8s",
             "shift", "n_bets", "win%", "ROI%", "ROI CI95", "base_U%", "lift_pp")
    for r in results:
        if r.n_bets < _MIN_BETS:
            log.info("%-6.1f %6d   (n<%d — недостаточно ставок)",
                     r.shift, r.n_bets, _MIN_BETS)
            continue
        log.info("%-6.1f %6d %8.1f%% %+8.2f%% [%+6.2f%%; %+6.2f%%] %9.1f%% %+7.1f",
                 r.shift, r.n_bets, r.win_rate, r.roi,
                 r.roi_ci_low, r.roi_ci_high, r.base_under, r.lift)


def main() -> None:
    """Run the 3-Over → Q4-Under backtest with honesty guards."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--league", default="CBA", help="tournament_name filter ('' = all)")
    p.add_argument("--odds", type=float, default=settings.evaluation.odds)
    p.add_argument("--margin", nargs=2, type=float, metavar=("LO", "HI"),
                   default=None, help="score-gap corridor entering Q4 (e.g. 10 18)")
    p.add_argument("--fatigue", action="store_true",
                   help="keep only games where Q3 points < Q1 points (fatigue proxy)")
    p.add_argument("--cross-season", action="store_true",
                   help="report the sweep per season (out-of-sample robustness)")
    args = p.parse_args()

    raw = load_matches(args.league or None)
    if raw.empty:
        log.error("Нет матчей для league=%r. Проверь tournament_name в БД.", args.league)
        return
    df = apply_filters(raw, tuple(args.margin) if args.margin else None, args.fatigue)
    log.info("league=%s | матчей после фильтров: %d (из %d)",
             args.league or "ALL", len(df), len(raw))
    log.info("Фильтры: margin=%s, fatigue=%s | odds=%.2f",
             args.margin, args.fatigue, args.odds)

    if args.cross_season:
        for season, chunk in df.groupby("season"):
            _print_table(f"season {season} (n={len(chunk)})",
                         sweep_shifts(chunk, args.odds))
    _print_table(f"ALL SEASONS (n={len(df)})", sweep_shifts(df, args.odds))
    log.info("\nЧитать так: эдж есть ТОЛЬКО если lift_pp > 0 стабильно по сдвигам "
             "и сезонам, а ROI CI95 не накрывает ноль. Иначе это базовая просадка Q4.")


if __name__ == "__main__":
    main()
