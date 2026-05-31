#!/usr/bin/env python3
"""Grid search CLI: per-league × feature-combo backtester sweep.

Thin orchestration layer over ``src.evaluation.grid_search``. All heavy logic
(dataset prep, per-cell training, table assembly) lives in that package so it
is independently testable and each module stays under the 250-line limit.

Three target periods via ``--period``:
    game (default) — полный тотал vs. закрывающая линия букмекера
    1q             — тотал 1-й четверти vs. синтетическая медиана
    1h             — тотал 1-й половины  vs. синтетическая медиана

Usage:
    python scripts/grid_search_leagues.py
    python scripts/grid_search_leagues.py --period 1q --csv results/grid_search_1q.csv
    python scripts/grid_search_leagues.py --leagues EuroLeague NBA --period game
    python scripts/grid_search_leagues.py --min-bets 15
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.evaluation.grid_search import (
    COMBOS,
    GRID_THRESHOLDS,
    PERIOD_CONFIGS,
    Period,
    build_full_table,
    build_winner_table,
    prepare_dataset,
    render_table,
    run_cell,
)
from src.evaluation.grid_search.cell import GridCellResult
from src.evaluation.simulation import SimulationParams
from src.models.evaluation import chrono_split

logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DEFAULT_MIN_BETS: int = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--period", choices=["game", "1q", "1h"], default="game",
                   help="Целевой период (game — полный тотал, 1q, 1h).")
    p.add_argument("--leagues", nargs="*", metavar="LEAGUE", default=None,
                   help="Лиги для прогона (по умолчанию — все).")
    p.add_argument("--csv", default="", metavar="PATH",
                   help="Путь для CSV (по умолчанию results/grid_search_{period}.csv).")
    p.add_argument("--min-bets", type=int, default=_DEFAULT_MIN_BETS, metavar="N",
                   help=f"Мин. ставок на пороге для учёта ROI (default {_DEFAULT_MIN_BETS}).")
    return p.parse_args()


def _resolve_leagues(df_leagues: list[str], requested: list[str] | None) -> list[str]:
    if not requested:
        return df_leagues
    unknown = [lg for lg in requested if lg not in df_leagues]
    if unknown:
        log.warning("Лиги не найдены в датасете: %s", unknown)
    chosen = [lg for lg in requested if lg in df_leagues]
    if not chosen:
        log.error("Ни одна из запрошенных лиг не найдена. Доступные: %s", df_leagues)
        sys.exit(1)
    return chosen


def _simulation_params() -> SimulationParams:
    cfg = settings.evaluation
    return SimulationParams(
        odds            = cfg.odds,
        stake           = cfg.flat_stake,
        thresholds      = GRID_THRESHOLDS,
        bootstrap_iters = cfg.bootstrap_iters,
        bootstrap_seed  = cfg.bootstrap_seed,
        ci_alpha        = cfg.bootstrap_ci_alpha,
    )


def main() -> None:
    args = parse_args()
    period: Period = args.period
    period_cfg = PERIOD_CONFIGS[period]

    import pandas as pd  # local import keeps module import cost off the CLI help path

    df = prepare_dataset(period)
    leagues = _resolve_leagues(
        sorted(df["league"].dropna().unique().tolist()), args.leagues,
    )
    log.info("Период: %s | Лиги: %d | Комбо: %d", period_cfg.label, len(leagues), len(COMBOS))

    params = _simulation_params()
    cells: list[GridCellResult] = []
    total, done = len(leagues) * len(COMBOS), 0

    for league in leagues:
        subset = df[df["league"] == league].reset_index(drop=True)
        log.info("── Лига: %-20s строк: %d", league, len(subset))
        if subset.empty:
            log.warning("  Нет данных для лиги '%s' — пропуск.", league)
            continue

        train_df, test_dict = chrono_split(subset)
        test_df = pd.concat(test_dict.values(), ignore_index=True)

        for combo_name, enabled in COMBOS.items():
            done += 1
            log.info("  [%d/%d] %s × %s", done, total, league, combo_name)
            cell = run_cell(train_df, test_df, enabled, params, args.min_bets, period)
            if cell is None:
                log.info("  → пропущено")
                continue
            cell = dataclasses.replace(cell, league=league, combo=combo_name)
            cells.append(cell)
            best_roi = 0.0 if math.isnan(cell.best_roi) else cell.best_roi
            log.info(
                "  → LogLoss=%.4f AUC=%.4f best_roi=%+.2f%% @ %s (%d ставок)",
                cell.log_loss, cell.roc_auc, best_roi, cell.best_thr, cell.best_bets,
            )

    if not cells:
        log.error("Ни одна ячейка не дала результатов — проверьте данные и min_rows.")
        sys.exit(1)

    full_df   = build_full_table(cells, period)
    winner_df = build_winner_table(full_df)

    note = "" if period_cfg.has_real_line else " (ROI vs. синтетическая медиана)"
    log.info("%s", render_table(full_df, f"ПОЛНАЯ ТАБЛИЦА [{period_cfg.label}{note}]"))
    log.info("%s", render_table(winner_df, f"ПОБЕДИТЕЛИ ПО ЛИГАМ [{period_cfg.label}]"))

    out = Path(args.csv or f"results/grid_search_{period}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out, index=False)
    log.info("Результаты сохранены → %s", out)


if __name__ == "__main__":
    main()
