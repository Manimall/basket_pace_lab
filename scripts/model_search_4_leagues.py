#!/usr/bin/env python3
"""Model search — 3 CatBoost configs × 4 high-pace leagues at a fixed threshold.

For each of KBL / WNBA / LNB_DR / BSN, train three hyperparameter configs on the
same honest 80/20 chronological split and score them at the SAME pre-registered
threshold (0.54), so the comparison is apple-to-apple:

    Baseline (V6)  — the frozen settings.model params.
    Light          — depth=3, iterations=500 (fight variance with a shallow model).
    Conservative   — l2_leaf_reg=10, learning_rate=0.01 (strong reg, slow learning).

Answers: does any config turn a positive ROI, and if not, does any cut the loss
vs the V6 baseline? Regime diagnostic, not a bankroll claim (tiny per-league n).

Run:
    python scripts/model_search_4_leagues.py
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.evaluation.backtest_league import build_simulation_params
from src.evaluation.backtester import prepare_dataset
from src.evaluation.config import BIN_TARGET, LINE_COL
from src.evaluation.model import ModelHyperparams, default_hyperparams, get_x, train_classifier
from src.evaluation.simulation import BetReport, SimulationParams, run_threshold_sweep
from src.features.score_features import TARGET
from src.models.evaluation import chrono_split

logging.basicConfig(level=settings.app.log_level, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_LEAGUES:         tuple[str, ...] = ("KBL", "WNBA", "LNB_DR", "BSN")
_FIXED_THRESHOLD: str = "0.54"
# Light-model knobs (variance control); Conservative-model knobs (regularisation).
_LIGHT_DEPTH:      int   = 3
_LIGHT_ITERATIONS: int   = 500
_CONS_L2:          float = 10.0
_CONS_LR:          float = 0.01


@dataclass(frozen=True)
class SearchResult:
    """One (league × config) cell, scored at the fixed threshold."""

    league:  str
    config:  str
    auc:     float
    roi:     float
    n_bets:  int
    ci_low:  float
    ci_high: float


def _configs() -> tuple[tuple[str, ModelHyperparams], ...]:
    """The three hyperparameter configurations, derived from the V6 baseline."""
    base = default_hyperparams()
    return (
        ("Baseline (V6)", base),
        ("Light",         replace(base, depth=_LIGHT_DEPTH, iterations=_LIGHT_ITERATIONS)),
        ("Conservative",  replace(base, l2_leaf_reg=_CONS_L2, learning_rate=_CONS_LR)),
    )


def _report_at(reports: list[BetReport], label: str) -> BetReport | None:
    return next((r for r in reports if r.row_label == label), None)


def _evaluate(
    league: str, name: str, hp: ModelHyperparams,
    train_df: pd.DataFrame, test_df: pd.DataFrame, params: SimulationParams,
) -> SearchResult | None:
    """Train one config on ``train_df`` and score it on ``test_df`` at 0.54."""
    model = train_classifier(train_df, target_col=BIN_TARGET, hyperparams=hp)
    probs = model.predict_proba(get_x(test_df))[:, 1]
    y_te  = test_df[BIN_TARGET].astype(int).to_numpy()
    if len(np.unique(y_te)) < 2:
        log.warning("[%s/%s] single-class test — skip", league, name)
        return None
    auc = roc_auc_score(y_te, probs)
    reports = run_threshold_sweep(
        probs, test_df[TARGET].to_numpy(), test_df[LINE_COL].to_numpy(), params,
    )
    r = _report_at(reports, _FIXED_THRESHOLD)
    if r is None:
        return None
    return SearchResult(league, name, auc, r.roi, r.n, r.roi_ci_low, r.roi_ci_high)


def _print_table(results: list[SearchResult]) -> None:
    log.info("\n=== Model Search — fixed threshold %s ===", _FIXED_THRESHOLD)
    log.info("| %-8s | %-14s | %8s | %9s | %7s | %s |",
             "Лига", "Конфигурация", "Test AUC", "ROI %", "n_ставок", "ROI CI95")
    log.info("|%s|%s|%s|%s|%s|%s|", "-"*10, "-"*16, "-"*10, "-"*11, "-"*9, "-"*22)
    for r in results:
        log.info("| %-8s | %-14s | %8.4f | %+8.2f%% | %7d | [%+6.2f%%; %+6.2f%%] |",
                 r.league, r.config, r.auc, r.roi, r.n_bets, r.ci_low, r.ci_high)


def _analyse(results: list[SearchResult]) -> None:
    """Report any positive-ROI cell, else the best loss-reducer vs V6 per league."""
    log.info("\n=== Анализ ===")
    positive = sorted((r for r in results if r.roi > 0), key=lambda x: -x.roi)
    if positive:
        log.info("Плюсовой ROI найден в %d ячейках:", len(positive))
        for r in positive:
            log.info("  %s / %s: ROI=%+.2f%% (AUC=%.3f, n=%d, CI=[%+.2f%%;%+.2f%%])",
                     r.league, r.config, r.roi, r.auc, r.n_bets, r.ci_low, r.ci_high)
    else:
        log.info("Плюсового ROI нет ни в одной ячейке.")

    log.info("Лучшая (наименее убыточная) конфигурация на лигу vs V6-Baseline:")
    by_league: dict[str, list[SearchResult]] = {}
    for r in results:
        by_league.setdefault(r.league, []).append(r)
    for lg, cells in by_league.items():
        base = next((c for c in cells if c.config.startswith("Baseline")), None)
        best = max(cells, key=lambda c: c.roi)
        if base is None:
            log.info("  %-8s лучший: %-14s ROI=%+.2f%% (нет baseline)", lg, best.config, best.roi)
            continue
        delta = best.roi - base.roi
        flag = "✅ лучше V6" if best.config != base.config and delta > 0 else "= V6"
        log.info("  %-8s лучший: %-14s ROI=%+.2f%% (%+.2f пп vs V6) %s",
                 lg, best.config, best.roi, delta, flag)


def main() -> None:
    """Run the 3×4 model search and print the comparison table + analysis."""
    params = build_simulation_params()
    df = prepare_dataset()

    results: list[SearchResult] = []
    for league in _LEAGUES:
        subset = df[df["league"] == league].reset_index(drop=True)
        if subset.empty:
            log.warning("[%s] нет данных — пропуск", league)
            continue
        train_df, test_dict = chrono_split(subset)
        test_df = pd.concat(test_dict.values(), ignore_index=True)
        log.info("[%s] train=%d test=%d", league, len(train_df), len(test_df))
        for name, hp in _configs():
            res = _evaluate(league, name, hp, train_df, test_df, params)
            if res is not None:
                results.append(res)

    if not results:
        log.error("Нет результатов — проверьте данные/линии в БД.")
        return
    _print_table(results)
    _analyse(results)


if __name__ == "__main__":
    main()
