#!/usr/bin/env python3
"""Blind threshold validation — honest ROI without threshold-selection bias.

The per-league backtester reports the BEST threshold's ROI, picked post-hoc on
the test set (data snooping → optimistic). This probe fixes that with a 3-way
chronological split:

    [──── A: train model ────][── B: pick threshold ──][── C: blind test ──]

The model trains on A. The profit-maximising threshold is chosen on B (which the
model never saw). That fixed threshold is then applied BLINDLY on C. We print the
blind ROI next to the optimistic ROI (threshold peeked on C) — same model, same
test set C, so the gap is exactly the selection bias.

Run:
    python scripts/blind_threshold_validation.py --league BSL
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.evaluation.backtest_league import build_simulation_params
from src.evaluation.backtester import prepare_dataset
from src.evaluation.config import BIN_TARGET, LINE_COL
from src.evaluation.grid_search.cell import best_roi_from_reports
from src.evaluation.model import get_x, train_classifier
from src.evaluation.simulation import BetReport, SimulationParams, run_threshold_sweep
from src.features.score_features import TARGET
from src.models.evaluation import chrono_split_3way

logging.basicConfig(level=settings.app.log_level, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DEFAULT_LEAGUE: str = "BSL"
_MIN_BETS:       int = 10
_TRAIN_FRAC:     float = 0.60
_VAL_FRAC:       float = 0.20  # test = remaining 1 - train - val


def _sweep(
    model: CatBoostClassifier, frame: pd.DataFrame, params: SimulationParams,
) -> list[BetReport]:
    """Threshold sweep (with bootstrap CI) of a fitted model on ``frame``."""
    probs = model.predict_proba(get_x(frame))[:, 1]
    return run_threshold_sweep(
        probs, frame[TARGET].to_numpy(), frame[LINE_COL].to_numpy(), params,
    )


def _report_at(reports: list[BetReport], label: str) -> BetReport | None:
    """Find the swept BetReport for a specific threshold label (e.g. '0.54')."""
    return next((r for r in reports if r.row_label == label), None)


def _row(tag: str, r: BetReport | None) -> str:
    """Format one result line for the comparison table."""
    if r is None:
        return f"{tag:<22} {'—':>6}  (порог не дал ≥{_MIN_BETS} ставок на val)"
    return (f"{tag:<22} {r.row_label:>6} {r.n:>6} {r.winrate:>8.2f}% "
            f"{r.roi:>+8.2f}%   [{r.roi_ci_low:>+6.2f}%; {r.roi_ci_high:>+6.2f}%]")


def main() -> None:
    """Run the blind-vs-optimistic threshold comparison for one league."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default=_DEFAULT_LEAGUE)
    parser.add_argument(
        "--fixed-threshold", type=float, default=None,
        help="Pre-registered threshold (e.g. 0.54) applied blindly on test C, "
             "instead of picking it on validation.",
    )
    args   = parser.parse_args()
    params = build_simulation_params()

    df = prepare_dataset()
    df = df[df["league"] == args.league].reset_index(drop=True)
    train_df, val_df, test_df = chrono_split_3way(df, _TRAIN_FRAC, _VAL_FRAC)
    log.info("[%s] split: train=%d (model) | val=%d (порог) | test=%d (слепой)",
             args.league, len(train_df), len(val_df), len(test_df))

    model = train_classifier(train_df, target_col=BIN_TARGET)

    # Threshold picked on validation B (model never saw B) → applied blindly on C.
    val_reports  = _sweep(model, val_df,  params)
    _, blind_thr, _ = best_roi_from_reports(val_reports, _MIN_BETS)

    test_reports = _sweep(model, test_df, params)
    blind_report = _report_at(test_reports, blind_thr)

    # Optimistic: threshold peeked directly on the test set C (the biased way).
    _, opt_thr, _ = best_roi_from_reports(test_reports, _MIN_BETS)
    opt_report    = _report_at(test_reports, opt_thr)

    probs_c = model.predict_proba(get_x(test_df))[:, 1]
    y_c     = test_df[BIN_TARGET].astype(int).to_numpy()
    auc     = roc_auc_score(y_c, probs_c) if len(np.unique(y_c)) > 1 else float("nan")

    log.info("\n=== %s — слепая валидация порога (3-way chrono) ===", args.league)
    log.info("Test AUC (порого-независим, одинаков для обоих): %.4f", auc)
    log.info("%-22s %6s %6s %9s %9s   %s",
             "метод", "порог", "ставок", "winrate", "ROI", "ROI CI95")
    log.info(_row("ОПТИМИСТИЧНЫЙ (peek C)", opt_report))
    log.info(_row("ЧЕСТНЫЙ (blind, val→C)", blind_report))
    if args.fixed_threshold is not None:
        fixed_report = _report_at(test_reports, f"{args.fixed_threshold:.2f}")
        log.info(_row(f"ПРЕ-РЕГ (fixed {args.fixed_threshold:.2f})", fixed_report))
    if opt_report and blind_report:
        log.info("\nΔROI (оптимизм − честность) = %+.2f пп — это и есть selection bias.",
                 opt_report.roi - blind_report.roi)


if __name__ == "__main__":
    main()
