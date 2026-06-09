#!/usr/bin/env python3
"""Seasonality window probe — is the pace signal cleaner in the Dec–Mar window?

Deliberately holds out a calendar window (golden Dec–Mar vs anomaly Apr–Jun) as
the TEST set, trains the V6 CatBoost on the rest, and compares held-out ROC-AUC
between the two windows. Higher AUC on the golden window supports the thesis in
``docs/betting_seasonality.md``.

NOTE: this is a regime DIAGNOSTIC, not a time-honest backtest — the train side
may include matches dated AFTER the test window (future leakage). It answers
"which window is more predictable", not "what ROI a live bettor would earn".

Run:
    python scripts/seasonality_window_probe.py --league BSL
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.evaluation.backtest_league import build_simulation_params
from src.evaluation.backtester import prepare_dataset
from src.evaluation.config import BIN_TARGET, LINE_COL
from src.evaluation.grid_search.cell import best_roi_from_reports
from src.evaluation.model import get_x, train_classifier
from src.evaluation.seasonality import ANOMALY, GOLDEN_WINDOW, SeasonTier, partition_by_months
from src.evaluation.simulation import SimulationParams, run_threshold_sweep
from src.features.score_features import TARGET

logging.basicConfig(level=settings.app.log_level, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DEFAULT_LEAGUE: str = "BSL"
_MIN_BETS: int = 10


@dataclass(frozen=True)
class WindowResult:
    """Held-out metrics for one calendar window."""

    label:     str
    n_train:   int
    n_test:    int
    auc:       float
    log_loss:  float
    best_roi:  float
    best_thr:  str


def _probe_window(
    df: pd.DataFrame, league_key: str, tier: SeasonTier, params: SimulationParams,
) -> WindowResult | None:
    """Train on all-but-the-window, evaluate on the held-out ``tier`` window."""
    train_df, test_df = partition_by_months(df, tier.months)
    if test_df.empty or train_df.empty:
        log.warning("[%s] empty split (train=%d test=%d) — skip",
                    tier.label, len(train_df), len(test_df))
        return None

    model = train_classifier(train_df, target_col=BIN_TARGET, league_key=league_key)
    x_te  = get_x(test_df, league_key=league_key)
    probs = model.predict_proba(x_te)[:, 1]
    y_te  = test_df[BIN_TARGET].astype(int).to_numpy()
    if len(np.unique(y_te)) < 2:
        log.warning("[%s] single-class test window — skip", tier.label)
        return None

    reports = run_threshold_sweep(
        probs, test_df[TARGET].to_numpy(), test_df[LINE_COL].to_numpy(), params,
    )
    best_roi, best_thr, _ = best_roi_from_reports(reports, _MIN_BETS)
    return WindowResult(
        label    = tier.label,
        n_train  = len(train_df),
        n_test   = len(test_df),
        auc      = roc_auc_score(y_te, probs),
        log_loss = log_loss(y_te, probs, labels=[0, 1]),
        best_roi = 0.0 if best_roi != best_roi else best_roi,  # NaN → 0.0
        best_thr = best_thr,
    )


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default=_DEFAULT_LEAGUE,
                        help=f"tournament_name to probe (default {_DEFAULT_LEAGUE}).")
    return parser.parse_args()


def main() -> None:
    """Run the golden vs anomaly window probe and print the AUC comparison."""
    args   = _parse_cli()
    params = build_simulation_params()

    df = prepare_dataset()
    df = df[df["league"] == args.league].reset_index(drop=True)
    log.info("[%s] rows with line + target: %d", args.league, len(df))

    results = [
        r for r in (
            _probe_window(df, args.league, GOLDEN_WINDOW, params),
            _probe_window(df, args.league, ANOMALY, params),
        ) if r is not None
    ]
    if len(results) < 2:
        log.error("Need both windows to compare; got %d.", len(results))
        return

    log.info("\n=== Seasonality window probe: %s ===", args.league)
    log.info("%-9s %7s %7s %9s %10s %12s",
             "window", "train", "test", "ROC-AUC", "LogLoss", "best ROI")
    for r in results:
        log.info("%-9s %7d %7d %9.4f %10.4f %11.2f%% @ %s",
                 r.label, r.n_train, r.n_test, r.auc, r.log_loss, r.best_roi, r.best_thr)

    golden, anomaly = results[0], results[1]
    delta = golden.auc - anomaly.auc
    verdict = (
        "ПОДТВЕРЖДЕНО: сигнал чище в золотом окне"
        if delta > 0 else
        "НЕ подтверждено: золотое окно не предсказуемее аномалии"
    )
    log.info("\nΔAUC (golden − anomaly) = %+.4f → %s", delta, verdict)


if __name__ == "__main__":
    main()
