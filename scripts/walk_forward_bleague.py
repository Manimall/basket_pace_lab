#!/usr/bin/env python3
"""Walk-forward validation for B.League — is the edge systematic across seasons?

Strictly chronological, expanding-window: for each test season, train the V6
model on ALL prior seasons only, then bet the held-out season at a fixed 0.54
threshold. This is a true out-of-sample cross-season test — the +19% single-
season result could be one-season luck; surviving here (a model trained on a
DIFFERENT season still profits) is evidence of a systematic bookmaker
inefficiency.

    train[S1]        -> test[S2]
    train[S1..S2]    -> test[S3]
    ...

Strictly current working config: V6, BASE-only features (feature_selector picks
BASE for BLeague), threshold 0.54, no hyperparameter tuning. Reports per-season
metrics + a pooled aggregate ROI/CI95 over all test seasons.

Run:
    python scripts/walk_forward_bleague.py
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.evaluation.backtest_league import build_simulation_params
from src.evaluation.backtester import prepare_dataset
from src.evaluation.config import BIN_TARGET, LINE_COL
from src.evaluation.model import get_x, train_classifier
from src.evaluation.simulation import BetReport, SimulationParams, aggregate, simulate
from src.features.score_features import TARGET

logging.basicConfig(level=settings.app.log_level, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_LEAGUE:             str   = "BLeague"
_THRESHOLD:          float = 0.54
_SEASON_START_MONTH: int   = 8       # a season S spans Aug(S) .. Jul(S+1)
_HISTORY_GATE:       str   = "2000-08-01"  # widen the season gate to load all history
_MIN_TRAIN:          int   = 100
_MIN_TEST:           int   = 30
_DATE_COL:           str   = "scheduled_at"


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold: train = all prior seasons, test = one season."""

    season:  str
    n_train: int
    auc:     float
    report:  BetReport


def _season_of(dates: pd.Series) -> pd.Series:
    """Map each match date to its season start-year (Aug rollover)."""
    dt = pd.to_datetime(dates, utc=True)
    return dt.dt.year - (dt.dt.month < _SEASON_START_MONTH).astype(int)


def _run_fold(
    train_df: pd.DataFrame, test_df: pd.DataFrame, label: str, params: SimulationParams,
) -> tuple[Fold, np.ndarray, np.ndarray] | None:
    """Train on ``train_df``, bet ``test_df`` at the fixed threshold. Returns the
    fold summary plus the per-row (label, pnl) arrays for pooling."""
    model = train_classifier(train_df, target_col=BIN_TARGET, league_key=_LEAGUE)
    probs = model.predict_proba(get_x(test_df, _LEAGUE))[:, 1]
    y_te  = test_df[BIN_TARGET].astype(int).to_numpy()
    if len(np.unique(y_te)) < 2:
        log.warning("[%s] single-class test — skip", label)
        return None
    auc = roc_auc_score(y_te, probs)
    lab, pnl = simulate(
        probs, test_df[TARGET].to_numpy(), test_df[LINE_COL].to_numpy(),
        _THRESHOLD, params.odds, params.stake,
    )
    report = aggregate(lab, pnl, label, params.bootstrap_iters,
                       params.bootstrap_seed, params.ci_alpha)
    return Fold(label, len(train_df), auc, report), lab, pnl


def _print_report(folds: list[Fold], pooled: BetReport) -> None:
    log.info("\n=== B.League walk-forward (V6 BASE, threshold %.2f) ===", _THRESHOLD)
    log.info("| %-9s | %7s | %8s | %9s | %8s | %s |",
             "Test сезон", "n_train", "Test AUC", "ROI %", "n ставок", "ROI CI95")
    log.info("|%s|%s|%s|%s|%s|%s|", "-"*11, "-"*9, "-"*10, "-"*11, "-"*10, "-"*22)
    for f in folds:
        r = f.report
        log.info("| %-9s | %7d | %8.4f | %+8.2f%% | %8d | [%+6.2f%%; %+6.2f%%] |",
                 f.season, f.n_train, f.auc, r.roi, r.n, r.roi_ci_low, r.roi_ci_high)
    log.info("|%s|%s|%s|%s|%s|%s|", "-"*11, "-"*9, "-"*10, "-"*11, "-"*10, "-"*22)
    star = " ★ (CI95 > 0)" if pooled.roi_ci_low > 0 else ""
    log.info("| %-9s | %7s | %8s | %+8.2f%% | %8d | [%+6.2f%%; %+6.2f%%] |%s",
             "АГРЕГАТ", "—", "—", pooled.roi, pooled.n,
             pooled.roi_ci_low, pooled.roi_ci_high, star)


def main() -> None:
    """Run the expanding-window walk-forward and print per-season + pooled ROI."""
    settings.features.current_season_start = _HISTORY_GATE  # load all seasons
    params = build_simulation_params()

    df = prepare_dataset()
    df = df[df["league"] == _LEAGUE].reset_index(drop=True)
    if df.empty:
        log.error("Нет данных по %s (проверьте сбор/линии).", _LEAGUE)
        return
    df["_season"] = _season_of(df[_DATE_COL])
    seasons = sorted(int(s) for s in df["_season"].unique())
    log.info("[%s] сезоны: %s | всего строк с линией: %d",
             _LEAGUE, [f"{s}/{s+1}" for s in seasons], len(df))
    if len(seasons) < 2:
        log.error("Нужно ≥2 сезона для walk-forward; есть %d.", len(seasons))
        return

    folds: list[Fold] = []
    all_lab: list[np.ndarray] = []
    all_pnl: list[np.ndarray] = []
    for s in seasons[1:]:
        train_df = df[df["_season"] < s]
        test_df  = df[df["_season"] == s]
        if len(train_df) < _MIN_TRAIN or len(test_df) < _MIN_TEST:
            log.warning("[%d/%d] мало строк (train=%d test=%d) — пропуск",
                        s, s + 1, len(train_df), len(test_df))
            continue
        out = _run_fold(train_df, test_df, f"{s}/{s + 1}", params)
        if out is None:
            continue
        fold, lab, pnl = out
        folds.append(fold)
        all_lab.append(lab)
        all_pnl.append(pnl)

    if not folds:
        log.error("Нет валидных фолдов.")
        return
    pooled = aggregate(np.concatenate(all_lab), np.concatenate(all_pnl),
                       "ALL", params.bootstrap_iters, params.bootstrap_seed, params.ci_alpha)
    _print_report(folds, pooled)


if __name__ == "__main__":
    main()
