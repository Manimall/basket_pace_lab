"""Single (league × combo) grid cell: train a model and score one threshold sweep.

Owns the feature-exclusion helpers, the leak-free synthetic-line derivation for
sub-game periods, and the typed ``GridCellResult`` that replaces the previously
untyped ``dict[str, Any]`` passed between layers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from src.config import settings
from src.evaluation.config import BIN_TARGET, LINE_COL
from src.evaluation.feature_selector import (
    _ALWAYS_EXCLUDED,
    _GROUP_COLUMNS,
    FeatureGroup,
)
from src.evaluation.grid_search.periods import PERIOD_CONFIGS, Period
from src.evaluation.simulation import BetReport, SimulationParams, run_threshold_sweep
from src.features.score_features import ALL_FEAT, CAT_COLS

log = logging.getLogger(__name__)

# CatBoost binary-classifier loss/metric — fixed for the grid (not tunable here).
_LOSS_FUNCTION: str = "Logloss"
_EVAL_METRIC:   str = "AUC"
_ROUND_NDIGITS: int = 4
_ROI_NDIGITS:   int = 2
_NO_THRESHOLD:  str = "—"


@dataclass(frozen=True)
class GridCellResult:
    """Metrics for one (league × combo) cell.

    Per-threshold ROIs/bets live in dicts keyed by the threshold label
    ("0.54", …) since the threshold set is configuration-driven.
    """

    n_train:    int
    n_test:     int
    n_features: int
    log_loss:   float
    roc_auc:    float
    best_roi:   float
    best_thr:   str
    best_bets:  int
    league:     str = ""
    combo:      str = ""
    synthetic_line:    float | None = None
    roi_by_threshold:  dict[str, float] = field(default_factory=dict)
    bets_by_threshold: dict[str, int]   = field(default_factory=dict)


def excluded_for_combo(enabled: frozenset[FeatureGroup]) -> frozenset[str]:
    """Return the column-exclusion set for a given enabled-group frozenset."""
    excluded: frozenset[str] = _ALWAYS_EXCLUDED
    for group, cols in _GROUP_COLUMNS.items():
        if group not in enabled:
            excluded = excluded | cols
    return excluded


def select_features(df: pd.DataFrame, excluded: frozenset[str]) -> pd.DataFrame:
    """Project ``df`` onto the allowed feature columns, casting league to str."""
    feat_cols = [c for c in ALL_FEAT if c in df.columns and c not in excluded]
    X = df[feat_cols].copy()
    if "league" in X.columns:
        X["league"] = X["league"].astype(str)
    return X


def best_roi_from_reports(
    reports: list[BetReport], min_bets: int,
) -> tuple[float, str, int]:
    """Return (best_roi%, threshold_label, n_bets) over qualifying thresholds."""
    best_roi  = float("nan")
    best_thr  = _NO_THRESHOLD
    best_bets = 0
    for r in reports:
        if r.n < min_bets:
            continue
        if np.isnan(best_roi) or r.roi > best_roi:
            best_roi, best_thr, best_bets = r.roi, r.row_label, r.n
    rounded = round(best_roi, _ROI_NDIGITS) if not np.isnan(best_roi) else best_roi
    return rounded, best_thr, best_bets


def derive_bin_target_from_median(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Set BIN_TARGET/LINE_COL from the *training* median (leak-free synthetic line).

    Returns copies of train/test with the binary target and synthetic line set,
    plus the median value used as the line.
    """
    median_val = float(train_df[target_col].median())
    train_df = train_df.copy()
    test_df  = test_df.copy()
    for frame in (train_df, test_df):
        frame[BIN_TARGET] = (frame[target_col] > median_val).astype(int)
        frame[LINE_COL]   = median_val
    return train_df, test_df, median_val


def _build_classifier() -> CatBoostClassifier:
    cfg = settings.model
    return CatBoostClassifier(
        iterations    = cfg.catboost_iterations,
        learning_rate = cfg.catboost_lr,
        depth         = cfg.catboost_depth,
        loss_function = _LOSS_FUNCTION,
        eval_metric   = _EVAL_METRIC,
        cat_features  = CAT_COLS,
        random_seed   = cfg.catboost_seed,
        verbose       = 0,
    )


def run_cell(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    enabled:  frozenset[FeatureGroup],
    params:   SimulationParams,
    min_bets: int,
    period:   Period,
) -> GridCellResult | None:
    """Train + evaluate one (league × combo) cell. Returns None when skipped.

    Defensive: any malformed slice (all-NaN target, single-class label, empty
    feature matrix) is logged at WARNING and skipped; unexpected exceptions are
    caught, logged, and also yield None so one bad cell never aborts the sweep.
    """
    guards = settings.evaluation
    if len(train_df) < guards.min_train_rows or len(test_df) < guards.min_test_rows:
        log.warning(
            "  Пропуск: train=%d (min %d) / test=%d (min %d)",
            len(train_df), guards.min_train_rows, len(test_df), guards.min_test_rows,
        )
        return None

    try:
        cfg = PERIOD_CONFIGS[period]

        synthetic_line: float | None = None
        if not cfg.has_real_line:
            if train_df[cfg.target_col].isna().all():
                log.warning("  Пропуск: все значения %s — NaN.", cfg.target_col)
                return None
            train_df, test_df, synthetic_line = derive_bin_target_from_median(
                train_df, test_df, cfg.target_col,
            )

        if BIN_TARGET not in train_df.columns or train_df[BIN_TARGET].nunique() < 2:
            log.warning("  Пропуск: целевая переменная однородна (один класс).")
            return None

        excluded = excluded_for_combo(enabled)
        X_tr = select_features(train_df, excluded)
        X_te = select_features(test_df,  excluded)
        if X_tr.empty or X_te.empty or X_tr.shape[1] == 0:
            log.warning("  Пропуск: пустая матрица фичей после исключений.")
            return None

        model = _build_classifier()
        model.fit(X_tr, train_df[BIN_TARGET].astype(int).to_numpy())

        probs   = model.predict_proba(X_te)[:, 1]
        y_te    = test_df[BIN_TARGET].astype(int).to_numpy()
        actuals = test_df[cfg.target_col].to_numpy()
        lines   = test_df[LINE_COL].to_numpy()

        ll  = log_loss(y_te, probs, labels=[0, 1])
        auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else float("nan")
        reports = run_threshold_sweep(probs, actuals, lines, params)

        roi_by_thr  = {r.row_label: (round(r.roi, _ROI_NDIGITS) if r.n >= min_bets else float("nan"))
                       for r in reports}
        bets_by_thr = {r.row_label: r.n for r in reports}
        best_roi, best_thr, best_bets = best_roi_from_reports(reports, min_bets)

        return GridCellResult(
            n_train           = len(train_df),
            n_test            = len(test_df),
            n_features        = X_tr.shape[1],
            log_loss          = round(ll, _ROUND_NDIGITS),
            roc_auc           = round(auc, _ROUND_NDIGITS),
            best_roi          = best_roi,
            best_thr          = best_thr,
            best_bets         = best_bets,
            synthetic_line    = round(synthetic_line, _ROI_NDIGITS) if synthetic_line is not None else None,
            roi_by_threshold  = roi_by_thr,
            bets_by_threshold = bets_by_thr,
        )

    except Exception as exc:  # noqa: BLE001 — defensive: one cell must not abort the run
        log.error("  Ячейка завершилась с ошибкой: %s", exc, exc_info=True)
        return None
