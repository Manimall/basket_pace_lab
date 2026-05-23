"""
CatBoost training, evaluation, and result display helpers.

Used by validate_by_league to keep orchestration logic separate.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import settings
from src.features.score_features import ALL_FEAT, CAT_COLS, TARGET

log = logging.getLogger(__name__)


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feat_cols = [c for c in ALL_FEAT if c in df.columns]
    X = df[feat_cols].copy()
    X["league"] = X["league"].astype(str)
    return X, df[TARGET]


def chrono_split(
    df: pd.DataFrame,
    test_frac: float | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Last `test_frac` of each league (by date) = test; rest = train."""
    frac = test_frac if test_frac is not None else settings.model.test_frac
    test_parts: dict[str, pd.DataFrame] = {}
    train_idx: list[int] = []
    for league, grp in df.groupby("league", sort=False):
        grp_sorted = grp.sort_values("scheduled_at")
        n_test = max(1, int(len(grp_sorted) * frac))
        test_parts[str(league)] = df.loc[grp_sorted.index[-n_test:]]
        train_idx.extend(grp_sorted.index[:-n_test].tolist())
    return df.loc[train_idx].sort_values("scheduled_at"), test_parts


def train(train_df: pd.DataFrame) -> CatBoostRegressor:
    train_df = train_df.dropna(subset=[TARGET])
    X_tr, y_tr = get_xy(train_df)
    cfg = settings.model
    model = CatBoostRegressor(
        iterations=cfg.catboost_iterations,
        learning_rate=cfg.catboost_lr,
        depth=cfg.catboost_depth,
        loss_function="MAE",
        eval_metric="MAE",
        cat_features=CAT_COLS,
        random_seed=cfg.catboost_seed,
        verbose=0,
    )
    model.fit(X_tr, y_tr)
    log.info("CatBoost training complete.")
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return mae, rmse


def print_results_table(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["mae"])
    hdr = (
        f"{'League':<20s} {'N':>5s} {'MAE':>6s} {'RMSE':>6s} "
        f"{'Baseline':>8s} {'Δ vs base':>10s} {'AvgQ1':>6s}"
    )
    sep = "─" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in rows:
        delta = r["baseline_mae"] - r["mae"]
        flag  = "✓" if delta > 0 else "✗"
        print(
            f"{r['league']:<20s} {r['n']:>5d} "
            f"{r['mae']:>6.2f} {r['rmse']:>6.2f} "
            f"{r['baseline_mae']:>8.2f} {delta:>+10.2f} {flag} "
            f"{r['avg_q1']:>6.1f}"
        )
    print(sep)
    total_n = sum(r["n"] for r in rows)
    w_mae   = sum(r["mae"]  * r["n"] for r in rows) / total_n
    w_rmse  = sum(r["rmse"] * r["n"] for r in rows) / total_n
    w_base  = sum(r["baseline_mae"] * r["n"] for r in rows) / total_n
    w_delta = w_base - w_mae
    print(
        f"{'OVERALL (weighted)':<20s} {total_n:>5d} "
        f"{w_mae:>6.2f} {w_rmse:>6.2f} "
        f"{w_base:>8.2f} {w_delta:>+10.2f} {'✓' if w_delta > 0 else '✗'}"
    )
    print(sep + "\n")
