"""CatBoost training, evaluation, and result display helpers.

Used by validate_by_league to keep orchestration logic separate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import settings
from src.features.score_features import ALL_FEAT, CAT_COLS, TARGET

log = logging.getLogger(__name__)

_TABLE_BORDER_CHAR: str = "─"


@dataclass(frozen=True)
class LeagueResult:
    """One league's evaluation outcome for the results table.

    Attributes:
        league:       tournament_name of the evaluated league.
        n:            Number of test rows.
        mae:          Model mean absolute error on the test set.
        rmse:         Model root mean squared error on the test set.
        baseline_mae: MAE of the naive per-league-mean predictor.
        avg_q1:       Average Q1 total in the test set (context column).
    """

    league:       str
    n:            int
    mae:          float
    rmse:         float
    baseline_mae: float
    avg_q1:       float

    @property
    def delta_vs_baseline(self) -> float:
        """Improvement over baseline MAE (positive = model beats baseline)."""
        return self.baseline_mae - self.mae


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature-built frame into model input X and target y.

    Args:
        df: Feature-built DataFrame containing ``ALL_FEAT`` columns and TARGET.

    Returns:
        Tuple ``(X, y)`` where X holds the available feature columns (league
        cast to str) and y is the continuous TARGET series.
    """
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


def chrono_split_3way(
    df: pd.DataFrame, train_frac: float, val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by date and slice into (train, val, test) by leading fractions.

    Used by blind-threshold validation: the model trains on ``train``, the bet
    threshold is selected on ``val`` (unseen by the model), and the fixed
    threshold is applied on ``test`` — removing post-hoc threshold-selection bias.

    Args:
        df: Rows to split (mixed leagues are sorted globally by ``scheduled_at``).
        train_frac: Leading fraction used for model training.
        val_frac: Next fraction used for threshold selection; ``test`` is the
            remaining ``1 - train_frac - val_frac``.

    Returns:
        ``(train_df, val_df, test_df)`` in chronological order.
    """
    ordered = df.sort_values("scheduled_at").reset_index(drop=True)
    n = len(ordered)
    a = int(n * train_frac)
    b = int(n * (train_frac + val_frac))
    return ordered.iloc[:a], ordered.iloc[a:b], ordered.iloc[b:]


def train(train_df: pd.DataFrame) -> CatBoostRegressor:
    """Train the global CatBoost MAE regressor on game_total.

    Args:
        train_df: Training rows (rows without TARGET are dropped).

    Returns:
        A fitted ``CatBoostRegressor`` using ``settings.model`` hyperparameters.
    """
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
    """Return (MAE, RMSE) for the given predictions."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def render_results_table(rows: list[LeagueResult]) -> str:
    """Render the per-league results table as a single multi-line string.

    Args:
        rows: One LeagueResult per evaluated league.

    Returns:
        Formatted table (header, sorted rows, weighted-overall footer).
    """
    rows = sorted(rows, key=lambda r: r.mae)
    hdr = (
        f"{'League':<20s} {'N':>5s} {'MAE':>6s} {'RMSE':>6s} "
        f"{'Baseline':>8s} {'Δ vs base':>10s} {'AvgQ1':>6s}"
    )
    sep = _TABLE_BORDER_CHAR * len(hdr)
    lines: list[str] = [sep, hdr, sep]
    for r in rows:
        flag = "✓" if r.delta_vs_baseline > 0 else "✗"
        lines.append(
            f"{r.league:<20s} {r.n:>5d} "
            f"{r.mae:>6.2f} {r.rmse:>6.2f} "
            f"{r.baseline_mae:>8.2f} {r.delta_vs_baseline:>+10.2f} {flag} "
            f"{r.avg_q1:>6.1f}"
        )
    lines.append(sep)

    total_n = sum(r.n for r in rows)
    if total_n > 0:
        w_mae   = sum(r.mae  * r.n for r in rows) / total_n
        w_rmse  = sum(r.rmse * r.n for r in rows) / total_n
        w_base  = sum(r.baseline_mae * r.n for r in rows) / total_n
        w_delta = w_base - w_mae
        lines.append(
            f"{'OVERALL (weighted)':<20s} {total_n:>5d} "
            f"{w_mae:>6.2f} {w_rmse:>6.2f} "
            f"{w_base:>8.2f} {w_delta:>+10.2f} {'✓' if w_delta > 0 else '✗'}"
        )
        lines.append(sep)
    return "\n".join(lines)


def print_results_table(rows: list[LeagueResult]) -> None:
    """Log the per-league results table via the standard logging pipeline."""
    log.info("\n%s\n", render_results_table(rows))
