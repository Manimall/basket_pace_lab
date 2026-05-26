"""Model training and feature selection for the backtester.

Single responsibility: produce a trained CatBoostClassifier and a matching
feature matrix for prediction. Bet logic, simulation, and orchestration live
elsewhere.

The invariant established in V4 (see docs/backtester_evolution.md) — line-
derived features are **always** hidden from the model — is encoded as the
module-level ``EXCLUDED_FEATURES`` frozenset.
"""
from __future__ import annotations

import logging

import pandas as pd
from catboost import CatBoostClassifier

from src.config import settings
from src.features.score_features import ALL_FEAT, BM_COLS, CAT_COLS

log = logging.getLogger(__name__)

# Line-derived columns hidden from the model. Permanent invariant since V4 —
# including any of these in X causes anchor-on-market collapse (V3 disaster).
EXCLUDED_FEATURES: frozenset[str] = frozenset(BM_COLS)


def get_x(df: pd.DataFrame) -> pd.DataFrame:
    """Select model-input columns from a feature-built DataFrame.

    Drops everything in ``EXCLUDED_FEATURES`` and coerces categorical columns
    to ``str`` for CatBoost.

    Args:
        df: A DataFrame produced by
            ``src.features.score_features.build_features``.

    Returns:
        A copy of the input restricted to allowed feature columns, with
        categorical columns cast to string.
    """
    feat_cols = [c for c in ALL_FEAT if c in df.columns and c not in EXCLUDED_FEATURES]
    X = df[feat_cols].copy()
    X["league"] = X["league"].astype(str)
    return X


def train_classifier(train_df: pd.DataFrame, target_col: str) -> CatBoostClassifier:
    """Train a CatBoostClassifier with the project's standard hyperparameters.

    Hyperparameters come from ``settings.model`` (iterations / lr / depth /
    seed). Loss is Logloss, eval metric is AUC — matched to a binary
    classification task on ``target_col``.

    Args:
        train_df: Training rows with both the feature columns and the binary
            target column.
        target_col: Name of the binary target column (e.g. ``"over_hit"``).

    Returns:
        A fitted ``CatBoostClassifier``.
    """
    X_tr = get_x(train_df)
    y_tr = train_df[target_col].astype(int).to_numpy()
    cfg  = settings.model
    model = CatBoostClassifier(
        iterations    = cfg.catboost_iterations,
        learning_rate = cfg.catboost_lr,
        depth         = cfg.catboost_depth,
        loss_function = "Logloss",
        eval_metric   = "AUC",
        cat_features  = CAT_COLS,
        random_seed   = cfg.catboost_seed,
        verbose       = 0,
    )
    model.fit(X_tr, y_tr)
    return model
