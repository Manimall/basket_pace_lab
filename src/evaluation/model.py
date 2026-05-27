"""Model training and feature selection for the backtester.

Single responsibility: produce a trained CatBoostClassifier and a matching
feature matrix for prediction. Bet logic, simulation, and orchestration live
elsewhere.

Two invariants drive feature exclusion:

1. **V4 invariant** — line-derived features (``BM_COLS``) are always hidden
   from the model. Including them collapses the model into a noise-fitting
   mirror of the bookmaker line (V3 disaster).

2. **Per-league strategy** — fatigue features are dropped for sparsely-
   scheduled leagues to avoid signal-shape-mismatch (V7 finding). Resolved
   via ``src.evaluation.feature_selector.get_excluded_features(league_key)``.
"""
from __future__ import annotations

import logging

import pandas as pd
from catboost import CatBoostClassifier

from src.config import settings
from src.evaluation.feature_selector import get_excluded_features
from src.features.score_features import ALL_FEAT, BM_COLS, CAT_COLS

log = logging.getLogger(__name__)

# Backwards-compatible alias: callers that don't pass ``league_key`` to
# ``get_x`` get the bare V4 invariant (BM_COLS only hidden, fatigue stays).
# New code should pass ``league_key`` to engage the per-league selector.
EXCLUDED_FEATURES: frozenset[str] = frozenset(BM_COLS)


def get_x(df: pd.DataFrame, league_key: str | None = None) -> pd.DataFrame:
    """Select model-input columns from a feature-built DataFrame.

    Resolves the per-league exclusion set via ``feature_selector`` and drops
    those columns from the model input. Categorical columns are coerced to
    ``str`` for CatBoost.

    Args:
        df: A DataFrame produced by
            ``src.features.score_features.build_features``.
        league_key: League identifier (matches ``tournament_name``) to drive
            per-league feature selection. ``None`` falls back to the V4
            baseline (only ``BM_COLS`` hidden) — used for backwards
            compatibility and mixed-league pipelines.

    Returns:
        A copy of the input restricted to allowed feature columns, with
        categorical columns cast to string.
    """
    # Always go through the selector — including the ``None`` case so that
    # mixed-league pipelines (e.g. OTHER control group) get the default-fallback
    # exclusions instead of the bare V4 invariant. ``EXCLUDED_FEATURES`` is kept
    # as a public alias of the V4 baseline for external consumers.
    excluded = get_excluded_features(league_key)
    feat_cols = [c for c in ALL_FEAT if c in df.columns and c not in excluded]
    X = df[feat_cols].copy()
    X["league"] = X["league"].astype(str)
    return X


def train_classifier(
    train_df:  pd.DataFrame,
    target_col: str,
    league_key: str | None = None,
) -> CatBoostClassifier:
    """Train a CatBoostClassifier with the project's standard hyperparameters.

    Hyperparameters come from ``settings.model`` (iterations / lr / depth /
    seed). Loss is Logloss, eval metric is AUC — matched to a binary
    classification task on ``target_col``.

    Args:
        train_df: Training rows with both the feature columns and the binary
            target column.
        target_col: Name of the binary target column (e.g. ``"over_hit"``).
        league_key: Forwarded to ``get_x`` for per-league feature selection.

    Returns:
        A fitted ``CatBoostClassifier``.
    """
    X_tr = get_x(train_df, league_key)
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
    log.info(
        "Trained CatBoostClassifier on %d rows × %d features (league=%s).",
        len(X_tr), X_tr.shape[1], league_key,
    )
    return model
