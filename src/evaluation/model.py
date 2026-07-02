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
from dataclasses import dataclass

import numpy as np
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


@dataclass(frozen=True)
class ModelHyperparams:
    """A typed CatBoost hyperparameter set (single source for model search).

    ``l2_leaf_reg = None`` leaves CatBoost's own default in place (so the V6
    baseline is not pinned to a magic number); set it to regularise harder.
    """

    iterations:    int
    learning_rate: float
    depth:         int
    random_seed:   int
    l2_leaf_reg:   float | None = None


def default_hyperparams() -> ModelHyperparams:
    """The frozen-V6 hyperparameters, sourced from ``settings.model``."""
    cfg = settings.model
    return ModelHyperparams(
        iterations    = cfg.catboost_iterations,
        learning_rate = cfg.catboost_lr,
        depth         = cfg.catboost_depth,
        random_seed   = cfg.catboost_seed,
    )


def build_classifier(hp: ModelHyperparams) -> CatBoostClassifier:
    """Construct (unfitted) a CatBoostClassifier from a hyperparameter set."""
    extra = {} if hp.l2_leaf_reg is None else {"l2_leaf_reg": hp.l2_leaf_reg}
    return CatBoostClassifier(
        iterations    = hp.iterations,
        learning_rate = hp.learning_rate,
        depth         = hp.depth,
        loss_function = "Logloss",
        eval_metric   = "AUC",
        cat_features  = CAT_COLS,
        random_seed   = hp.random_seed,
        verbose       = 0,
        **extra,
    )


def train_classifier(
    train_df:  pd.DataFrame,
    target_col: str,
    league_key: str | None = None,
    sample_weight: np.ndarray | None = None,
    hyperparams: ModelHyperparams | None = None,
) -> CatBoostClassifier:
    """Train a CatBoostClassifier with the project's standard hyperparameters.

    Hyperparameters default to ``settings.model`` (the frozen V6 set); pass
    ``hyperparams`` to override them (used by model search). Loss is Logloss,
    eval metric is AUC — matched to a binary classification task.

    Args:
        train_df: Training rows with both the feature columns and the binary
            target column.
        target_col: Name of the binary target column (e.g. ``"over_hit"``).
        league_key: Forwarded to ``get_x`` for per-league feature selection.
        sample_weight: Optional per-row training weights aligned with
            ``train_df`` (e.g. seasonal weights from
            :func:`src.evaluation.seasonality.seasonal_sample_weights`). ``None``
            trains every row with equal weight (the frozen-V6 default).
        hyperparams: Optional CatBoost override. ``None`` uses
            :func:`default_hyperparams` (unchanged V6 behaviour).

    Returns:
        A fitted ``CatBoostClassifier``.
    """
    X_tr = get_x(train_df, league_key)
    y_tr = train_df[target_col].astype(int).to_numpy()
    model = build_classifier(hyperparams or default_hyperparams())
    model.fit(X_tr, y_tr, sample_weight=sample_weight)
    log.info(
        "Trained CatBoostClassifier on %d rows × %d features (league=%s, weighted=%s).",
        len(X_tr), X_tr.shape[1], league_key, sample_weight is not None,
    )
    return model
