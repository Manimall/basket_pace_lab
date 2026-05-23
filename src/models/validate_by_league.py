"""
Per-league cross-validation for game_total prediction.

Features
--------
  Rolling L3 / L5 / EMA-5 of score-based per-team stats (shift-1, no leakage).
  Matchup features: attack/defense strength ratios, win_rate_diff, expected_pace.
  NaN from a team's first game is filled with the column global mean.

Split
-----
  Chronological: last 20% of each league = test; rest = global train.
  One global CatBoostRegressor trained on all-league training data.
  Baseline: per-league mean game_total (naive predictor).

Run:
    python -m src.models.validate_by_league
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np
import pandas as pd

from src.config import settings
from src.features.score_features import ALL_FEAT, TARGET, build_features, load_data
from src.models.evaluation import (
    chrono_split,
    compute_metrics,
    get_xy,
    print_results_table,
    train,
)

logging.basicConfig(level=settings.app.log_level, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    log.info("Loading data from DB…")
    matches, qs = asyncio.run(load_data())
    log.info("  Matches: %d  |  QuarterStats rows: %d", len(matches), len(qs))

    log.info("Building features (rolling L3/L5/EMA-5 + matchup)…")
    df = build_features(matches, qs).dropna(subset=[TARGET]).reset_index(drop=True)
    feat_cols = [c for c in ALL_FEAT if c in df.columns]
    log.info("  Feature rows: %d  |  Features: %d", len(df), len(feat_cols))

    log.info("Splitting per-league (last %.0f%% = test)…", settings.model.test_frac * 100)
    train_df, test_dict = chrono_split(df)
    log.info("  Train: %d  |  Test leagues: %d", len(train_df), len(test_dict))

    log.info("Training CatBoost…")
    model = train(train_df)

    result_rows: list[dict] = []
    for league, test_df in sorted(test_dict.items()):
        test_df = test_df.dropna(subset=[TARGET])
        if len(test_df) < settings.model.min_test_rows:
            continue
        X_te, y_te = get_xy(test_df)
        preds = model.predict(X_te)
        mae, rmse = compute_metrics(y_te.values, preds)

        train_league  = train_df[train_df["league"] == league]
        baseline_mean = train_league[TARGET].mean() if len(train_league) > 0 else test_df[TARGET].mean()
        baseline_mae, _ = compute_metrics(y_te.values, np.full(len(y_te), baseline_mean))

        avg_q1 = test_df["q1_total"].mean()
        result_rows.append({
            "league":       league,
            "n":            len(test_df),
            "mae":          mae,
            "rmse":         rmse,
            "baseline_mae": baseline_mae,
            "avg_q1":       avg_q1 if pd.notna(avg_q1) else 0.0,
        })

    print_results_table(result_rows)

    feat_imp = pd.DataFrame({
        "feature":    feat_cols,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False).head(15)
    print("Top-15 Feature Importances:")
    for _, row in feat_imp.iterrows():
        bar = "█" * int(row["importance"] / feat_imp["importance"].max() * 25)
        print(f"  {row['feature']:<45s} {row['importance']:5.1f}%  {bar}")
    print()


if __name__ == "__main__":
    main()
