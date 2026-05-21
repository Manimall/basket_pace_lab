"""
MVP model: predict q1_avg_pace (first-quarter pace).

Why pace, not total:
  Shot realisation adds irreducible noise to scoring predictions.
  Pace is a stable team-system characteristic — much lower variance,
  better suited for finding live value in Q1 betting markets.

Run:
    python -m src.models.train_model
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.features.build_features import BasketballFeatureBuilder

TARGET = "q1_avg_pace"

# All target columns that must be excluded from X to prevent data leakage
_ALL_TARGETS = [
    "game_total", "game_avg_pace",
    *(f"q{p}_{t}" for p in range(1, 5) for t in ("total", "avg_pace")),
]

# Columns that are identifiers / raw scores — not predictive at inference time
_DROP_COLS = [
    "match_id", "external_id", "season",
    "home_team", "away_team",
    "home_team_id", "away_team_id",
    "home_score_final", "away_score_final",
    "home_score_regulation", "away_score_regulation",
    "went_to_overtime",
    "home_poss", "away_poss",
    "home_fga", "away_fga", "home_fta", "away_fta",
    "home_off_reb", "away_off_reb",
    "home_to", "away_to",
    "home_pace", "away_pace",
    "home_opp_pace", "away_opp_pace",
    "home_ft_rate", "away_ft_rate",
    "home_to_rate", "away_to_rate",
    "home_oreb_rate", "away_oreb_rate",
    "home_pts_per_poss", "away_pts_per_poss",
    # per-quarter raw scores
    *(f"q{p}_{s}" for p in range(1, 5) for s in ("home_score", "away_score", "home_pace", "away_pace")),
]


def load_dataset() -> pd.DataFrame:
    builder = BasketballFeatureBuilder()
    return builder.build()


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    # Strict chronological order
    df = df.sort_values("scheduled_at").reset_index(drop=True)

    # Drop rows without rolling history (first few games of the season)
    feature_cols = BasketballFeatureBuilder.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + [TARGET]).reset_index(drop=True)

    # Build feature matrix — remove all targets and raw columns
    exclude = set(_ALL_TARGETS) | set(_DROP_COLS) | {"scheduled_at"}
    X = df.drop(columns=[c for c in exclude if c in df.columns])
    y = df[TARGET]

    return X, y, list(X.columns)


def train(X_tr: pd.DataFrame, y_tr: pd.Series,
          X_te: pd.DataFrame, y_te: pd.Series) -> CatBoostRegressor:
    train_pool = Pool(X_tr, y_tr)
    eval_pool  = Pool(X_te, y_te)

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=100,
    )
    model.fit(
        train_pool,
        eval_set=eval_pool,
        early_stopping_rounds=50,
    )
    return model


def evaluate(model: CatBoostRegressor,
             X_te: pd.DataFrame, y_te: pd.Series) -> dict[str, float]:
    preds = model.predict(X_te)
    mae  = mean_absolute_error(y_te, preds)
    rmse = mean_squared_error(y_te, preds) ** 0.5
    return {"MAE": mae, "RMSE": rmse, "preds": preds}


def feature_importance(model: CatBoostRegressor,
                       feature_names: list[str],
                       top_n: int = 10) -> pd.DataFrame:
    imp = model.get_feature_importance()
    return (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def save_model(model: CatBoostRegressor) -> Path:
    out = Path(__file__).parent / "q1_pace_v1.cbm"
    model.save_model(str(out))
    return out


def main() -> None:
    print("Loading dataset…")
    df = load_dataset()
    print(f"  Raw rows: {len(df)}")

    X, y, feat_names = prepare_xy(df)
    print(f"  After dropna: {len(X)} rows  |  {len(feat_names)} features")

    # Chronological split — NO shuffle
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"  Train: {len(X_tr)}  |  Test: {len(X_te)}")
    print()

    print("Training CatBoostRegressor…")
    model = train(X_tr, y_tr, X_te, y_te)

    print()
    print("─" * 45)
    metrics = evaluate(model, X_te, y_te)
    print(f"  MAE  : {metrics['MAE']:.4f}")
    print(f"  RMSE : {metrics['RMSE']:.4f}")
    print("─" * 45)

    print()
    print("TOP-10 Feature Importance:")
    fi = feature_importance(model, feat_names)
    for i, row in fi.iterrows():
        bar = "█" * int(row["importance"] / fi["importance"].max() * 20)
        print(f"  {i+1:>2}. {row['feature']:<35} {row['importance']:6.2f}%  {bar}")

    path = save_model(model)
    print()
    print(f"Model saved → {path}")


if __name__ == "__main__":
    main()
