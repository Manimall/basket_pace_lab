"""
Per-league cross-validation for game_total prediction.

Features
--------
  Rolling L3 / L5 and EMA-5 of score-based per-team stats (shift-1, no leakage).
  NaN from a team's first game is filled with the column global mean.

Split
-----
  Chronological: last 20% of each league = test; rest = global train.
  One global CatBoostRegressor trained on all-league training data.
  Baseline: per-league mean game_total (naive predictor).

Output
------
  League | N | MAE | RMSE | Baseline MAE | Δ vs base | AvgQ1

Run:
    python -m src.models.validate_by_league
"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import text

from src.database.engine import dispose_engine, get_session_factory
from src.features.rolling_utils import (
    SCORE_STAT_COLS,
    compute_rolling_ema,
    fill_feature_nans,
)

# ── Config ────────────────────────────────────────────────────────────────────

TARGET    = "game_total"
TEST_FRAC = 0.20
MIN_TEST  = 20

_REST_CLIP_MAX = 21
_REST_DEFAULT  = 7

# ── SQL ───────────────────────────────────────────────────────────────────────

_SQL_MATCHES = """
    SELECT
        m.id           AS match_id,
        m.scheduled_at,
        m.home_team_id,
        m.away_team_id,
        m.home_score_final,
        m.away_score_final,
        m.season_type::text AS season_type,
        COALESCE(m.tournament_name, 'NBA') AS league
    FROM matches m
    WHERE m.home_score_final IS NOT NULL
      AND m.away_score_final IS NOT NULL
      AND m.has_quarter_breakdown = TRUE
    ORDER BY m.scheduled_at
"""

_SQL_QS = """
    SELECT match_id, period_number, home_score, away_score
    FROM quarter_stats
    WHERE period_type::text = 'QUARTER'
      AND period_number IN (1, 2, 3, 4)
    ORDER BY match_id, period_number
"""

# ── Data loading ──────────────────────────────────────────────────────────────


async def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    sf = get_session_factory()
    async with sf() as db:
        m_rows = (await db.execute(text(_SQL_MATCHES))).mappings().all()
        q_rows = (await db.execute(text(_SQL_QS))).mappings().all()
    await dispose_engine()
    matches = pd.DataFrame(m_rows)
    qs      = pd.DataFrame(q_rows)
    matches["scheduled_at"] = pd.to_datetime(matches["scheduled_at"], utc=True)
    return matches, qs


# ── Feature engineering ───────────────────────────────────────────────────────


def _build_features(matches: pd.DataFrame, qs: pd.DataFrame) -> pd.DataFrame:
    """
    Score-based rolling L3/L5 and EMA-5 per team, shift-1 (no leakage).
    NaN in first-game rows filled with column mean.
    """
    wide = (
        qs.pivot_table(
            index="match_id",
            columns="period_number",
            values=["home_score", "away_score"],
            aggfunc="first",
        )
    )
    wide.columns = [
        f"{'h' if v == 'home_score' else 'a'}_q{p}" for v, p in wide.columns
    ]
    wide = wide.reset_index()

    df = matches.merge(wide, on="match_id", how="left")
    df["game_total"] = df["home_score_final"] + df["away_score_final"]
    df["q1_total"]   = df["h_q1"] + df["a_q1"]
    # h_q2/a_q2 always present: SQL filters has_quarter_breakdown=TRUE
    df["h1_total"]   = (
        df["h_q1"].fillna(0) + df["h_q2"].fillna(0) +
        df["a_q1"].fillna(0) + df["a_q2"].fillna(0)
    )

    home_tl = pd.DataFrame({
        "match_id":         df["match_id"],
        "team_id":          df["home_team_id"],
        "scheduled_at":     df["scheduled_at"],
        "pts_scored_q1":    df["h_q1"],
        "pts_allowed_q1":   df["a_q1"],
        "pts_scored_h1":    df["h_q1"].fillna(0) + df["h_q2"].fillna(0),
        "pts_allowed_h1":   df["a_q1"].fillna(0) + df["a_q2"].fillna(0),
        "pts_scored_game":  df["home_score_final"],
        "pts_allowed_game": df["away_score_final"],
    })
    away_tl = pd.DataFrame({
        "match_id":         df["match_id"],
        "team_id":          df["away_team_id"],
        "scheduled_at":     df["scheduled_at"],
        "pts_scored_q1":    df["a_q1"],
        "pts_allowed_q1":   df["h_q1"],
        "pts_scored_h1":    df["a_q1"].fillna(0) + df["a_q2"].fillna(0),
        "pts_allowed_h1":   df["h_q1"].fillna(0) + df["h_q2"].fillna(0),
        "pts_scored_game":  df["away_score_final"],
        "pts_allowed_game": df["home_score_final"],
    })

    timeline = pd.concat([home_tl, away_tl], ignore_index=True)
    rolling  = compute_rolling_ema(timeline)

    roll_cols = [
        f"{col}_{sfx}"
        for col in SCORE_STAT_COLS
        for sfx in ("L3", "L5", f"EMA{5}")
    ]
    keep = ["match_id", "team_id"] + roll_cols

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        side_roll = (
            rolling
            .merge(df[["match_id", team_col]], left_on=["match_id", "team_id"],
                   right_on=["match_id", team_col], how="inner")[keep]
            .rename(columns={c: f"{side}_{c}" for c in roll_cols})
            .drop(columns="team_id")
        )
        df = df.merge(side_roll, on="match_id", how="left")

    feat_cols = [f"{side}_{c}" for side in ("home", "away") for c in roll_cols]
    fill_feature_nans(df, feat_cols)

    # ── days rest ─────────────────────────────────────────────────────────────
    all_apps = pd.concat([
        df[["match_id", "scheduled_at", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
        df[["match_id", "scheduled_at", "away_team_id"]].rename(columns={"away_team_id": "team_id"}),
    ]).sort_values(["team_id", "scheduled_at"])
    all_apps["days_rest"] = all_apps.groupby("team_id")["scheduled_at"].diff().dt.days
    all_apps["days_rest"] = all_apps["days_rest"].fillna(_REST_DEFAULT).clip(0, _REST_CLIP_MAX)

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        rest = (
            all_apps
            .merge(df[["match_id", team_col]].rename(columns={team_col: "team_id"}),
                   on=["match_id", "team_id"])[["match_id", "days_rest"]]
            .rename(columns={"days_rest": f"{side}_days_rest"})
        )
        df = df.merge(rest, on="match_id", how="left")

    df["is_playoff"]      = (df["season_type"] == "playoffs").astype(int)
    df["home_days_rest"]  = df["home_days_rest"].fillna(_REST_DEFAULT).clip(0, _REST_CLIP_MAX)
    df["away_days_rest"]  = df["away_days_rest"].fillna(_REST_DEFAULT).clip(0, _REST_CLIP_MAX)
    return df


# ── Feature / target selection ────────────────────────────────────────────────

ROLL_FEAT_COLS = [
    f"{side}_{stat}_{sfx}"
    for side in ("home", "away")
    for stat in SCORE_STAT_COLS
    for sfx in ("L3", "L5", "EMA5")
]
CTX_COLS  = ["home_days_rest", "away_days_rest", "is_playoff"]
CAT_COLS  = ["league"]
ALL_FEAT  = ROLL_FEAT_COLS + CTX_COLS + CAT_COLS


def _get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[ALL_FEAT].copy()
    X["league"] = X["league"].astype(str)
    return X, df[TARGET]


# ── Train / test split ────────────────────────────────────────────────────────


def _chrono_split(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    test_parts: dict[str, pd.DataFrame] = {}
    train_idx: list[int] = []
    for league, grp in df.groupby("league", sort=False):
        grp_sorted = grp.sort_values("scheduled_at")
        n_test = max(1, int(len(grp_sorted) * TEST_FRAC))
        test_parts[str(league)] = df.loc[grp_sorted.index[-n_test:]]
        train_idx.extend(grp_sorted.index[:-n_test].tolist())
    return df.loc[train_idx].sort_values("scheduled_at"), test_parts


# ── Model ─────────────────────────────────────────────────────────────────────


def _train(train_df: pd.DataFrame) -> CatBoostRegressor:
    train_df = train_df.dropna(subset=[TARGET])
    X_tr, y_tr = _get_xy(train_df)
    model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=5,
        loss_function="MAE",
        eval_metric="MAE",
        cat_features=CAT_COLS,
        random_seed=42,
        verbose=0,
    )
    model.fit(X_tr, y_tr)
    return model


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return mae, rmse


# ── Pretty-print table ────────────────────────────────────────────────────────


def _print_table(rows: list[dict]) -> None:
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


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading data from DB…")
    matches, qs = asyncio.run(_load())
    print(f"  Matches: {len(matches):,}  |  QuarterStats rows: {len(qs):,}")

    print("Building rolling L3/L5/EMA-5 features…")
    df = _build_features(matches, qs).dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"  Feature rows: {len(df):,}  |  Features: {len(ALL_FEAT)}")

    print("Splitting per-league (last 20% = test)…")
    train_df, test_dict = _chrono_split(df)
    print(f"  Train: {len(train_df):,}  |  Test leagues: {len(test_dict)}")

    print("Training CatBoost…")
    model = _train(train_df)
    print("  Done.\n")

    result_rows: list[dict] = []
    for league, test_df in sorted(test_dict.items()):
        test_df = test_df.dropna(subset=[TARGET])
        if len(test_df) < MIN_TEST:
            continue
        X_te, y_te = _get_xy(test_df)
        preds = model.predict(X_te)
        mae, rmse = _metrics(y_te.values, preds)

        train_league  = train_df[train_df["league"] == league]
        baseline_mean = train_league[TARGET].mean() if len(train_league) > 0 else test_df[TARGET].mean()
        baseline_mae, _ = _metrics(y_te.values, np.full(len(y_te), baseline_mean))

        result_rows.append({
            "league":       league,
            "n":            len(test_df),
            "mae":          mae,
            "rmse":         rmse,
            "baseline_mae": baseline_mae,
            "avg_q1":       test_df["q1_total"].mean() if pd.notna(test_df["q1_total"].mean()) else 0.0,
        })

    _print_table(result_rows)

    feat_imp = pd.DataFrame({
        "feature":    ALL_FEAT,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False).head(12)
    print("Top-12 Feature Importances:")
    for _, row in feat_imp.iterrows():
        bar = "█" * int(row["importance"] / feat_imp["importance"].max() * 25)
        print(f"  {row['feature']:<45s} {row['importance']:5.1f}%  {bar}")
    print()


if __name__ == "__main__":
    main()
