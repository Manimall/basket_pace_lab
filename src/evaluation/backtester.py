"""
Profit simulation backtester V6 — significance test + per-league split.

Logic
-----
  V4 architecture restored: raw CatBoostClassifier.predict_proba, no calibration,
  no Kelly. Pure 1u flat betting on Over/Under signal from a classifier trained
  with line-derived features hidden.

  V6 additions:
    * Bootstrap 95% CI for ROI on each threshold — answers "is this result
      distinguishable from zero on the current sample size?"
    * Three independent league-segregated pipelines, each with its own model:
        - NBA only           (main US market)
        - EuroLeague only    (V1 hot spot, 65% WR baseline)
        - OTHER leagues      (control — everything except NBA & EuroLeague)

Output
------
  Three threshold-rowed tables. Columns:
      Thr | Bets | W | L | P | Winrate | ROI | Profit(u) | ROI CI95

Run:
    python -m src.evaluation.backtester
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.config import settings
from src.features.score_features import (
    ALL_FEAT, BM_COLS, CAT_COLS, TARGET, build_features, load_data,
)
from src.models.evaluation import chrono_split

logging.basicConfig(level=settings.app.log_level, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

LINE_COL   = "bookmaker_total_closing"
BIN_TARGET = "over_hit"

ODDS     = 1.90
WIN_MULT = ODDS - 1.0    # +0.90u profit on a winning 1u flat bet
STAKE    = 1.0

PROB_THRESHOLDS: list[float] = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]

BOOTSTRAP_ITERS = 5000
BOOTSTRAP_SEED  = 42

_EXCLUDED_FEATS: set[str] = set(BM_COLS)  # line-derived features hidden from the model


@dataclass
class BetReport:
    row_label:   str
    n:           int
    wins:        int
    losses:      int
    pushes:      int
    profit:      float
    roi_ci_low:  float    # bootstrap 2.5th percentile of ROI%
    roi_ci_high: float    # bootstrap 97.5th percentile of ROI%

    @property
    def winrate(self) -> float:
        decided = self.wins + self.losses
        return 100.0 * self.wins / decided if decided > 0 else 0.0

    @property
    def roi(self) -> float:
        return 100.0 * self.profit / self.n if self.n > 0 else 0.0


def simulate(
    probs: np.ndarray, actuals: np.ndarray, lines: np.ndarray, threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (label, pnl). label: 1=win, 0=push, -1=loss, NaN=no-bet. pnl: 1u flat."""
    over_mask  = probs >= threshold
    under_mask = (probs <= 1.0 - threshold) & ~over_mask

    direction = np.zeros(len(probs))
    direction[over_mask]  =  1.0
    direction[under_mask] = -1.0
    placed = direction != 0
    actual = np.sign(actuals - lines)

    push = placed & (actual == 0)
    win  = placed & (actual == direction)
    loss = placed & ~push & ~win

    pnl = np.zeros(len(probs))
    pnl[win]  = STAKE * WIN_MULT
    pnl[loss] = -STAKE

    label = np.full(len(probs), np.nan)
    label[win]  = 1
    label[push] = 0
    label[loss] = -1
    return label, pnl


def _bootstrap_roi_ci(
    pnl: np.ndarray, n_iter: int = BOOTSTRAP_ITERS, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% CI for ROI% via bootstrap on the placed-bets pnl array."""
    n = len(pnl)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, n))
    boot_means = pnl[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)) * 100, \
           float(np.percentile(boot_means, 97.5)) * 100


def _aggregate(
    label: np.ndarray, pnl: np.ndarray, row_label: str,
) -> BetReport:
    placed = ~np.isnan(label)
    placed_pnl = pnl[placed]
    lo, hi = _bootstrap_roi_ci(placed_pnl)
    return BetReport(
        row_label   = row_label,
        n           = int(placed.sum()),
        wins        = int(np.sum(label ==  1)),
        losses      = int(np.sum(label == -1)),
        pushes      = int(np.sum(label ==  0)),
        profit      = float(placed_pnl.sum()),
        roi_ci_low  = lo,
        roi_ci_high = hi,
    )


def run_threshold_sweep(
    probs: np.ndarray, actuals: np.ndarray, lines: np.ndarray, thresholds: list[float],
) -> list[BetReport]:
    return [
        _aggregate(*simulate(probs, actuals, lines, t), row_label=f"{t:.2f}")
        for t in thresholds
    ]


def _fmt_ci(lo: float, hi: float) -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return f"{'—':>18s}"
    return f"[{lo:+6.2f}%; {hi:+6.2f}%]"


def _print_table(title: str, rows: list[BetReport]) -> None:
    hdr = (
        f"{'Thr':>5s} {'Bets':>5s} {'W':>4s} {'L':>4s} {'P':>3s} "
        f"{'Winrate':>8s} {'ROI':>8s} {'Profit(u)':>10s}  {'ROI CI95':>18s}"
    )
    sep = "─" * len(hdr)
    print(f"\n{title}")
    print(f"{sep}\n{hdr}\n{sep}")
    for r in rows:
        flag = "★" if r.n > 0 and r.roi_ci_low > 0 else (
               "✓" if r.profit > 0 else ("✗" if r.n > 0 else " "))
        wr  = f"{r.winrate:>6.2f}%" if (r.wins + r.losses) > 0 else f"{'—':>7s}"
        roi = f"{r.roi:>+6.2f}%"    if r.n > 0 else f"{'—':>7s}"
        ci  = _fmt_ci(r.roi_ci_low, r.roi_ci_high)
        print(
            f"{r.row_label:>5s} {r.n:>5d} {r.wins:>4d} {r.losses:>4d} {r.pushes:>3d} "
            f"{wr:>8s} {roi:>8s} {r.profit:>+10.2f}  {ci:>18s} {flag}"
        )
    print(sep)
    print("  ★ = CI95 lower bound > 0 (statistically distinguishable from zero)\n")


def _get_x(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in ALL_FEAT if c in df.columns and c not in _EXCLUDED_FEATS]
    X = df[feat_cols].copy()
    X["league"] = X["league"].astype(str)
    return X


def _train_classifier(train_df: pd.DataFrame) -> CatBoostClassifier:
    X_tr = _get_x(train_df)
    y_tr = train_df[BIN_TARGET].astype(int).to_numpy()
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


def _run_pipeline(df_subset: pd.DataFrame, name: str) -> list[BetReport] | None:
    """Train on the train portion of df_subset, test on its test portion, return sweep."""
    if df_subset.empty:
        log.warning("[%s] empty subset — skipping", name)
        return None

    leagues = sorted(df_subset["league"].unique().tolist())
    log.info("[%s] rows=%d  leagues=%s", name, len(df_subset), leagues)

    train_df, test_dict = chrono_split(df_subset)
    test_df = pd.concat(test_dict.values(), ignore_index=True)
    log.info("[%s]  train=%d  test=%d  (OVER rate in train: %.2f%%)",
             name, len(train_df), len(test_df), 100 * train_df[BIN_TARGET].mean())

    if len(train_df) < 100 or len(test_df) < 30:
        log.warning("[%s] too few rows (train=%d, test=%d) — skipping",
                    name, len(train_df), len(test_df))
        return None

    model   = _train_classifier(train_df)
    X_te    = _get_x(test_df)
    probs   = model.predict_proba(X_te)[:, 1]
    actuals = test_df[TARGET].to_numpy()
    lines   = test_df[LINE_COL].to_numpy()

    return run_threshold_sweep(probs, actuals, lines, PROB_THRESHOLDS)


def main() -> None:
    log.info("Loading data from DB…")
    matches, qs = asyncio.run(load_data())

    log.info("Building features…")
    df = build_features(matches, qs).dropna(subset=[TARGET]).reset_index(drop=True)
    log.info("  Rows after feature build: %d", len(df))

    before = len(df)
    df = df.dropna(subset=[LINE_COL]).reset_index(drop=True)
    log.info("  Dropped %d rows without %s → %d remain",
             before - len(df), LINE_COL, len(df))

    pushes = int((df[TARGET] == df[LINE_COL]).sum())
    if pushes:
        log.info("  Dropping %d push rows (game_total == line)", pushes)
        df = df[df[TARGET] != df[LINE_COL]].reset_index(drop=True)

    df[BIN_TARGET] = (df[TARGET] > df[LINE_COL]).astype(int)
    log.info("  Overall OVER rate: %.2f%%", 100 * df[BIN_TARGET].mean())

    feat_count = len([c for c in ALL_FEAT if c in df.columns and c not in _EXCLUDED_FEATS])
    log.info("Using %d features (hidden: %s)", feat_count, sorted(_EXCLUDED_FEATS))

    # Per-league pipelines
    pipelines: list[tuple[str, pd.DataFrame]] = [
        ("NBA",                 df[df["league"] == "NBA"]),
        ("EuroLeague",          df[df["league"] == "EuroLeague"]),
        ("OTHER (control)",     df[~df["league"].isin(["NBA", "EuroLeague"])]),
    ]

    results: dict[str, list[BetReport]] = {}
    for name, subset in pipelines:
        rows = _run_pipeline(subset.reset_index(drop=True), name)
        if rows is not None:
            results[name] = rows

    for name, rows in results.items():
        _print_table(f"{name} — by probability threshold (Bootstrap CI95, {BOOTSTRAP_ITERS} iters):", rows)


if __name__ == "__main__":
    main()
