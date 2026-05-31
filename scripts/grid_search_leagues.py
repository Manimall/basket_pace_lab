#!/usr/bin/env python3
"""Grid search: per-league × feature-combo backtester sweep.

Loads the full current-season dataset once, then runs the standard
CatBoost classifier pipeline for every (league, combo) cell and
reports LogLoss / ROC-AUC / ROI at working thresholds (0.54–0.58).

The output shows which feature combination yields the best ROI signal
for each league — feed the winners into feature_selector.py.

Feature combos tested (BASE is always on):
    BASE                — rolling stats, matchup, context (V6 baseline)
    BASE + FATIGUE      — + schedule density / B2B / rest_diff
    BASE + TEAM_ADV     — + box-score ORtg/DRtg/3PA from Go scout
    BASE + FATIGUE + TEAM_ADV — both add-ons together

Usage:
    cd /path/to/basket_pace_lab
    python scripts/grid_search_leagues.py
    python scripts/grid_search_leagues.py --leagues EuroLeague NBA
    python scripts/grid_search_leagues.py --csv results/grid_$(date +%F).csv
    python scripts/grid_search_leagues.py --min-bets 15 --csv out.csv
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

# Add project root so `src.*` imports resolve when running from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.evaluation.config import BIN_TARGET, LINE_COL
from src.evaluation.feature_selector import (
    FeatureGroup,
    _ALWAYS_EXCLUDED,
    _GROUP_COLUMNS,
)
from src.evaluation.simulation import BetReport, SimulationParams, run_threshold_sweep
from src.features.score_features import ALL_FEAT, CAT_COLS, TARGET, build_features, load_data
from src.models.evaluation import chrono_split

logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Grid definition ───────────────────────────────────────────────────────────

COMBOS: dict[str, frozenset[FeatureGroup]] = {
    "BASE":                  frozenset({FeatureGroup.BASE}),
    "BASE+FATIGUE":          frozenset({FeatureGroup.BASE, FeatureGroup.FATIGUE}),
    "BASE+TEAM_ADV":         frozenset({FeatureGroup.BASE, FeatureGroup.TEAM_ADV}),
    "BASE+FATIGUE+TEAM_ADV": frozenset({
        FeatureGroup.BASE, FeatureGroup.FATIGUE, FeatureGroup.TEAM_ADV,
    }),
}

# Thresholds to evaluate — the "working" confidence range.
GRID_THRESHOLDS: tuple[float, ...] = (0.54, 0.56, 0.58)


# ── Feature helpers (bypass FEATURES_BY_LEAGUE for the grid) ─────────────────

def _excluded_for_combo(enabled: frozenset[FeatureGroup]) -> frozenset[str]:
    """Build exclusion set for a given enabled-group frozenset.

    Mirrors feature_selector.get_excluded_features but takes the enabled set
    directly instead of a league key — lets the grid override the lookup table.
    """
    excluded: frozenset[str] = _ALWAYS_EXCLUDED
    for group, cols in _GROUP_COLUMNS.items():
        if group not in enabled:
            excluded = excluded | cols
    return excluded


def _get_x(df: pd.DataFrame, excluded: frozenset[str]) -> pd.DataFrame:
    feat_cols = [c for c in ALL_FEAT if c in df.columns and c not in excluded]
    X = df[feat_cols].copy()
    if "league" in X.columns:
        X["league"] = X["league"].astype(str)
    return X


# ── Dataset preparation ───────────────────────────────────────────────────────

def prepare_dataset() -> pd.DataFrame:
    """Load + feature-build once; derive binary target. Mirrors backtester."""
    log.info("Loading matches + quarter_stats from DB…")
    matches, qs = asyncio.run(load_data())

    log.info("Building features…")
    df = build_features(matches, qs).dropna(subset=[TARGET]).reset_index(drop=True)
    log.info("  Rows after feature build: %d", len(df))

    before = len(df)
    df = df.dropna(subset=[LINE_COL]).reset_index(drop=True)
    log.info("  Dropped %d rows without closing line → %d remain",
             before - len(df), len(df))

    pushes = int((df[TARGET] == df[LINE_COL]).sum())
    if pushes:
        log.info("  Dropping %d push rows (game_total == line)", pushes)
        df = df[df[TARGET] != df[LINE_COL]].reset_index(drop=True)

    df[BIN_TARGET] = (df[TARGET] > df[LINE_COL]).astype(int)
    log.info("  Binary target set. Overall OVER rate: %.2f%%",
             100.0 * df[BIN_TARGET].mean())
    return df


# ── Per-cell runner ───────────────────────────────────────────────────────────

def _best_roi_from_reports(
    reports: list[BetReport],
    min_bets: int,
) -> tuple[float, str, int]:
    """Return (best_roi %, threshold_str, n_bets) for the best working threshold.

    A threshold qualifies only when it has at least min_bets placed bets.
    Returns (nan, '—', 0) when no threshold qualifies.
    """
    best_roi  = float("nan")
    best_thr  = "—"
    best_bets = 0
    for r in reports:
        if r.n < min_bets:
            continue
        if np.isnan(best_roi) or r.roi > best_roi:
            best_roi  = r.roi
            best_thr  = r.row_label
            best_bets = r.n
    return round(best_roi, 2) if not np.isnan(best_roi) else best_roi, best_thr, best_bets


def run_cell(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    enabled: frozenset[FeatureGroup],
    params: SimulationParams,
    min_bets: int,
) -> dict[str, Any] | None:
    """Train + evaluate one (league, combo) cell.

    Takes pre-split train/test DataFrames (caller computes the split once per
    league, not once per combo). Returns a metrics dict or None when skipped.
    """
    guards = settings.evaluation
    if len(train_df) < guards.min_train_rows or len(test_df) < guards.min_test_rows:
        log.warning(
            "  Skipped: train=%d (min %d) / test=%d (min %d)",
            len(train_df), guards.min_train_rows,
            len(test_df),  guards.min_test_rows,
        )
        return None

    try:
        excluded = _excluded_for_combo(enabled)
        X_tr = _get_x(train_df, excluded)
        X_te = _get_x(test_df,  excluded)

        if X_tr.empty or X_te.empty or X_tr.shape[1] == 0:
            log.warning("  Skipped: empty feature matrix after exclusion.")
            return None

        y_tr = train_df[BIN_TARGET].astype(int).to_numpy()

        if len(np.unique(y_tr)) < 2:
            log.warning("  Skipped: training labels are single-class.")
            return None

        cfg = settings.model
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

        probs   = model.predict_proba(X_te)[:, 1]
        y_te    = test_df[BIN_TARGET].astype(int).to_numpy()
        actuals = test_df[TARGET].to_numpy()
        lines   = test_df[LINE_COL].to_numpy()

        ll  = log_loss(y_te, probs, labels=[0, 1])
        auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else float("nan")

        reports = run_threshold_sweep(probs, actuals, lines, params)

        result: dict[str, Any] = {
            "n_train":    len(train_df),
            "n_test":     len(test_df),
            "n_features": X_tr.shape[1],
            "log_loss":   round(ll, 4),
            "roc_auc":    round(auc, 4),
        }
        for r in reports:
            result[f"roi_{r.row_label}"]  = round(r.roi, 2) if r.n >= min_bets else float("nan")
            result[f"bets_{r.row_label}"] = r.n

        best_roi, best_thr, best_bets = _best_roi_from_reports(reports, min_bets)
        result["best_roi"]  = best_roi
        result["best_thr"]  = best_thr
        result["best_bets"] = best_bets
        return result

    except Exception as exc:
        log.error("  Cell failed with exception: %s", exc, exc_info=True)
        return None


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'═' * 80}")
    print(f"  {title}")
    print(f"{'═' * 80}")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))
    print()


def _build_full_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    thr_cols: list[str] = []
    for t in GRID_THRESHOLDS:
        ts = f"{t:.2f}"
        thr_cols += [f"roi_{ts}", f"bets_{ts}"]
    cols = [
        "league", "combo",
        "n_train", "n_test", "n_features",
        "log_loss", "roc_auc",
        *thr_cols,
        "best_roi", "best_thr", "best_bets",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values(["league", "combo"]).reset_index(drop=True)


def _build_winner_table(full: pd.DataFrame) -> pd.DataFrame:
    """One row per league: the combo with the highest best_roi (enough bets)."""
    if full.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for league, grp in full.groupby("league"):
        viable = grp.dropna(subset=["best_roi"])
        if viable.empty:
            rows.append({
                "league": league, "winner_combo": "—",
                "best_roi": float("nan"), "best_thr": "—", "best_bets": 0,
                "log_loss": float("nan"), "roc_auc": float("nan"),
            })
            continue
        # Tie-break: highest best_roi first, then most bets (better-sampled combo wins).
        idx = viable.sort_values(
            ["best_roi", "best_bets"], ascending=False
        ).index[0]
        best = viable.loc[idx]
        rows.append({
            "league":       league,
            "winner_combo": best["combo"],
            "best_roi":     best["best_roi"],
            "best_thr":     best["best_thr"],
            "best_bets":    int(best["best_bets"]),
            "log_loss":     best["log_loss"],
            "roc_auc":      best["roc_auc"],
        })
    return (
        pd.DataFrame(rows)
        .sort_values("best_roi", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--leagues", nargs="*", metavar="LEAGUE", default=None,
        help="Leagues to include (default: all in dataset). E.g. --leagues EuroLeague NBA",
    )
    p.add_argument(
        "--csv", default="", metavar="PATH",
        help="Save full grid results to this CSV (optional)",
    )
    p.add_argument(
        "--min-bets", type=int, default=10, metavar="N",
        help="Min bets at a threshold to treat its ROI as reliable (default: 10)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = prepare_dataset()

    all_leagues: list[str] = sorted(df["league"].dropna().unique().tolist())
    if args.leagues:
        unknown = [lg for lg in args.leagues if lg not in all_leagues]
        if unknown:
            log.warning("Unknown leagues (not in dataset): %s", unknown)
        leagues: list[str] = [lg for lg in args.leagues if lg in all_leagues]
        if not leagues:
            log.error("None of the requested leagues found in the dataset. Available: %s",
                      all_leagues)
            sys.exit(1)
    else:
        leagues = all_leagues
    log.info("Running grid for %d league(s): %s", len(leagues), leagues)

    params = SimulationParams(
        odds            = settings.evaluation.odds,
        stake           = settings.evaluation.flat_stake,
        thresholds      = GRID_THRESHOLDS,
        bootstrap_iters = settings.evaluation.bootstrap_iters,
        bootstrap_seed  = settings.evaluation.bootstrap_seed,
        ci_alpha        = settings.evaluation.bootstrap_ci_alpha,
    )

    records: list[dict[str, Any]] = []
    total = len(leagues) * len(COMBOS)
    done  = 0

    for league in leagues:
        subset = df[df["league"] == league].reset_index(drop=True)
        log.info("── League: %-20s  rows: %d", league, len(subset))

        if subset.empty:
            log.warning("  No data for league '%s' — skipping.", league)
            continue

        # Split once per league; reuse across all 4 combos (efficiency).
        train_df, test_dict = chrono_split(subset)
        test_df = pd.concat(test_dict.values(), ignore_index=True)

        for combo_name, enabled in COMBOS.items():
            done += 1
            log.info("  [%d/%d] %s × %s", done, total, league, combo_name)
            cell = run_cell(train_df, test_df, enabled, params, args.min_bets)
            if cell is None:
                log.info("  → skipped")
                continue
            cell["league"] = league
            cell["combo"]  = combo_name
            records.append(cell)
            log.info(
                "  → LogLoss=%.4f  AUC=%.4f  best_roi=%+.2f%% @ thr=%s (%d bets)",
                cell["log_loss"], cell["roc_auc"],
                cell["best_roi"] if not np.isnan(cell["best_roi"]) else 0.0,
                cell["best_thr"], cell["best_bets"],
            )

    if not records:
        log.error("No cells produced results — check DB data and min_rows settings.")
        sys.exit(1)

    full_df   = _build_full_table(records)
    winner_df = _build_winner_table(full_df)

    _print_table(full_df, "FULL GRID — all (league × combo) cells")
    _print_table(
        winner_df,
        "WINNER SUMMARY — best combo per league "
        "(→ paste into feature_selector.py FEATURES_BY_LEAGUE)",
    )

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(out, index=False)
        log.info("Full grid saved → %s", out)


if __name__ == "__main__":
    main()
