#!/usr/bin/env python3
"""Grid search: per-league × feature-combo backtester sweep.

Loads the full current-season dataset once, then runs the standard
CatBoost classifier pipeline for every (league, combo) cell and
reports LogLoss / ROC-AUC / ROI at working thresholds.

Three target periods are supported via ``--period``:

    game (default)  — полный тотал матча vs. закрывающая линия букмекера
    1q              — тотал 1-й четверти vs. синтетическая медиана (нет линий)
    1h              — тотал 1-й половины  vs. синтетическая медиана (нет линий)

Для 1Q и 1H реальных линий букмекеров в базе нет, поэтому ROI считается
относительно медианы тренировочной выборки (метрика качества модели,
а не реального ROI на рынке).

Feature combos tested (BASE is always on):
    BASE                — rolling stats, matchup, context (V6 baseline)
    BASE + FATIGUE      — + schedule density / B2B / rest_diff
    BASE + TEAM_ADV     — + box-score ORtg/DRtg/3PA from Go scout
    BASE + FATIGUE + TEAM_ADV — both add-ons together

Usage:
    cd /path/to/basket_pace_lab
    python scripts/grid_search_leagues.py
    python scripts/grid_search_leagues.py --period 1q --csv results/grid_search_1q.csv
    python scripts/grid_search_leagues.py --period 1h --csv results/grid_search_1h.csv
    python scripts/grid_search_leagues.py --leagues EuroLeague NBA --period game
    python scripts/grid_search_leagues.py --min-bets 15 --csv out.csv
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

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

# ── Period configuration ──────────────────────────────────────────────────────

Period = Literal["game", "1q", "1h"]


@dataclass(frozen=True)
class PeriodConfig:
    """Configuration for one target period.

    Attributes:
        period:     Identifier passed via --period flag.
        target_col: Column name of the raw score total to predict.
        label:      Human-readable label for table headers.
        has_real_line: True only for "game" where bookmaker_total_closing exists.
    """
    period:        Period
    target_col:    str
    label:         str
    has_real_line: bool


_PERIOD_CONFIGS: dict[Period, PeriodConfig] = {
    "game": PeriodConfig(
        period        = "game",
        target_col    = "game_total",
        label         = "GAME TOTAL",
        has_real_line = True,
    ),
    "1q": PeriodConfig(
        period        = "1q",
        target_col    = "q1_total",
        label         = "1-я ЧЕТВЕРТЬ",
        has_real_line = False,
    ),
    "1h": PeriodConfig(
        period        = "1h",
        target_col    = "h1_total",
        label         = "1-я ПОЛОВИНА",
        has_real_line = False,
    ),
}

# ── Grid definition ───────────────────────────────────────────────────────────

COMBOS: dict[str, frozenset[FeatureGroup]] = {
    "BASE":                  frozenset({FeatureGroup.BASE}),
    "BASE+FATIGUE":          frozenset({FeatureGroup.BASE, FeatureGroup.FATIGUE}),
    "BASE+TEAM_ADV":         frozenset({FeatureGroup.BASE, FeatureGroup.TEAM_ADV}),
    "BASE+FATIGUE+TEAM_ADV": frozenset({
        FeatureGroup.BASE, FeatureGroup.FATIGUE, FeatureGroup.TEAM_ADV,
    }),
}

GRID_THRESHOLDS: tuple[float, ...] = (0.54, 0.56, 0.58)


# ── Feature helpers ───────────────────────────────────────────────────────────

def _excluded_for_combo(enabled: frozenset[FeatureGroup]) -> frozenset[str]:
    """Build exclusion set for a given enabled-group frozenset."""
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

def prepare_dataset(period: Period) -> pd.DataFrame:
    """Load + feature-build; derive binary target for the given period.

    For "game": uses bookmaker_total_closing as the line (real market).
    For "1q"/"1h": the binary target is derived inside run_cell from the
    training-set median — the raw scores are returned without BIN_TARGET
    so each cell can compute a leak-free per-league median.

    Args:
        period: Target period — "game", "1q", or "1h".

    Returns:
        DataFrame with feature columns + period target column populated.
        BIN_TARGET is set only for "game"; absent for "1q"/"1h".
    """
    cfg = _PERIOD_CONFIGS[period]
    log.info("Загрузка матчей + quarter_stats из БД…")
    matches, qs = asyncio.run(load_data())

    log.info("Построение фичей…")
    df = build_features(matches, qs)

    # Drop rows where period target is missing
    before = len(df)
    df = df.dropna(subset=[cfg.target_col]).reset_index(drop=True)
    if len(df) < before:
        log.info("  Удалено %d строк без %s", before - len(df), cfg.target_col)
    log.info("  Строк после фильтрации: %d", len(df))

    if period == "game":
        # Game period: use real bookmaker line, drop rows without it
        df = df.dropna(subset=[LINE_COL]).reset_index(drop=True)
        pushes = int((df[cfg.target_col] == df[LINE_COL]).sum())
        if pushes:
            log.info("  Удаляю %d push-строк (game_total == line)", pushes)
            df = df[df[cfg.target_col] != df[LINE_COL]].reset_index(drop=True)
        df[BIN_TARGET] = (df[cfg.target_col] > df[LINE_COL]).astype(int)
        log.info(
            "  OVER rate: %.2f%%  (линия: %s)",
            100.0 * df[BIN_TARGET].mean(), LINE_COL,
        )
    else:
        # Sub-game periods: no real line — BIN_TARGET derived per cell from median
        log.info(
            "  Период '%s': нет реальных линий букмекеров. "
            "ROI считается относительно медианы тренировочного сета.",
            period,
        )

    return df


# ── Per-cell runner ───────────────────────────────────────────────────────────

def _best_roi_from_reports(
    reports: list[BetReport],
    min_bets: int,
) -> tuple[float, str, int]:
    """Return (best_roi %, threshold_str, n_bets) for the best qualifying threshold."""
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
    return (round(best_roi, 2) if not np.isnan(best_roi) else best_roi), best_thr, best_bets


def _derive_bin_target_from_median(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Create binary over/under target based on training-set median.

    Uses training median as the synthetic line — prevents look-ahead leakage.
    Returns updated train_df, test_df, and the median value used as the line.
    """
    median_val = float(train_df[target_col].median())
    for df_ref, df_out in [(train_df, train_df), (test_df, test_df)]:
        df_out = df_out.copy()
        df_out[BIN_TARGET] = (df_out[target_col] > median_val).astype(int)
        df_out[LINE_COL]   = median_val
    # Re-assign with copies
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[BIN_TARGET] = (train_df[target_col] > median_val).astype(int)
    train_df[LINE_COL]   = median_val
    test_df[BIN_TARGET]  = (test_df[target_col] > median_val).astype(int)
    test_df[LINE_COL]    = median_val
    return train_df, test_df, median_val


def run_cell(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    enabled:  frozenset[FeatureGroup],
    params:   SimulationParams,
    min_bets: int,
    period:   Period,
) -> dict[str, Any] | None:
    """Train + evaluate one (league, combo) cell for the given period.

    Args:
        train_df: Training split (pre-split by caller for efficiency).
        test_df:  Test split.
        enabled:  Feature groups enabled for this combo.
        params:   Simulation knobs (odds, stake, thresholds, bootstrap).
        min_bets: Minimum bets at threshold to treat ROI as reliable.
        period:   Target period — affects binary target derivation.

    Returns:
        Metrics dict, or None when the cell is skipped.
    """
    guards = settings.evaluation
    if len(train_df) < guards.min_train_rows or len(test_df) < guards.min_test_rows:
        log.warning(
            "  Пропуск: train=%d (min %d) / test=%d (min %d)",
            len(train_df), guards.min_train_rows,
            len(test_df),  guards.min_test_rows,
        )
        return None

    try:
        cfg = _PERIOD_CONFIGS[period]

        # For sub-game periods: compute synthetic line from training set
        synthetic_line: float | None = None
        if not cfg.has_real_line:
            if train_df[cfg.target_col].isna().all():
                log.warning("  Пропуск: все значения %s — NaN.", cfg.target_col)
                return None
            train_df, test_df, synthetic_line = _derive_bin_target_from_median(
                train_df, test_df, cfg.target_col
            )

        # Guard: single-class target
        if BIN_TARGET not in train_df.columns or len(train_df[BIN_TARGET].unique()) < 2:
            log.warning("  Пропуск: целевая переменная однородна (один класс).")
            return None

        excluded = _excluded_for_combo(enabled)
        X_tr = _get_x(train_df, excluded)
        X_te = _get_x(test_df,  excluded)

        if X_tr.empty or X_te.empty or X_tr.shape[1] == 0:
            log.warning("  Пропуск: пустая матрица фичей после исключений.")
            return None

        y_tr = train_df[BIN_TARGET].astype(int).to_numpy()

        model_cfg = settings.model
        model = CatBoostClassifier(
            iterations    = model_cfg.catboost_iterations,
            learning_rate = model_cfg.catboost_lr,
            depth         = model_cfg.catboost_depth,
            loss_function = "Logloss",
            eval_metric   = "AUC",
            cat_features  = CAT_COLS,
            random_seed   = model_cfg.catboost_seed,
            verbose       = 0,
        )
        model.fit(X_tr, y_tr)

        probs   = model.predict_proba(X_te)[:, 1]
        y_te    = test_df[BIN_TARGET].astype(int).to_numpy()
        actuals = test_df[cfg.target_col].to_numpy()
        lines   = test_df[LINE_COL].to_numpy()

        ll  = log_loss(y_te, probs, labels=[0, 1])
        auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else float("nan")

        reports = run_threshold_sweep(probs, actuals, lines, params)

        result: dict[str, Any] = {
            "n_train":        len(train_df),
            "n_test":         len(test_df),
            "n_features":     X_tr.shape[1],
            "log_loss":       round(ll, 4),
            "roc_auc":        round(auc, 4),
        }
        if synthetic_line is not None:
            result["synthetic_line"] = round(synthetic_line, 2)

        for r in reports:
            result[f"roi_{r.row_label}"]  = round(r.roi, 2) if r.n >= min_bets else float("nan")
            result[f"bets_{r.row_label}"] = r.n

        best_roi, best_thr, best_bets = _best_roi_from_reports(reports, min_bets)
        result["best_roi"]  = best_roi
        result["best_thr"]  = best_thr
        result["best_bets"] = best_bets
        return result

    except Exception as exc:
        log.error("  Ячейка завершилась с ошибкой: %s", exc, exc_info=True)
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


def _build_full_table(records: list[dict[str, Any]], period: Period) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    thr_cols: list[str] = []
    for t in GRID_THRESHOLDS:
        ts = f"{t:.2f}"
        thr_cols += [f"roi_{ts}", f"bets_{ts}"]
    base_cols = ["league", "combo", "n_train", "n_test", "n_features", "log_loss", "roc_auc"]
    if period != "game" and "synthetic_line" in df.columns:
        base_cols.append("synthetic_line")
    cols = base_cols + thr_cols + ["best_roi", "best_thr", "best_bets"]
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
        idx  = viable.sort_values(["best_roi", "best_bets"], ascending=False).index[0]
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
        "--period", choices=["game", "1q", "1h"], default="game",
        help=(
            "Целевой период: 'game' — полный тотал (по умолчанию), "
            "'1q' — 1-я четверть, '1h' — 1-я половина"
        ),
    )
    p.add_argument(
        "--leagues", nargs="*", metavar="LEAGUE", default=None,
        help="Лиги для прогона (по умолчанию — все). Пример: --leagues EuroLeague NBA",
    )
    p.add_argument(
        "--csv", default="", metavar="PATH",
        help="Путь для сохранения полных результатов в CSV (опционально)",
    )
    p.add_argument(
        "--min-bets", type=int, default=10, metavar="N",
        help="Минимальное число ставок на пороге для учёта ROI (по умолчанию: 10)",
    )
    return p.parse_args()


def _default_csv(period: Period) -> str:
    return f"results/grid_search_{period}.csv"


def main() -> None:
    args = parse_args()
    period: Period = args.period
    period_cfg = _PERIOD_CONFIGS[period]

    df = prepare_dataset(period)

    all_leagues: list[str] = sorted(df["league"].dropna().unique().tolist())
    if args.leagues:
        unknown = [lg for lg in args.leagues if lg not in all_leagues]
        if unknown:
            log.warning("Лиги не найдены в датасете: %s", unknown)
        leagues: list[str] = [lg for lg in args.leagues if lg in all_leagues]
        if not leagues:
            log.error("Ни одна из запрошенных лиг не найдена. Доступные: %s", all_leagues)
            sys.exit(1)
    else:
        leagues = all_leagues
    log.info("Период: %s  |  Лиги: %d  |  Комбо: %d", period_cfg.label, len(leagues), len(COMBOS))

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
        log.info("── Лига: %-20s  строк: %d", league, len(subset))

        if subset.empty:
            log.warning("  Нет данных для лиги '%s' — пропуск.", league)
            continue

        # Split once per league; reuse across all combos (efficiency).
        train_df, test_dict = chrono_split(subset)
        test_df = pd.concat(test_dict.values(), ignore_index=True)

        for combo_name, enabled in COMBOS.items():
            done += 1
            log.info("  [%d/%d] %s × %s", done, total, league, combo_name)
            cell = run_cell(train_df, test_df, enabled, params, args.min_bets, period)
            if cell is None:
                log.info("  → пропущено")
                continue
            cell["league"] = league
            cell["combo"]  = combo_name
            records.append(cell)
            log.info(
                "  → LogLoss=%.4f  AUC=%.4f  best_roi=%+.2f%% @ thr=%s (%d ставок)",
                cell["log_loss"], cell["roc_auc"],
                cell["best_roi"] if not np.isnan(cell["best_roi"]) else 0.0,
                cell["best_thr"], cell["best_bets"],
            )

    if not records:
        log.error("Ни одна ячейка не дала результатов — проверьте данные и min_rows.")
        sys.exit(1)

    full_df   = _build_full_table(records, period)
    winner_df = _build_winner_table(full_df)

    period_note = (
        "" if period_cfg.has_real_line
        else f" (ROI vs. синтетическая медиана, не реальная букмекерская линия)"
    )
    _print_table(
        full_df,
        f"ПОЛНАЯ ТАБЛИЦА — все (лига × комбо)  [{period_cfg.label}{period_note}]",
    )
    _print_table(
        winner_df,
        f"ПОБЕДИТЕЛИ ПО ЛИГАМ [{period_cfg.label}]"
        " (→ вставить в feature_selector.py)",
    )

    csv_path = args.csv or _default_csv(period)
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out, index=False)
    log.info("Результаты сохранены → %s", out)


if __name__ == "__main__":
    main()
