"""Tests for the pure logic of src.evaluation.grid_search.

I/O-free: no DB, no CatBoost. Covers feature exclusion, best-ROI selection,
leak-free synthetic-line derivation, and winner-table tie-breaking.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.evaluation.config import BIN_TARGET, LINE_COL
from src.evaluation.feature_selector import FeatureGroup
from src.evaluation.grid_search.cell import (
    GridCellResult,
    best_roi_from_reports,
    derive_bin_target_from_median,
    excluded_for_combo,
)
from src.evaluation.grid_search.periods import COMBOS, GRID_THRESHOLDS, PERIOD_CONFIGS
from src.evaluation.grid_search.tables import build_full_table, build_winner_table
from src.features.fatigue import FATIGUE_FEATURE_COLS
from src.features.team_advanced import TEAM_ADV_FEAT_COLS
from src.evaluation.simulation import BetReport


# ── periods config ────────────────────────────────────────────────────────────

def test_all_three_periods_present() -> None:
    assert set(PERIOD_CONFIGS) == {"game", "1q", "1h"}


def test_only_game_has_real_line() -> None:
    assert PERIOD_CONFIGS["game"].has_real_line is True
    assert PERIOD_CONFIGS["1q"].has_real_line is False
    assert PERIOD_CONFIGS["1h"].has_real_line is False


def test_combos_all_contain_base() -> None:
    for groups in COMBOS.values():
        assert FeatureGroup.BASE in groups


# ── excluded_for_combo ────────────────────────────────────────────────────────

def test_base_combo_excludes_fatigue_and_team_adv() -> None:
    excluded = excluded_for_combo(COMBOS["BASE"])
    for col in FATIGUE_FEATURE_COLS:
        assert col in excluded
    for col in TEAM_ADV_FEAT_COLS:
        assert col in excluded


def test_full_combo_keeps_fatigue_and_team_adv() -> None:
    excluded = excluded_for_combo(COMBOS["BASE+FATIGUE+TEAM_ADV"])
    for col in FATIGUE_FEATURE_COLS:
        assert col not in excluded
    for col in TEAM_ADV_FEAT_COLS:
        assert col not in excluded


# ── best_roi_from_reports ─────────────────────────────────────────────────────

def _report(label: str, roi: float, n: int) -> BetReport:
    # profit chosen so BetReport.roi == roi for the given n (roi = 100*profit/n)
    profit = roi * n / 100.0
    return BetReport(
        row_label=label, n=n, wins=0, losses=0, pushes=0,
        profit=profit, roi_ci_low=float("nan"), roi_ci_high=float("nan"),
    )


def test_best_roi_picks_highest_qualifying() -> None:
    reports = [_report("0.54", 5.0, 50), _report("0.56", 12.0, 40), _report("0.58", 8.0, 30)]
    roi, thr, bets = best_roi_from_reports(reports, min_bets=10)
    assert thr == "0.56"
    assert roi == 12.0
    assert bets == 40


def test_best_roi_skips_below_min_bets() -> None:
    reports = [_report("0.54", 5.0, 50), _report("0.56", 99.0, 3)]
    roi, thr, bets = best_roi_from_reports(reports, min_bets=10)
    assert thr == "0.54"
    assert bets == 50


def test_best_roi_returns_nan_when_none_qualify() -> None:
    reports = [_report("0.54", 5.0, 2)]
    roi, thr, bets = best_roi_from_reports(reports, min_bets=10)
    assert math.isnan(roi)
    assert thr == "—"
    assert bets == 0


# ── derive_bin_target_from_median (leak-free) ─────────────────────────────────

def test_synthetic_line_uses_training_median_only() -> None:
    train = pd.DataFrame({"q1_total": [40, 42, 44, 46]})   # median = 43
    test  = pd.DataFrame({"q1_total": [100, 10]})           # must NOT shift the line
    train_out, test_out, line = derive_bin_target_from_median(train, test, "q1_total")

    assert line == 43.0
    # test labels derived against the *training* median, not the test's own
    assert test_out[BIN_TARGET].tolist() == [1, 0]
    assert (test_out[LINE_COL] == 43.0).all()
    assert (train_out[LINE_COL] == 43.0).all()


def test_derive_does_not_mutate_inputs() -> None:
    train = pd.DataFrame({"q1_total": [40, 42, 44, 46]})
    test  = pd.DataFrame({"q1_total": [100, 10]})
    derive_bin_target_from_median(train, test, "q1_total")
    assert BIN_TARGET not in train.columns
    assert BIN_TARGET not in test.columns


# ── winner table tie-break ────────────────────────────────────────────────────

def _cell(league: str, combo: str, best_roi: float, best_bets: int) -> GridCellResult:
    return GridCellResult(
        n_train=200, n_test=50, n_features=58, log_loss=0.8, roc_auc=0.6,
        best_roi=best_roi, best_thr="0.56", best_bets=best_bets,
        league=league, combo=combo,
    )


def test_winner_table_breaks_tie_by_bets() -> None:
    cells = [
        _cell("NBA", "BASE", 10.0, 30),
        _cell("NBA", "BASE+FATIGUE", 10.0, 90),  # same ROI, more bets → wins
    ]
    full = build_full_table(cells, period="game")
    winners = build_winner_table(full)
    assert len(winners) == 1
    assert winners.iloc[0]["winner_combo"] == "BASE+FATIGUE"
    assert winners.iloc[0]["best_bets"] == 90


def test_build_full_table_empty_returns_empty_frame() -> None:
    assert build_full_table([], period="game").empty


def test_full_table_includes_synthetic_line_for_subgame() -> None:
    cell = GridCellResult(
        n_train=200, n_test=50, n_features=58, log_loss=0.8, roc_auc=0.6,
        best_roi=5.0, best_thr="0.56", best_bets=40,
        league="NBA", combo="BASE", synthetic_line=42.5,
        roi_by_threshold={f"{t:.2f}": 1.0 for t in GRID_THRESHOLDS},
        bets_by_threshold={f"{t:.2f}": 40 for t in GRID_THRESHOLDS},
    )
    full = build_full_table([cell], period="1q")
    assert "synthetic_line" in full.columns
    assert full.iloc[0]["synthetic_line"] == 42.5
