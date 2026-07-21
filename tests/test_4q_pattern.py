"""Tests for the 3-Over → Q4-Under backtest logic (scripts/test_4q_pattern.py).

Pure-function coverage, no DB:
  * _profit_units — win/push/loss economics
  * apply_filters — score-corridor and fatigue (Q3 < Q1) gates
  * evaluate_shift — conditional bets, base-rate, lift, push handling

Deterministic fixture (total_line=100 → quarter proxy line = 25):
  A: q1..q3 = 30 (all Over), q4=20  → condition True,  Q4 Under (win)
  B: q1..q3 = 30 (all Over), q4=40  → condition True,  Q4 Over  (loss)
  C: q1=10,q2=q3=30          → condition False (q1),   q4=10
  D: q1..q3 = 26 (all Over), q4=25  → condition True,  q4 == line (push at shift=0)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Load the script module directly (scripts/ is not a package). Register it in
# sys.modules under a unique name BEFORE exec so @dataclass can resolve its
# module (dataclasses looks the class module up in sys.modules).
_SPEC = importlib.util.spec_from_file_location(
    "q4_pattern_script",
    Path(__file__).resolve().parent.parent / "scripts" / "test_4q_pattern.py",
)
assert _SPEC and _SPEC.loader
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


def _fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "match_id":         [1, 2, 3, 4],
        "scheduled_at":     pd.to_datetime(
            ["2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04"], utc=True),
        "season":           ["2025-26"] * 4,
        "league":           ["CBA"] * 4,
        "total_line":       [100.0, 100.0, 100.0, 100.0],
        "q1":               [30, 30, 10, 26],
        "q2":               [30, 30, 30, 26],
        "q3":               [30, 30, 30, 26],
        "q4":               [20, 40, 10, 25],
        "margin_before_q4": [5, 15, 25, 12],
    })


def test_profit_units_win_push_loss() -> None:
    win  = pd.Series([True, False, False])
    push = pd.Series([False, False, True])
    profit = mod._profit_units(win, push, odds=1.90)
    assert profit.tolist() == pytest.approx([0.90, -1.0, 0.0])


def test_apply_filters_margin_corridor() -> None:
    out = mod.apply_filters(_fixture(), margin=(10.0, 18.0), fatigue=False)
    # Only B (15) and D (12) fall inside [10, 18].
    assert sorted(out["match_id"].tolist()) == [2, 4]


def test_apply_filters_fatigue_keeps_q3_below_q1() -> None:
    out = mod.apply_filters(_fixture(), margin=None, fatigue=True)
    # q3 < q1 nowhere in the fixture (all q3 >= q1) → empty.
    assert out.empty


def test_evaluate_shift_counts_and_lift_at_zero_shift() -> None:
    r = mod.evaluate_shift(_fixture(), shift=0.0, odds=1.90)
    # Condition (q1,q2,q3 all > 25) holds for A, B, D → 3 bets.
    assert r.n_bets == 3
    # Wins among bets: A only (q4=20<25). B loss, D push → win_rate 1/3.
    assert r.win_rate == pytest.approx(100 / 3, abs=1e-6)
    # Unconditional Q4 Under at line 25: A(20) and C(10) → 2/4 = 50%.
    assert r.base_under == pytest.approx(50.0)
    # Lift = conditional − base.
    assert r.lift == pytest.approx(100 / 3 - 50.0, abs=1e-6)
    # ROI = (+0.90 − 1.0 + 0.0)/3 → negative here (no edge in this toy set).
    assert r.roi == pytest.approx((0.90 - 1.0) / 3 * 100, abs=1e-6)
    assert np.isfinite(r.roi_ci_low) and np.isfinite(r.roi_ci_high)


def test_evaluate_shift_no_bets_returns_nan() -> None:
    # Impossible corridor → no matches, evaluate must not crash.
    empty = _fixture().iloc[0:0]
    r = mod.evaluate_shift(empty, shift=0.0, odds=1.90)
    assert r.n_bets == 0
    assert np.isnan(r.roi)
