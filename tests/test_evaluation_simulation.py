"""
Tests for pure betting-math helpers in src.evaluation.simulation.

These tests are I/O-free: no DB, no CatBoost. They cover the contract of
simulate / bootstrap_roi_ci / aggregate / run_threshold_sweep and the
BetReport derived properties.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.simulation import (
    BetReport,
    SimulationParams,
    aggregate,
    bootstrap_roi_ci,
    run_threshold_sweep,
    simulate,
)

_ODDS  = 1.90
_STAKE = 1.0
_WIN   = _STAKE * (_ODDS - 1.0)
_LOSS  = -_STAKE


def _params(thresholds: tuple[float, ...] = (0.50,)) -> SimulationParams:
    """Build a SimulationParams with deterministic bootstrap settings."""
    return SimulationParams(
        odds            = _ODDS,
        stake           = _STAKE,
        thresholds      = thresholds,
        bootstrap_iters = 200,
        bootstrap_seed  = 0,
        ci_alpha        = 0.05,
    )


# ── simulate ──────────────────────────────────────────────────────────────────


def test_simulate_handles_over_win_under_win_loss_push_and_skip():
    """One bet of each outcome class on a high threshold."""
    # threshold=0.60 ⇒ OVER if prob>=0.60, UNDER if prob<=0.40, else no bet
    probs   = np.array([0.70, 0.30, 0.70, 0.70, 0.50])
    lines   = np.array([200.0, 200.0, 200.0, 200.0, 200.0])
    actuals = np.array([210.0, 190.0, 190.0, 200.0, 210.0])

    label, pnl = simulate(probs, actuals, lines, threshold=0.60, odds=_ODDS, stake=_STAKE)

    assert label[0] == 1               # OVER bet, OVER hit
    assert label[1] == 1               # UNDER bet, UNDER hit
    assert label[2] == -1              # OVER bet, UNDER hit
    assert label[3] == 0               # OVER bet, exact push
    assert math.isnan(label[4])        # no-bet zone

    assert pnl[0] == pytest.approx(_WIN)
    assert pnl[1] == pytest.approx(_WIN)
    assert pnl[2] == pytest.approx(_LOSS)
    assert pnl[3] == 0.0
    assert pnl[4] == 0.0


def test_simulate_no_bet_window_inside_threshold():
    """Probs strictly inside the deadband must never trigger a bet."""
    probs   = np.array([0.51, 0.49, 0.55, 0.45])
    actuals = np.array([200.0, 200.0, 200.0, 200.0])
    lines   = np.array([200.0, 200.0, 200.0, 200.0])

    label, pnl = simulate(probs, actuals, lines, threshold=0.60, odds=_ODDS, stake=_STAKE)
    assert np.all(np.isnan(label))
    assert np.all(pnl == 0.0)


def test_simulate_threshold_boundary_inclusive_on_over_and_under():
    """prob == threshold is OVER; prob == 1-threshold is UNDER."""
    label, _ = simulate(
        probs   = np.array([0.60, 0.40]),
        actuals = np.array([210.0, 190.0]),
        lines   = np.array([200.0, 200.0]),
        threshold=0.60, odds=_ODDS, stake=_STAKE,
    )
    assert label[0] == 1  # OVER triggered at the boundary, OVER hit
    assert label[1] == 1  # UNDER triggered at the boundary, UNDER hit


# ── bootstrap_roi_ci ──────────────────────────────────────────────────────────


def test_bootstrap_roi_ci_returns_nan_for_empty_input():
    lo, hi = bootstrap_roi_ci(np.array([]), n_iter=100, seed=0, alpha=0.05)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_bootstrap_roi_ci_is_deterministic_for_same_seed():
    pnl = np.array([_WIN, _LOSS, _WIN, _LOSS, _WIN])
    a = bootstrap_roi_ci(pnl, n_iter=500, seed=42, alpha=0.05)
    b = bootstrap_roi_ci(pnl, n_iter=500, seed=42, alpha=0.05)
    assert a == b


def test_bootstrap_roi_ci_lower_bound_below_upper():
    pnl = np.array([_WIN, _LOSS, _WIN, _LOSS, _WIN, _WIN, _LOSS])
    lo, hi = bootstrap_roi_ci(pnl, n_iter=1000, seed=42, alpha=0.05)
    assert lo < hi


def test_bootstrap_roi_ci_excludes_zero_for_all_winning_array():
    """An all-win PnL must have a CI strictly above 0 even with few samples."""
    pnl = np.full(20, _WIN)
    lo, hi = bootstrap_roi_ci(pnl, n_iter=1000, seed=42, alpha=0.05)
    assert lo > 0
    assert hi > 0


# ── aggregate ─────────────────────────────────────────────────────────────────


def test_aggregate_counts_wins_losses_pushes_and_profit():
    label = np.array([1.0, -1.0, 0.0, 1.0, np.nan])
    pnl   = np.array([_WIN, _LOSS, 0.0, _WIN, 0.0])
    r = aggregate(
        label, pnl, row_label="x",
        bootstrap_iters=100, bootstrap_seed=0, ci_alpha=0.05,
    )
    assert r.n      == 4
    assert r.wins   == 2
    assert r.losses == 1
    assert r.pushes == 1
    assert r.profit == pytest.approx(2 * _WIN + _LOSS)


def test_aggregate_handles_all_no_bet_rows():
    label = np.array([np.nan, np.nan])
    pnl   = np.array([0.0, 0.0])
    r = aggregate(
        label, pnl, row_label="empty",
        bootstrap_iters=10, bootstrap_seed=0, ci_alpha=0.05,
    )
    assert r.n == 0
    assert r.profit == 0.0
    assert math.isnan(r.roi_ci_low)
    assert math.isnan(r.roi_ci_high)


# ── BetReport properties ──────────────────────────────────────────────────────


def test_betreport_winrate_excludes_pushes_from_denominator():
    r = BetReport(
        row_label="x", n=8, wins=2, losses=1, pushes=5,
        profit=0.0, roi_ci_low=0.0, roi_ci_high=0.0,
    )
    # decided = 2 + 1 = 3, wins / decided = 2/3
    assert r.winrate == pytest.approx(2.0 / 3.0 * 100.0)


def test_betreport_winrate_zero_when_no_decided_bets():
    r = BetReport(
        row_label="x", n=3, wins=0, losses=0, pushes=3,
        profit=0.0, roi_ci_low=0.0, roi_ci_high=0.0,
    )
    assert r.winrate == 0.0


def test_betreport_roi_uses_n_as_denominator():
    r = BetReport(
        row_label="x", n=10, wins=5, losses=5, pushes=0,
        profit=2.0, roi_ci_low=0.0, roi_ci_high=0.0,
    )
    assert r.roi == pytest.approx(20.0)


def test_betreport_roi_zero_when_no_bets_placed():
    r = BetReport(
        row_label="x", n=0, wins=0, losses=0, pushes=0,
        profit=0.0, roi_ci_low=0.0, roi_ci_high=0.0,
    )
    assert r.roi == 0.0


# ── run_threshold_sweep ───────────────────────────────────────────────────────


def test_run_threshold_sweep_returns_one_report_per_threshold():
    probs   = np.array([0.70, 0.30, 0.60, 0.40])
    actuals = np.array([210.0, 190.0, 210.0, 190.0])
    lines   = np.array([200.0, 200.0, 200.0, 200.0])
    params = _params(thresholds=(0.50, 0.55, 0.60))

    rows = run_threshold_sweep(probs, actuals, lines, params)

    assert [r.row_label for r in rows] == ["0.50", "0.55", "0.60"]
    assert all(isinstance(r, BetReport) for r in rows)


def test_run_threshold_sweep_higher_threshold_places_fewer_bets():
    rng = np.random.default_rng(0)
    probs   = rng.uniform(0.3, 0.7, size=200)
    actuals = rng.normal(220.0, 10.0, size=200)
    lines   = np.full(200, 220.0)
    params = _params(thresholds=(0.50, 0.60))

    rows = run_threshold_sweep(probs, actuals, lines, params)
    assert rows[1].n <= rows[0].n
