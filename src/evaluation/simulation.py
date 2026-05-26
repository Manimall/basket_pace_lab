"""Bet outcome simulation and ROI statistics — pure numerical helpers.

This module is intentionally I/O-free: every function operates on numpy arrays
and primitive parameters. No DB calls, no model training, no printing. This
keeps the betting math fully testable without fixtures.

The module owns three pieces:
    * `BetReport` — aggregate statistics for one (threshold × subset) cell.
    * `SimulationParams` — frozen knob bundle passed top-down by orchestration.
    * Pure functions: simulate / bootstrap_roi_ci / aggregate / run_threshold_sweep.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BetReport:
    """Aggregate statistics for one threshold cell of a per-league sweep.

    Attributes:
        row_label: Identifier shown in the output table's first column.
        n: Number of bets placed by the threshold.
        wins: Bets where the picked side hit.
        losses: Bets where the opposite side hit.
        pushes: Bets where game_total exactly equalled the line (stake returned).
        profit: Sum of per-bet PnL with the configured flat stake.
        roi_ci_low: Bootstrap lower-percentile bound of ROI in %.
        roi_ci_high: Bootstrap upper-percentile bound of ROI in %.
    """
    row_label:   str
    n:           int
    wins:        int
    losses:      int
    pushes:      int
    profit:      float
    roi_ci_low:  float
    roi_ci_high: float

    @property
    def winrate(self) -> float:
        """Hit rate among decided bets (pushes excluded), in %.

        Returns:
            Percentage of wins out of wins+losses, or 0.0 if no decided bets.
        """
        decided = self.wins + self.losses
        return 100.0 * self.wins / decided if decided > 0 else 0.0

    @property
    def roi(self) -> float:
        """Profit as a percentage of placed-bet count (denominator = n).

        Returns:
            Profit divided by the number of placed bets, in %. 0.0 when n==0.
        """
        return 100.0 * self.profit / self.n if self.n > 0 else 0.0


@dataclass(frozen=True)
class SimulationParams:
    """Frozen bundle of all knobs the simulation layer needs from settings.

    Attributes:
        odds: Decimal odds offered on both sides (e.g. 1.90).
        stake: Flat stake amount per bet.
        thresholds: Probability thresholds to sweep, ascending recommended.
        bootstrap_iters: Number of bootstrap resamples for the ROI CI.
        bootstrap_seed: RNG seed for reproducible bootstrap.
        ci_alpha: Two-sided confidence level (0.05 → 95% CI).
    """
    odds:            float
    stake:           float
    thresholds:      tuple[float, ...]
    bootstrap_iters: int
    bootstrap_seed:  int
    ci_alpha:        float


def simulate(
    probs:     np.ndarray,
    actuals:   np.ndarray,
    lines:     np.ndarray,
    threshold: float,
    odds:      float,
    stake:     float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decide bets at a single threshold and compute per-row PnL.

    A bet is placed only when the model is at least `threshold` confident:
        prob >= threshold       → OVER
        prob <= 1 - threshold   → UNDER
        otherwise               → no bet

    Args:
        probs: P(OVER) predictions, shape (n,).
        actuals: Actual game_total values, shape (n,).
        lines: Bookmaker closing lines, shape (n,).
        threshold: Probability threshold in [0.5, 1.0).
        odds: Decimal odds offered on both sides (e.g. 1.90).
        stake: Flat stake amount per bet.

    Returns:
        A tuple ``(label, pnl)`` of two numpy arrays of length ``n``:
            * ``label`` — 1=win, 0=push, -1=loss, NaN=no-bet.
            * ``pnl`` — per-row profit/loss for placed bets, 0 elsewhere.
    """
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
    pnl[win]  = stake * (odds - 1.0)
    pnl[loss] = -stake

    label = np.full(len(probs), np.nan)
    label[win]  = 1
    label[push] = 0
    label[loss] = -1
    return label, pnl


def bootstrap_roi_ci(
    pnl:    np.ndarray,
    n_iter: int,
    seed:   int,
    alpha:  float,
) -> tuple[float, float]:
    """Bootstrap-resampled CI for ROI% over a placed-bets PnL array.

    ROI on flat 1u stakes equals ``mean(pnl) * 100``. We resample ``pnl`` with
    replacement ``n_iter`` times, then take the ``alpha/2`` and ``1-alpha/2``
    percentiles of the resampled means.

    Args:
        pnl: PnL of placed bets only (no-bet rows must be excluded).
        n_iter: Number of bootstrap resamples (e.g. 5000).
        seed: RNG seed for reproducibility.
        alpha: Two-sided confidence level (0.05 → 95% CI).

    Returns:
        Lower and upper bounds of the CI in percent. ``(NaN, NaN)`` when ``pnl``
        is empty.
    """
    n = len(pnl)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, n))
    boot_means = pnl[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100.0 * alpha / 2.0)) * 100.0
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0))) * 100.0
    return lo, hi


def aggregate(
    label:           np.ndarray,
    pnl:             np.ndarray,
    row_label:       str,
    bootstrap_iters: int,
    bootstrap_seed:  int,
    ci_alpha:        float,
) -> BetReport:
    """Reduce per-row simulation arrays into a single BetReport with bootstrap CI.

    Args:
        label: Per-row outcome labels from ``simulate``.
        pnl: Per-row PnL from ``simulate``.
        row_label: Identifier for the resulting report's first column.
        bootstrap_iters: Forwarded to ``bootstrap_roi_ci``.
        bootstrap_seed: Forwarded to ``bootstrap_roi_ci``.
        ci_alpha: Forwarded to ``bootstrap_roi_ci``.

    Returns:
        A BetReport with counts, profit, and bootstrap CI populated.
    """
    placed     = ~np.isnan(label)
    placed_pnl = pnl[placed]
    lo, hi = bootstrap_roi_ci(placed_pnl, bootstrap_iters, bootstrap_seed, ci_alpha)
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
    probs:   np.ndarray,
    actuals: np.ndarray,
    lines:   np.ndarray,
    params:  SimulationParams,
) -> list[BetReport]:
    """Run simulate + aggregate for each threshold in ``params.thresholds``.

    Args:
        probs: P(OVER) predictions on the test set.
        actuals: Actual game totals.
        lines: Bookmaker closing lines.
        params: All simulation knobs (odds, stake, thresholds, bootstrap …).

    Returns:
        One ``BetReport`` per threshold, in the order of ``params.thresholds``.
    """
    return [
        aggregate(
            *simulate(probs, actuals, lines, t, params.odds, params.stake),
            row_label       = f"{t:.2f}",
            bootstrap_iters = params.bootstrap_iters,
            bootstrap_seed  = params.bootstrap_seed,
            ci_alpha        = params.ci_alpha,
        )
        for t in params.thresholds
    ]
