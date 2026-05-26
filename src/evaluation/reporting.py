"""Console output formatting for backtester reports.

Single responsibility: turn ``BetReport`` lists into a readable table on
stdout. No business logic, no I/O other than printing.

We use ``print`` here on purpose: a formatted table is presentation output,
not a log event. Log records are emitted by the orchestration layer for
progress, warnings, and errors.
"""
from __future__ import annotations

import logging
import math

from src.evaluation.simulation import BetReport

log = logging.getLogger(__name__)

_TABLE_BORDER_CHAR = "─"
_CI_PLACEHOLDER    = f"{'—':>18s}"
_FLAG_NONE         = " "
_FLAG_LOSS         = "✗"
_FLAG_WIN          = "✓"
_FLAG_SIGNIFICANT  = "★"


def format_roi_ci(low: float, high: float) -> str:
    """Render a CI95 cell.

    Args:
        low: Lower bound of the CI in %.
        high: Upper bound of the CI in %.

    Returns:
        ``"[+x.xx%; +y.yy%]"`` when both bounds are finite, an em-dash
        placeholder otherwise.
    """
    if not (math.isfinite(low) and math.isfinite(high)):
        return _CI_PLACEHOLDER
    return f"[{low:+6.2f}%; {high:+6.2f}%]"


def _row_flag(r: BetReport) -> str:
    """Return the right-most flag symbol for one report row.

    Priority order: significance star > positive profit ✓ > loss ✗ > empty.
    """
    if r.n == 0:
        return _FLAG_NONE
    if r.roi_ci_low > 0:
        return _FLAG_SIGNIFICANT
    return _FLAG_WIN if r.profit > 0 else _FLAG_LOSS


def print_threshold_table(title: str, rows: list[BetReport]) -> None:
    """Print a per-threshold report table to stdout.

    Args:
        title: Header text shown immediately above the table.
        rows: One ``BetReport`` per threshold row, in display order.
    """
    hdr = (
        f"{'Thr':>5s} {'Bets':>5s} {'W':>4s} {'L':>4s} {'P':>3s} "
        f"{'Winrate':>8s} {'ROI':>8s} {'Profit(u)':>10s}  {'ROI CI95':>18s}"
    )
    sep = _TABLE_BORDER_CHAR * len(hdr)
    print(f"\n{title}")
    print(f"{sep}\n{hdr}\n{sep}")
    for r in rows:
        flag = _row_flag(r)
        wr   = f"{r.winrate:>6.2f}%" if (r.wins + r.losses) > 0 else f"{'—':>7s}"
        roi  = f"{r.roi:>+6.2f}%"    if r.n > 0 else f"{'—':>7s}"
        ci   = format_roi_ci(r.roi_ci_low, r.roi_ci_high)
        print(
            f"{r.row_label:>5s} {r.n:>5d} {r.wins:>4d} {r.losses:>4d} {r.pushes:>3d} "
            f"{wr:>8s} {roi:>8s} {r.profit:>+10.2f}  {ci:>18s} {flag}"
        )
    print(sep)
    print(f"  {_FLAG_SIGNIFICANT} = CI95 lower bound > 0 "
          "(statistically distinguishable from zero)\n")
