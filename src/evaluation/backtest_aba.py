"""ABA League backtester — backwards-compatible shim over backtest_league.

Kept as a named entry-point because ABA is the documented pivot target
(see ``docs/postmortem_v1_v6.md`` V6.1 / V7 sections). Delegates entirely
to the generic ``backtest_league`` module to avoid duplication.

Run:
    python -m src.evaluation.backtest_aba
"""
from __future__ import annotations

import logging

from src.config import settings
from src.evaluation.backtest_league import build_simulation_params, run_league_backtest
from src.evaluation.reporting import print_threshold_table

logging.basicConfig(level=settings.app.log_level, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ABA_LEAGUE_KEY: str = "ABA"


def main() -> None:
    """Entry point: run the per-league backtest for ABA and print the report."""
    rows = run_league_backtest(ABA_LEAGUE_KEY)
    if rows is None:
        log.warning(
            "ABA pipeline returned no rows — likely below min_train_rows / "
            "min_test_rows guard. Adjust settings.evaluation.min_test_rows.",
        )
        return

    params = build_simulation_params()
    print_threshold_table(
        f"{ABA_LEAGUE_KEY} — per-league (Bootstrap CI95, {params.bootstrap_iters} iters):",
        rows,
    )


if __name__ == "__main__":
    main()
