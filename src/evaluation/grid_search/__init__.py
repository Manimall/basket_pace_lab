"""Grid-search package: per-league × feature-combo backtester sweep.

Public surface re-exported for the thin CLI in ``scripts/grid_search_leagues.py``
and for unit tests. Core logic lives in the submodules so each file stays well
under the 250-line limit and the heavy pieces are independently testable.
"""
from __future__ import annotations

from src.evaluation.grid_search.cell import GridCellResult, run_cell
from src.evaluation.grid_search.dataset import prepare_dataset
from src.evaluation.grid_search.periods import (
    COMBOS,
    GRID_THRESHOLDS,
    PERIOD_CONFIGS,
    Period,
    PeriodConfig,
)
from src.evaluation.grid_search.tables import (
    build_full_table,
    build_winner_table,
    render_table,
)

__all__ = [
    "COMBOS",
    "GRID_THRESHOLDS",
    "PERIOD_CONFIGS",
    "Period",
    "PeriodConfig",
    "GridCellResult",
    "run_cell",
    "prepare_dataset",
    "build_full_table",
    "build_winner_table",
    "render_table",
]
