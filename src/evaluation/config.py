"""Pipeline definitions and column-name constants for the backtester.

This module is intentionally tiny and pure: it holds the static configuration
shared between the orchestration layer (`backtester.py`) and the data-prep
helpers, without pulling in heavy dependencies (no pandas, no DB).

The values defined here are code-level configuration — they describe the
experiment shape (which league subsets to compare, what column names to use).
Env-overridable runtime knobs (odds, bootstrap iters, …) live in
`src.config.settings.EvaluationConfig` instead.
"""
from __future__ import annotations

from dataclasses import dataclass

# Names of the columns produced by `src.features.score_features.build_features`.
LINE_COL:   str = "bookmaker_total_closing"   # closing O/U line
BIN_TARGET: str = "over_hit"                  # 1 iff game_total > line


@dataclass(frozen=True)
class PipelineSpec:
    """Specification for one independent per-league backtester run.

    Attributes:
        name: Human-readable identifier shown in the output table title.
        include: League names to include. Empty tuple = include all leagues.
        exclude: League names to remove after `include`. Useful for building
            "everything except X, Y" control groups.
    """
    name:    str
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


# Default per-league pipelines for V6. Order is preserved in the output.
PIPELINES: tuple[PipelineSpec, ...] = (
    PipelineSpec(name="NBA",             include=("NBA",)),
    PipelineSpec(name="EuroLeague",      include=("EuroLeague",)),
    PipelineSpec(name="OTHER (control)", exclude=("NBA", "EuroLeague")),
)
