"""Static configuration for the grid search: target periods and feature combos.

Pure data — no pandas, no DB, no model. Imported by every other grid_search
submodule and by the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.evaluation.feature_selector import FeatureGroup

Period = Literal["game", "1q", "1h"]


@dataclass(frozen=True)
class PeriodConfig:
    """Configuration for one target period.

    Attributes:
        period:        Identifier passed via the --period flag.
        target_col:    Column name of the raw score total to predict.
        label:         Human-readable label for table headers.
        has_real_line: True only for "game", where a real bookmaker closing
                       line exists. Sub-game periods derive a synthetic line.
    """

    period:        Period
    target_col:    str
    label:         str
    has_real_line: bool


PERIOD_CONFIGS: dict[Period, PeriodConfig] = {
    "game": PeriodConfig("game", "game_total", "GAME TOTAL",   has_real_line=True),
    "1q":   PeriodConfig("1q",   "q1_total",   "1-я ЧЕТВЕРТЬ", has_real_line=False),
    "1h":   PeriodConfig("1h",   "h1_total",   "1-я ПОЛОВИНА", has_real_line=False),
}

# Feature combinations swept per league. BASE is always present.
COMBOS: dict[str, frozenset[FeatureGroup]] = {
    "BASE":                  frozenset({FeatureGroup.BASE}),
    "BASE+FATIGUE":          frozenset({FeatureGroup.BASE, FeatureGroup.FATIGUE}),
    "BASE+TEAM_ADV":         frozenset({FeatureGroup.BASE, FeatureGroup.TEAM_ADV}),
    "BASE+FATIGUE+TEAM_ADV": frozenset({
        FeatureGroup.BASE, FeatureGroup.FATIGUE, FeatureGroup.TEAM_ADV,
    }),
}

# Probability thresholds evaluated by the sweep (the "working" confidence band).
GRID_THRESHOLDS: tuple[float, ...] = (0.54, 0.56, 0.58)
