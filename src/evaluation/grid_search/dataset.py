"""Dataset preparation for the grid search.

Loads matches + quarter_stats once, builds features, and derives the binary
over/under target for the requested period. For "game" the real bookmaker
closing line is used; sub-game periods leave BIN_TARGET unset (each cell
derives a leak-free synthetic line from its own training median).
"""
from __future__ import annotations

import asyncio
import logging

import pandas as pd

from src.evaluation.config import BIN_TARGET, LINE_COL
from src.evaluation.grid_search.periods import PERIOD_CONFIGS, Period
from src.features.score_features import build_features, load_data

log = logging.getLogger(__name__)

_PCT: float = 100.0


def prepare_dataset(period: Period) -> pd.DataFrame:
    """Load, feature-build, and derive the period target.

    Args:
        period: Target period — "game", "1q", or "1h".

    Returns:
        Feature DataFrame with the period target column populated. BIN_TARGET
        is set only for "game"; sub-game periods set it per-cell from the median.
    """
    cfg = PERIOD_CONFIGS[period]
    log.info("Загрузка матчей + quarter_stats из БД…")
    matches, qs = asyncio.run(load_data())

    log.info("Построение фичей…")
    df = build_features(matches, qs)

    before = len(df)
    df = df.dropna(subset=[cfg.target_col]).reset_index(drop=True)
    if len(df) < before:
        log.info("  Удалено %d строк без %s", before - len(df), cfg.target_col)
    log.info("  Строк после фильтрации: %d", len(df))

    if period == "game":
        return _apply_real_line(df, cfg.target_col)

    log.info(
        "  Период '%s': нет реальных линий букмекеров. "
        "ROI считается относительно медианы тренировочного сета.",
        period,
    )
    return df


def _apply_real_line(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Drop line-less + push rows and set BIN_TARGET from the real closing line."""
    df = df.dropna(subset=[LINE_COL]).reset_index(drop=True)
    pushes = int((df[target_col] == df[LINE_COL]).sum())
    if pushes:
        log.info("  Удаляю %d push-строк (total == line)", pushes)
        df = df[df[target_col] != df[LINE_COL]].reset_index(drop=True)
    df[BIN_TARGET] = (df[target_col] > df[LINE_COL]).astype(int)
    log.info("  OVER rate: %.2f%%  (линия: %s)", _PCT * df[BIN_TARGET].mean(), LINE_COL)
    return df
