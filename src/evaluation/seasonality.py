"""Seasonal sample weights for V6 training.

Encodes the seasonality thesis from ``docs/betting_seasonality.md``: matches
played in the Dec–Mar "golden window" carry the cleanest pace signal, Oct–Nov is
early-season noise, and Apr–Jun (end of regular season + playoffs) is anomalous.
Training rows are weighted accordingly so the model trusts golden-window matches
most; the test split is never weighted (honest, unbiased evaluation).

Pure and typed: the tier definitions live here as named constants (zero hardcode
at call sites — callers only import :func:`seasonal_sample_weights`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Match-date column carried through the feature pipeline (set in
# score_features.load_data); used to derive each row's calendar month.
DATE_COLUMN: str = "scheduled_at"


@dataclass(frozen=True)
class SeasonTier:
    """One seasonality tier: a set of calendar months and their training weight.

    Attributes:
        months: Calendar month numbers (1–12) belonging to this tier.
        weight: Training ``sample_weight`` applied to matches in these months.
        label: Short human-readable name for logging.
    """

    months: frozenset[int]
    weight: float
    label:  str


# Tiers per docs/betting_seasonality.md (newest-priority first is irrelevant —
# month sets are disjoint, so lookup order does not matter).
GOLDEN_WINDOW: SeasonTier = SeasonTier(frozenset({12, 1, 2, 3}), 1.0, "golden")
NOISE:         SeasonTier = SeasonTier(frozenset({10, 11}),      0.5, "noise")
ANOMALY:       SeasonTier = SeasonTier(frozenset({4, 5, 6}),     0.2, "anomaly")
SEASON_TIERS:  tuple[SeasonTier, ...] = (GOLDEN_WINDOW, NOISE, ANOMALY)

# Months outside the listed tiers (Jul–Sep preseason/offseason — e.g. a late-
# September season opener). Treated as early-season noise: down-weighted, never
# dropped, so a stray fixture still contributes a little.
DEFAULT_WEIGHT: float = NOISE.weight


def weight_for_month(month: int) -> float:
    """Return the training weight for a calendar month (1–12)."""
    for tier in SEASON_TIERS:
        if month in tier.months:
            return tier.weight
    return DEFAULT_WEIGHT


def seasonal_sample_weights(dates: pd.Series) -> np.ndarray:
    """Map each match date to its seasonal training weight.

    Args:
        dates: Match timestamps (e.g. ``train_df[DATE_COLUMN]``).

    Returns:
        Float array aligned with ``dates``: 1.0 golden / 0.5 noise / 0.2 anomaly
        (see :data:`SEASON_TIERS`), per ``docs/betting_seasonality.md``.
    """
    months = pd.to_datetime(dates, utc=True).dt.month
    return months.map(weight_for_month).to_numpy(dtype=float)


def partition_by_months(
    df: pd.DataFrame, months: frozenset[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (train, test) where test = rows whose month ∈ ``months``.

    Used by the seasonality window probe to deliberately hold out a calendar
    window (e.g. the Dec–Mar golden window) as the test set and train on the
    rest. NOTE: this is a regime diagnostic, not a time-honest backtest — the
    train side may contain matches dated AFTER the test window (future leakage).

    Args:
        df: Feature-built rows carrying :data:`DATE_COLUMN`.
        months: Calendar months (1–12) that define the held-out test window.

    Returns:
        ``(train_df, test_df)``, both reset-index; ``test_df`` holds the window.
    """
    in_window = pd.to_datetime(df[DATE_COLUMN], utc=True).dt.month.isin(months)
    train_df = df[~in_window].reset_index(drop=True)
    test_df  = df[in_window].reset_index(drop=True)
    return train_df, test_df
