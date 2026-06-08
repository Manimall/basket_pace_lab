"""Tests for seasonal sample weights (docs/betting_seasonality.md).

I/O-free: pure month→weight mapping. Covers each tier, the offseason fallback,
vectorised date→weight mapping, and the disjoint-tier invariant.
"""
from __future__ import annotations

import pandas as pd

from src.evaluation.seasonality import (
    ANOMALY,
    DEFAULT_WEIGHT,
    GOLDEN_WINDOW,
    NOISE,
    SEASON_TIERS,
    seasonal_sample_weights,
    weight_for_month,
)

# ── per-month weights ─────────────────────────────────────────────────────────

def test_golden_window_months_weight_one() -> None:
    for month in (12, 1, 2, 3):
        assert weight_for_month(month) == 1.0


def test_noise_months_weight_half() -> None:
    for month in (10, 11):
        assert weight_for_month(month) == 0.5


def test_anomaly_months_weight_low() -> None:
    for month in (4, 5, 6):
        assert weight_for_month(month) == 0.2


def test_offseason_months_fall_back_to_noise() -> None:
    """Jul–Sep are not in any tier → DEFAULT_WEIGHT (noise), never dropped."""
    for month in (7, 8, 9):
        assert weight_for_month(month) == DEFAULT_WEIGHT
        assert weight_for_month(month) == NOISE.weight


# ── tier invariants ───────────────────────────────────────────────────────────

def test_tiers_are_disjoint() -> None:
    """No month belongs to two tiers (lookup order must not matter)."""
    seen: set[int] = set()
    for tier in SEASON_TIERS:
        assert not (seen & tier.months), f"overlap in {tier.label}"
        seen |= tier.months


def test_tier_weights_strictly_ordered() -> None:
    assert GOLDEN_WINDOW.weight > NOISE.weight > ANOMALY.weight


# ── vectorised mapping ────────────────────────────────────────────────────────

def test_seasonal_sample_weights_maps_dates() -> None:
    dates = pd.to_datetime(
        ["2026-01-15", "2025-10-20", "2026-05-03"], utc=True
    )
    weights = seasonal_sample_weights(pd.Series(dates))
    assert list(weights) == [1.0, 0.5, 0.2]


def test_seasonal_sample_weights_length_matches_input() -> None:
    dates = pd.Series(pd.to_datetime(["2026-02-01"] * 7, utc=True))
    weights = seasonal_sample_weights(dates)
    assert len(weights) == 7
    assert weights.dtype == float
