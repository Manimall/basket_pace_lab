"""Tests for chrono_split_3way (blind-threshold validation split).

I/O-free: pure chronological slicing. Covers order, fraction sizing, the
disjoint-and-complete invariant, and that the split is time-ordered (train is
oldest, test is newest) — the property that makes blind validation honest.
"""
from __future__ import annotations

import pandas as pd

from src.models.evaluation import chrono_split_3way


def _dated_df(n: int) -> pd.DataFrame:
    """n rows with strictly increasing dates, shuffled to prove sort happens."""
    dates = pd.date_range("2025-10-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({"scheduled_at": dates, "id": list(range(n))})
    return df.sample(frac=1.0, random_state=1).reset_index(drop=True)


def test_fractions_size_the_parts() -> None:
    train, val, test = chrono_split_3way(_dated_df(100), 0.6, 0.2)
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20


def test_split_is_disjoint_and_complete() -> None:
    df = _dated_df(50)
    train, val, test = chrono_split_3way(df, 0.6, 0.2)
    ids = set(train["id"]) | set(val["id"]) | set(test["id"])
    assert len(train) + len(val) + len(test) == len(df)
    assert ids == set(df["id"])


def test_parts_are_chronologically_ordered() -> None:
    """train oldest, test newest — no future leakage across parts."""
    train, val, test = chrono_split_3way(_dated_df(60), 0.6, 0.2)
    assert train["scheduled_at"].max() <= val["scheduled_at"].min()
    assert val["scheduled_at"].max() <= test["scheduled_at"].min()
