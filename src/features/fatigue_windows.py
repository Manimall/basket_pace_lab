"""Per-team appearance-timeline computations for schedule-fatigue features.

Pure pandas helpers extracted from ``fatigue.py`` to keep each module under the
250-line limit. No settings, no DB — every function takes an explicit
appearance timeline and returns an index-aligned Series.
"""
from __future__ import annotations

import pandas as pd

# Sentinel for "no previous game" — any gap above off_season_max_days resets state.
NO_PREV_GAME_DAYS: int = 10**6

# Number of days that defines a back-to-back (previous game exactly yesterday).
_B2B_GAP_DAYS: int = 1


def build_team_appearances(matches: pd.DataFrame) -> pd.DataFrame:
    """Stack home + away rows into one team-appearance timeline.

    Args:
        matches: Match-level DataFrame; must contain ``match_id``,
            ``scheduled_at``, ``home_team_id``, ``away_team_id``.

    Returns:
        DataFrame with columns ``match_id``, ``team_id``, ``scheduled_at``,
        ``is_road``. Sorted by ``(team_id, scheduled_at)`` and reset-indexed.
    """
    home = matches[["match_id", "scheduled_at", "home_team_id"]].rename(
        columns={"home_team_id": "team_id"},
    )
    home["is_road"] = False
    away = matches[["match_id", "scheduled_at", "away_team_id"]].rename(
        columns={"away_team_id": "team_id"},
    )
    away["is_road"] = True
    apps = pd.concat([home, away], ignore_index=True)
    return apps.sort_values(["team_id", "scheduled_at"]).reset_index(drop=True)


def trailing_game_count(apps: pd.DataFrame, window_days: int) -> pd.Series:
    """Per-team trailing count of games (inclusive) in the last N days.

    Two-pointer sweep, O(N) per team. Off-season gaps need no special handling:
    the sliding window naturally drops dates outside the window.
    """
    window = pd.Timedelta(days=window_days)
    pieces: list[pd.Series] = []
    for _, group in apps.groupby("team_id", sort=False):
        dates  = group["scheduled_at"].to_numpy()
        counts = [0] * len(dates)
        left   = 0
        for right in range(len(dates)):
            while dates[right] - dates[left] > window:
                left += 1
            counts[right] = right - left + 1
        pieces.append(pd.Series(counts, index=group.index, dtype="int64"))
    return pd.concat(pieces).sort_index()


def b2b_flag(apps: pd.DataFrame, max_gap_days: int) -> pd.Series:
    """Per-team back-to-back flag: 1 iff the previous game was exactly yesterday.

    Off-season detection takes precedence — a gap above ``max_gap_days`` keeps
    the flag at 0 even if the day-diff somehow reads as 1.
    """
    gaps = apps.groupby("team_id", sort=False)["scheduled_at"].diff().dt.days
    flag = (gaps == _B2B_GAP_DAYS) & (gaps <= max_gap_days)
    return flag.fillna(False).astype("int64")


def consecutive_road_streak(apps: pd.DataFrame, max_gap_days: int) -> pd.Series:
    """Per-team count of consecutive road appearances ending at this game.

    Resets to 0 on a home game or when the gap from the previous appearance
    exceeds ``max_gap_days`` (off-season).
    """
    pieces: list[pd.Series] = []
    for _, group in apps.groupby("team_id", sort=False):
        road = group["is_road"].to_numpy()
        gaps = (
            group["scheduled_at"].diff().dt.days
            .fillna(NO_PREV_GAME_DAYS).astype(int).to_numpy()
        )
        streak  = [0] * len(road)
        current = 0
        for i in range(len(road)):
            if gaps[i] > max_gap_days:
                current = 0
            if road[i]:
                current  += 1
                streak[i] = current
            else:
                current   = 0
                streak[i] = 0
        pieces.append(pd.Series(streak, index=group.index, dtype="int64"))
    return pd.concat(pieces).sort_index()
