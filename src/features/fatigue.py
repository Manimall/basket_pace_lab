"""Schedule-fatigue feature engineering for the totals pipeline.

Computes per-team calendar / workload indicators that influence game pace:
back-to-back flag, density of recent appearances, consecutive-road streaks,
relative rest advantage. All features are derivable from
``matches.scheduled_at`` — no new data source required.

Off-season handling
-------------------
Any gap from a team's previous appearance that exceeds
``settings.features.days_rest_clip_max`` (default 21 days) is treated as a
season opener — fatigue indicators reset to their "fresh" defaults so the
long inter-season break does not leak as a "well-rested" signal.

Single responsibility
---------------------
This module emits only the per-team fatigue columns and merges them onto the
match-level frame. Rolling stats, matchup deltas, and the existing
``home_days_rest`` / ``away_days_rest`` computation stay in
``score_features.py`` / ``rolling_utils.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.config import settings

log = logging.getLogger(__name__)


FATIGUE_FEATURE_COLS: list[str] = [
    "home_is_b2b",
    "away_is_b2b",
    "home_games_last_short",
    "away_games_last_short",
    "home_games_last_long",
    "away_games_last_long",
    "home_is_high_density",
    "away_is_high_density",
    "away_consecutive_road",
    "rest_diff",
]

# Sentinel for "no previous game" — anything above off_season_max_days resets state.
_NO_PREV_GAME_DAYS: int = 10**6


@dataclass(frozen=True)
class FatigueWindowConfig:
    """Frozen bundle of knobs controlling fatigue computation.

    Attributes:
        short_window_days: Window size for the dense "games in last N days"
            metric (e.g. 4 — captures the "3 in 4 nights" NBA pattern).
        long_window_days: Wider context window (e.g. 7 — week-level density).
        high_density_threshold: Games in the short window above this trigger
            the ``*_is_high_density`` flag.
        off_season_max_days: Inter-game gap above this resets the team's
            fatigue state (treated as a new season).
    """
    short_window_days:      int
    long_window_days:       int
    high_density_threshold: int
    off_season_max_days:    int


def _config_from_settings() -> FatigueWindowConfig:
    """Snapshot fatigue knobs from ``settings.features``.

    Returns:
        A frozen ``FatigueWindowConfig`` populated from the env-loaded
        ``FeatureConfig``. ``off_season_max_days`` reuses the existing
        ``days_rest_clip_max`` to keep a single source of truth.
    """
    cfg = settings.features
    return FatigueWindowConfig(
        short_window_days      = cfg.fatigue_short_window_days,
        long_window_days       = cfg.fatigue_long_window_days,
        high_density_threshold = cfg.fatigue_high_density_threshold,
        off_season_max_days    = cfg.days_rest_clip_max,
    )


def build_team_appearances(matches: pd.DataFrame) -> pd.DataFrame:
    """Stack home + away rows into one team-appearance timeline.

    Args:
        matches: Match-level DataFrame; must contain ``match_id``,
            ``scheduled_at``, ``home_team_id``, ``away_team_id``.

    Returns:
        DataFrame with columns ``match_id``, ``team_id``, ``scheduled_at``,
        ``is_road``. Sorted by ``(team_id, scheduled_at)`` and
        reset-indexed — ready for per-team groupby work.
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


def _trailing_game_count(apps: pd.DataFrame, window_days: int) -> pd.Series:
    """Per-team trailing count of games (inclusive of current) in the last N days.

    Two-pointer sweep, O(N) per team. Off-season gaps need no special
    handling: the sliding window naturally drops dates outside the window.

    Args:
        apps: Team-appearance timeline from ``build_team_appearances``.
        window_days: Lookback window size in days.

    Returns:
        Series of int64 counts, aligned to ``apps.index``.
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


def _b2b_flag(apps: pd.DataFrame, max_gap_days: int) -> pd.Series:
    """Per-team back-to-back flag: 1 iff previous game was exactly yesterday.

    Off-season detection takes precedence — even if ``diff == 1`` somehow
    crosses an inter-season boundary by data quirk, the gap check keeps the
    flag at 0.

    Args:
        apps: Team-appearance timeline from ``build_team_appearances``.
        max_gap_days: Days threshold above which fatigue state resets.

    Returns:
        Series of int64 flags (0/1), aligned to ``apps.index``.
    """
    gaps = apps.groupby("team_id", sort=False)["scheduled_at"].diff().dt.days
    flag = (gaps == 1) & (gaps <= max_gap_days)
    return flag.fillna(False).astype("int64")


def _consecutive_road_streak(apps: pd.DataFrame, max_gap_days: int) -> pd.Series:
    """Per-team count of consecutive road appearances ending at this game.

    Resets to 0 when the current appearance is at home (``is_road == False``)
    or when the gap from the previous appearance exceeds ``max_gap_days``
    (off-season).

    Args:
        apps: Team-appearance timeline from ``build_team_appearances``.
        max_gap_days: Days threshold above which the streak resets.

    Returns:
        Series of int64 streak counts, aligned to ``apps.index``. Zero for
        any appearance that is not a road game.
    """
    pieces: list[pd.Series] = []
    for _, group in apps.groupby("team_id", sort=False):
        road = group["is_road"].to_numpy()
        gaps = (
            group["scheduled_at"]
            .diff()
            .dt.days
            .fillna(_NO_PREV_GAME_DAYS)
            .astype(int)
            .to_numpy()
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


def _merge_to_match_sides(
    matches: pd.DataFrame, apps: pd.DataFrame, columns: list[str],
) -> pd.DataFrame:
    """Merge per-appearance columns onto ``matches`` as ``home_*``/``away_*``.

    Args:
        matches: Match-level DataFrame with ``match_id``, ``home_team_id``,
            ``away_team_id``.
        apps: Team-appearance frame with ``match_id``, ``team_id``, plus all
            source columns listed in ``columns``.
        columns: Per-appearance column names to merge per side.

    Returns:
        ``matches`` extended with ``home_<col>`` and ``away_<col>`` for every
        entry in ``columns``.
    """
    apps_slim = apps[["match_id", "team_id", *columns]]
    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        side_df = (
            apps_slim
            .merge(
                matches[["match_id", team_col]].rename(columns={team_col: "team_id"}),
                on=["match_id", "team_id"],
            )
            .drop(columns="team_id")
            .rename(columns={c: f"{side}_{c}" for c in columns})
        )
        matches = matches.merge(side_df, on="match_id", how="left")
    return matches


def add_fatigue_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Enrich a match-level frame with all schedule-fatigue features.

    Computes per-team appearance indicators (B2B flag, short/long window
    density, high-density flag, road streak), then merges them onto
    ``matches`` as ``home_*``/``away_*`` columns. Adds ``rest_diff`` if
    ``home_days_rest`` and ``away_days_rest`` are present (they are when
    called from ``score_features.build_features``).

    The ``home_consecutive_road`` column is intentionally dropped after the
    per-side merge: structurally it is always 0 (the host is not on a road
    trip by definition).

    Args:
        matches: Match-level DataFrame; must contain ``match_id``,
            ``scheduled_at``, ``home_team_id``, ``away_team_id``.

    Returns:
        ``matches`` with the columns listed in ``FATIGUE_FEATURE_COLS``
        populated.
    """
    cfg  = _config_from_settings()
    apps = build_team_appearances(matches)

    apps["is_b2b"]           = _b2b_flag(apps, cfg.off_season_max_days)
    apps["games_last_short"] = _trailing_game_count(apps, cfg.short_window_days)
    apps["games_last_long"]  = _trailing_game_count(apps, cfg.long_window_days)
    apps["is_high_density"]  = (
        apps["games_last_short"] >= cfg.high_density_threshold
    ).astype("int64")
    apps["consecutive_road"] = _consecutive_road_streak(apps, cfg.off_season_max_days)

    matches = _merge_to_match_sides(
        matches, apps,
        columns=[
            "is_b2b",
            "games_last_short",
            "games_last_long",
            "is_high_density",
            "consecutive_road",
        ],
    )
    # The host team's road streak is structurally zero; drop the clutter column.
    matches = matches.drop(columns="home_consecutive_road")

    if "home_days_rest" in matches.columns and "away_days_rest" in matches.columns:
        matches["rest_diff"] = matches["home_days_rest"] - matches["away_days_rest"]

    log.info(
        "Fatigue features added (%d cols): %s",
        len(FATIGUE_FEATURE_COLS), FATIGUE_FEATURE_COLS,
    )
    return matches
