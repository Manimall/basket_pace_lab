"""Arena-context features (V8): home-court fortress / away vulnerability.

The feature-importance audit showed European totals hinge on the home-court
factor. These two indices quantify a team's home-vs-away win-rate split:

    home_fortress_index      = home team's (home win% − away win%)   [L5 split]
    away_vulnerability_index = away team's (home win% − away win%)   [L5 split]

Both derive from the **same** per-team quantity (home-minus-away win-rate gap),
attached to the home team (fortress) and the away team (vulnerability) of each
match.

Leakage-safe by construction: per team, the gap is computed from games
strictly BEFORE the current match, split by venue (separate home/away
histories). A team's gap is recorded *before* its current result is folded in.

Enabled per-league via the feature selector (European home-court regimes).
NaN until a team has both home and away history — CatBoost handles NaN natively.
"""
from __future__ import annotations

import logging
from collections import deque

import pandas as pd

from src.config import settings

log = logging.getLogger(__name__)

HOME_FORTRESS_COL:      str       = "home_fortress_index"
AWAY_VULNERABILITY_COL: str       = "away_vulnerability_index"
ARENA_FEAT_COLS:        list[str] = [HOME_FORTRESS_COL, AWAY_VULNERABILITY_COL]

_GAP_COL: str = "ha_gap"   # internal per-appearance home-minus-away win-rate gap


def _team_split_winrate_gap(games: pd.DataFrame, window: int) -> list[float]:
    """Per-appearance home-minus-away win-rate gap using prior games only.

    Args:
        games: One team's appearances sorted by ``scheduled_at``, with boolean
            ``is_road`` and float ``win`` (1.0 win / 0.0 loss) columns.
        window: Number of prior same-venue games to average (e.g. 5).

    Returns:
        Gap values aligned to ``games`` rows. NaN until the team has at least
        one prior home game AND one prior away game (gap undefined otherwise).
    """
    home_hist: deque[float] = deque(maxlen=window)
    away_hist: deque[float] = deque(maxlen=window)
    gaps: list[float] = []
    for is_road, win in zip(games["is_road"], games["win"], strict=True):
        home_wr = sum(home_hist) / len(home_hist) if home_hist else float("nan")
        away_wr = sum(away_hist) / len(away_hist) if away_hist else float("nan")
        gaps.append(home_wr - away_wr)        # NaN if either side has no history
        if is_road:
            away_hist.append(float(win))
        else:
            home_hist.append(float(win))
    return gaps


def _build_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """Stack home/away appearances with a venue flag and win result.

    Args:
        df: Match-level frame with ``match_id``, ``home_team_id``,
            ``away_team_id``, ``scheduled_at``, ``home_score_final``,
            ``away_score_final``.

    Returns:
        Per-appearance frame sorted by ``(team_id, scheduled_at)`` with
        ``is_road`` and ``win`` columns.
    """
    home_win = (df["home_score_final"] > df["away_score_final"]).astype(float)
    away_win = (df["away_score_final"] > df["home_score_final"]).astype(float)
    home = pd.DataFrame({
        "match_id":     df["match_id"],
        "team_id":      df["home_team_id"],
        "scheduled_at": df["scheduled_at"],
        "is_road":      False,
        "win":          home_win,
    })
    away = pd.DataFrame({
        "match_id":     df["match_id"],
        "team_id":      df["away_team_id"],
        "scheduled_at": df["scheduled_at"],
        "is_road":      True,
        "win":          away_win,
    })
    return pd.concat([home, away], ignore_index=True).sort_values(
        ["team_id", "scheduled_at"]
    )


def add_arena_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``home_fortress_index`` and ``away_vulnerability_index`` columns.

    Args:
        df: Match-level frame (see ``_build_appearances`` for required columns).

    Returns:
        ``df`` extended with ``ARENA_FEAT_COLS``. Values are NaN for teams that
        lack both home and away history (early season) — left NaN deliberately
        (no imputation; CatBoost handles it).
    """
    window = settings.features.arena_winrate_window
    apps   = _build_appearances(df)

    pieces: list[pd.DataFrame] = []
    for _, games in apps.groupby("team_id", sort=False):
        g = games.copy()
        g[_GAP_COL] = _team_split_winrate_gap(g, window)
        pieces.append(g[["match_id", "team_id", _GAP_COL]])
    gaps = pd.concat(pieces, ignore_index=True)

    home_gap = gaps.rename(columns={"team_id": "home_team_id", _GAP_COL: HOME_FORTRESS_COL})
    df = df.merge(home_gap, on=["match_id", "home_team_id"], how="left")
    away_gap = gaps.rename(columns={"team_id": "away_team_id", _GAP_COL: AWAY_VULNERABILITY_COL})
    df = df.merge(away_gap, on=["match_id", "away_team_id"], how="left")

    n_fort = int(df[HOME_FORTRESS_COL].notna().sum())
    log.info(
        "Arena context: %d/%d matches with fortress index (rest NaN — insufficient split history).",
        n_fort, len(df),
    )
    return df
