"""NULL-safe advanced (Four Factors) metrics — possession-normalised efficiency.

Only leagues with box-score data (currently NBA via Sofascore) carry
possessions/turnovers; score-only leagues (Flashscore: EuroLeague, ABA,
Israel, …) have them NULL. This module computes efficiency metrics where the
data exists and leaves them NaN elsewhere, plus a ``has_boxscore`` flag so the
model can gate its reliance on them.

Fallback policy: **NaN, not imputation.** Cross-league mean would inject NBA
pace into ABA (different pace regimes) — nonsense. CatBoost handles NaN
natively (``nan_mode`` routes missing rows to a consistent branch), so NaN is
the correct, leakage-free fallback.

Leakage guarantee: per-match metrics are post-game, but only their shift(1)
rolling versions (via ``compute_rolling_ema``) reach the model — identical to
the base-score features.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import settings
from src.features.rolling_utils import compute_rolling_ema, merge_rolling_by_side

log = logging.getLogger(__name__)

# Per-team advanced stat names (computed per match, then rolled).
_ADV_STATS: tuple[str, ...] = ("ortg", "drtg", "tov_rate", "true_pace")

# Advanced metrics use a shorter window set than base stats: the configured
# short rolling window + EMA. Derived from config — no hardcoded window value.
_ADV_WINDOWS:  tuple[int, ...] = (settings.features.score_roll_windows[0],)
_ADV_EMA_SPAN: int            = settings.features.ema_span
# Rolling suffixes derived from the windows above (single source of truth — the
# "L{w}" strings can never drift out of sync with _ADV_WINDOWS).
_ADV_SUFFIXES: tuple[str, ...] = tuple(f"L{w}" for w in _ADV_WINDOWS) + (f"EMA{_ADV_EMA_SPAN}",)

_PER_100:                float = 100.0
_MIN_VALID_POSSESSIONS:  float = 1.0   # below this = missing box-score

HAS_BOXSCORE_COL: str = "has_boxscore"

# Source box-score columns expected on the per-quarter frame.
_BOX_SOURCE: dict[str, str] = {
    "home_poss": "home_possessions",
    "away_poss": "away_possessions",
    "home_tov":  "home_turnovers",
    "away_tov":  "away_turnovers",
}

ADVANCED_FEAT_COLS: list[str] = [
    f"{side}_{stat}_{sfx}"
    for side in ("home", "away")
    for stat in _ADV_STATS
    for sfx in _ADV_SUFFIXES
] + [HAS_BOXSCORE_COL]


def aggregate_box_score(qs: pd.DataFrame) -> pd.DataFrame:
    """Sum per-quarter possessions and turnovers to match level.

    Args:
        qs: quarter_stats rows; box-score columns may be entirely NULL for
            score-only leagues.

    Returns:
        One row per ``match_id`` with ``home_poss``/``away_poss``/``home_tov``/
        ``away_tov``. All-NULL groups yield NaN (via ``min_count=1``) so
        downstream division stays NaN-safe rather than dividing by zero.
    """
    present = {out: src for out, src in _BOX_SOURCE.items() if src in qs.columns}
    if not present:
        log.warning("Box-score columns absent from quarter_stats — advanced metrics will be all-NaN.")
        return pd.DataFrame({"match_id": qs["match_id"].unique()})

    agg = (
        qs.groupby("match_id")
        .agg(**{
            out: pd.NamedAgg(column=src, aggfunc=lambda s: s.sum(min_count=1))
            for out, src in present.items()
        })
        .reset_index()
    )
    return agg


def _compute_match_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team ORtg/DRtg/TOV%/true_pace at match level (NaN-safe).

    Args:
        df: Match frame with possessions, turnovers, and final scores merged in.

    Returns:
        ``df`` with eight per-team metric columns added
        (``home_ortg`` … ``away_true_pace``). Division by missing/zero
        possessions yields NaN, never an exception.
    """
    df = df.copy()
    for col in ("home_poss", "away_poss"):
        if col in df.columns:
            df[col] = df[col].where(df[col] >= _MIN_VALID_POSSESSIONS)

    hp, ap   = df.get("home_poss"), df.get("away_poss")
    htov     = df.get("home_tov")
    atov     = df.get("away_tov")
    hs, as_  = df["home_score_final"], df["away_score_final"]

    # Home perspective: own offence vs opponent's possessions for defence.
    df["home_ortg"]      = hs   / hp * _PER_100
    df["home_drtg"]      = as_  / ap * _PER_100
    df["home_tov_rate"]  = htov / hp * _PER_100
    df["home_true_pace"] = hp
    # Away perspective is the mirror image.
    df["away_ortg"]      = as_  / ap * _PER_100
    df["away_drtg"]      = hs   / hp * _PER_100
    df["away_tov_rate"]  = atov / ap * _PER_100
    df["away_true_pace"] = ap
    return df


def _build_advanced_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Stack home/away per-team advanced metrics into one appearance timeline."""
    sides = []
    for side in ("home", "away"):
        sides.append(pd.DataFrame({
            "match_id":     df["match_id"],
            "team_id":      df[f"{side}_team_id"],
            "scheduled_at": df["scheduled_at"],
            "ortg":         df[f"{side}_ortg"],
            "drtg":         df[f"{side}_drtg"],
            "tov_rate":     df[f"{side}_tov_rate"],
            "true_pace":    df[f"{side}_true_pace"],
        }))
    return pd.concat(sides, ignore_index=True)


def add_advanced_metrics(df: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """Merge box-score, compute advanced metrics, roll them, attach flag.

    Args:
        df: Match-level frame (needs ``match_id``, ``home_team_id``,
            ``away_team_id``, ``scheduled_at``, ``home_score_final``,
            ``away_score_final``).
        box: Output of ``aggregate_box_score``.

    Returns:
        ``df`` extended with the columns in ``ADVANCED_FEAT_COLS``. Advanced
        rolling columns are NaN for score-only leagues; ``has_boxscore`` is
        ``1`` where possessions were present, ``0`` otherwise. Advanced columns
        are intentionally NOT NaN-filled (no cross-league imputation).
    """
    df = df.merge(box, on="match_id", how="left")
    df = _compute_match_advanced(df)

    timeline = _build_advanced_timeline(df)
    rolling  = compute_rolling_ema(
        timeline, stat_cols=_ADV_STATS, windows=_ADV_WINDOWS, ema_span=_ADV_EMA_SPAN,
    )
    roll_cols = [
        f"{stat}_{sfx}"
        for stat in _ADV_STATS
        for sfx in _ADV_SUFFIXES
    ]
    df = merge_rolling_by_side(df, rolling, roll_cols)

    has_box = df["home_poss"].notna() if "home_poss" in df.columns else pd.Series(False, index=df.index)
    df[HAS_BOXSCORE_COL] = has_box.astype(int)

    n_box = int(df[HAS_BOXSCORE_COL].sum())
    log.info(
        "Advanced metrics: %d/%d matches have box-score (%.1f%%); rest left NaN.",
        n_box, len(df), 100.0 * n_box / len(df) if len(df) else 0.0,
    )
    return df
