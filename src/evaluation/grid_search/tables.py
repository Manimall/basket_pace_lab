"""Result-table assembly and rendering for the grid search.

Converts a list of ``GridCellResult`` into two pandas frames (full grid and
per-league winners) and renders them as Markdown strings for logging — no bare
``print`` and no untyped dict passing.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.evaluation.grid_search.cell import GridCellResult
from src.evaluation.grid_search.periods import GRID_THRESHOLDS, Period

log = logging.getLogger(__name__)

_TABLE_WIDTH: int = 80
_TABLE_RULE:  str = "═" * _TABLE_WIDTH


def _cell_to_row(cell: GridCellResult) -> dict[str, Any]:
    """Flatten a GridCellResult into a flat dict for DataFrame assembly."""
    row: dict[str, Any] = {
        "league":     cell.league,
        "combo":      cell.combo,
        "n_train":    cell.n_train,
        "n_test":     cell.n_test,
        "n_features": cell.n_features,
        "log_loss":   cell.log_loss,
        "roc_auc":    cell.roc_auc,
    }
    if cell.synthetic_line is not None:
        row["synthetic_line"] = cell.synthetic_line
    for thr in GRID_THRESHOLDS:
        label = f"{thr:.2f}"
        row[f"roi_{label}"]  = cell.roi_by_threshold.get(label)
        row[f"bets_{label}"] = cell.bets_by_threshold.get(label)
    row["best_roi"]  = cell.best_roi
    row["best_thr"]  = cell.best_thr
    row["best_bets"] = cell.best_bets
    return row


def build_full_table(cells: list[GridCellResult], period: Period) -> pd.DataFrame:
    """Build the full (league × combo) grid table, column-ordered for display."""
    if not cells:
        return pd.DataFrame()
    df = pd.DataFrame([_cell_to_row(c) for c in cells])

    thr_cols: list[str] = []
    for thr in GRID_THRESHOLDS:
        label = f"{thr:.2f}"
        thr_cols += [f"roi_{label}", f"bets_{label}"]
    base_cols = ["league", "combo", "n_train", "n_test", "n_features", "log_loss", "roc_auc"]
    if period != "game" and "synthetic_line" in df.columns:
        base_cols.append("synthetic_line")
    cols = [c for c in base_cols + thr_cols + ["best_roi", "best_thr", "best_bets"]
            if c in df.columns]
    return df[cols].sort_values(["league", "combo"]).reset_index(drop=True)


def build_winner_table(full: pd.DataFrame) -> pd.DataFrame:
    """Reduce the full table to one winning combo per league (highest best_roi).

    Tie-break: higher best_roi first, then more bets (better-sampled combo wins).
    """
    if full.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for league, grp in full.groupby("league"):
        viable = grp.dropna(subset=["best_roi"])
        if viable.empty:
            rows.append({
                "league": league, "winner_combo": "—",
                "best_roi": float("nan"), "best_thr": "—", "best_bets": 0,
                "log_loss": float("nan"), "roc_auc": float("nan"),
            })
            continue
        best = viable.loc[viable.sort_values(["best_roi", "best_bets"], ascending=False).index[0]]
        rows.append({
            "league":       league,
            "winner_combo": best["combo"],
            "best_roi":     best["best_roi"],
            "best_thr":     best["best_thr"],
            "best_bets":    int(best["best_bets"]),
            "log_loss":     best["log_loss"],
            "roc_auc":      best["roc_auc"],
        })
    return (
        pd.DataFrame(rows)
        .sort_values("best_roi", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def render_table(df: pd.DataFrame, title: str) -> str:
    """Render a frame as a titled Markdown table string (falls back to to_string)."""
    try:
        body = df.to_markdown(index=False)
    except ImportError:  # tabulate missing — plain text fallback
        body = df.to_string(index=False)
    return f"\n{_TABLE_RULE}\n  {title}\n{_TABLE_RULE}\n{body}\n"
