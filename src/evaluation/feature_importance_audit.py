"""V6 per-league feature-importance audit (analysis-only).

Trains CatBoost classifiers under the production V6 pipeline for a set of
audited leagues, then dumps each model's ``get_feature_importance()``. No
betting logic is touched — this module exists purely to inform the next
feature-pruning iteration.

Output:
    * Top-N feature ranking per league (default N=15) as a bar chart.
    * "Dead weight" list: features present in every audited league's model
      but with importance below the configured threshold in all of them —
      candidates for removal from the global feature set.

Run:
    python -m src.evaluation.feature_importance_audit
    python -m src.evaluation.feature_importance_audit --top-n 20 --dead-threshold 0.3
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import pandas as pd

from src.config import settings
from src.evaluation.backtester import prepare_dataset
from src.evaluation.config import BIN_TARGET
from src.evaluation.model import get_x, train_classifier
from src.models.evaluation import chrono_split

logging.basicConfig(level=settings.app.log_level, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Defaults — overridable via CLI.
DEFAULT_TOP_N:          int          = 15
DEFAULT_DEAD_THRESHOLD: float        = 0.5
LEAGUES_UNDER_AUDIT:    tuple[str, ...] = ("NBA", "EuroLeague", "ABA")

# Bar-chart width (characters) for the importance visualisation.
_BAR_WIDTH: int = 25


@dataclass(frozen=True)
class LeagueImportance:
    """Per-league feature-importance ranking.

    Attributes:
        league_key: Value of ``matches.tournament_name`` identifying the league.
        n_train: Number of rows the model was trained on.
        importance: DataFrame with columns ``feature`` and ``importance``
            (percentage), sorted descending by importance.
    """
    league_key:  str
    n_train:     int
    importance:  pd.DataFrame


def compute_league_importance(
    df: pd.DataFrame, league_key: str,
) -> LeagueImportance | None:
    """Train a per-league CatBoost classifier and read feature importance.

    Mirrors the production V6 path: same chrono-split, same hyperparameters,
    same per-league feature selector. Only the output differs — instead of
    predicting on the test split, we read the trained model's
    ``get_feature_importance()``.

    Args:
        df: Output of ``prepare_dataset()``.
        league_key: Value of ``matches.tournament_name`` to isolate.

    Returns:
        A ``LeagueImportance`` bundle, or ``None`` if the league has no rows
        in the current-season dataset.
    """
    subset = df[df["league"] == league_key].reset_index(drop=True)
    if subset.empty:
        log.warning("League %s: empty subset, skipping", league_key)
        return None

    train_df, _ = chrono_split(subset)
    model       = train_classifier(train_df, target_col=BIN_TARGET, league_key=league_key)
    X_train     = get_x(train_df, league_key=league_key)

    fi = (
        pd.DataFrame(
            {"feature": X_train.columns, "importance": model.get_feature_importance()},
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    log.info(
        "League %s: trained on %d rows × %d features, max importance %.2f%%",
        league_key, len(train_df), len(X_train.columns), fi["importance"].max(),
    )
    return LeagueImportance(league_key=league_key, n_train=len(train_df), importance=fi)


def find_dead_weight(
    reports: list[LeagueImportance], threshold: float,
) -> list[str]:
    """Return features present in every report but sub-threshold in all of them.

    Uses an inner join across leagues — features missing from any league's
    model (e.g. fatigue cols dropped by the selector for sparse leagues) are
    excluded from the intersection so we only flag truly universal dead weight.

    Args:
        reports: Output of ``compute_league_importance`` for the audited leagues.
        threshold: Importance percentage below which a feature is "dead" in
            that league.

    Returns:
        Sorted list of feature names that are sub-threshold in every audited
        league. Empty if no candidates exist.
    """
    if not reports:
        return []

    indexed = [
        r.importance.set_index("feature")["importance"].rename(r.league_key)
        for r in reports
    ]
    merged   = pd.concat(indexed, axis=1, join="inner")
    dead_msk = (merged < threshold).all(axis=1)
    return sorted(merged[dead_msk].index.tolist())


def render_top_n_table(report: LeagueImportance, top_n: int) -> str:
    """Render the top-N features for one league as a multi-line bar chart."""
    fi      = report.importance.head(top_n)
    max_imp = float(fi["importance"].max() or 1.0)
    feat_n  = len(report.importance)
    lines = [
        f"=== {report.league_key} "
        f"(train n={report.n_train}, {feat_n} features) — TOP {top_n} ==="
    ]
    for _, row in fi.iterrows():
        bar = "█" * int(row["importance"] / max_imp * _BAR_WIDTH)
        lines.append(f"  {row['feature']:<42s} {row['importance']:6.2f}%  {bar}")
    return "\n".join(lines)


def render_dead_weight(features: list[str], threshold: float) -> str:
    """Render the dead-weight candidate list as a multi-line string."""
    lines = [
        f"=== DEAD WEIGHT (importance < {threshold:.2f}% in ALL audited leagues) ==="
    ]
    if not features:
        lines.append("  (none found)")
        return "\n".join(lines)
    lines.append(f"  Found {len(features)} candidate(s) for pruning:")
    lines.extend(f"    - {f}" for f in features)
    return "\n".join(lines)


def _parse_cli() -> argparse.Namespace:
    """Parse optional ``--top-n`` and ``--dead-threshold`` overrides."""
    parser = argparse.ArgumentParser(
        description="V6 per-league feature-importance audit (read-only).",
    )
    parser.add_argument(
        "--top-n", type=int, default=DEFAULT_TOP_N,
        help=f"Top features to print per league (default {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--dead-threshold", type=float, default=DEFAULT_DEAD_THRESHOLD,
        help=(
            f"Importance %% cutoff for dead-weight detection "
            f"(default {DEFAULT_DEAD_THRESHOLD})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: train, extract, print top-N and dead-weight lists."""
    args = _parse_cli()
    df   = prepare_dataset()

    reports: list[LeagueImportance] = []
    for league in LEAGUES_UNDER_AUDIT:
        report = compute_league_importance(df, league)
        if report is not None:
            reports.append(report)
            log.info("\n%s", render_top_n_table(report, args.top_n))

    if reports:
        dead = find_dead_weight(reports, args.dead_threshold)
        log.info("\n%s", render_dead_weight(dead, args.dead_threshold))


if __name__ == "__main__":
    main()
