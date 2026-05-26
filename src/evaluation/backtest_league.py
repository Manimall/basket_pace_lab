"""Generic per-league backtester — single CLI for any tournament_name.

Wraps the V6 architecture (per-league CatBoostClassifier, bootstrap CI95,
fatigue-aware feature set) for arbitrary ``tournament_name`` values in DB.
Used as the sole entry-point for league-specific R&D runs (ABA, Israel,
future candidates).

The frozen V6 main entry-point (``src.evaluation.backtester``) stays the
NBA/EuroLeague/OTHER baseline; this module is the parametric counterpart.

Run:
    python -m src.evaluation.backtest_league --league Israel
    python -m src.evaluation.backtest_league --league ABA --min-test-rows 20
"""
from __future__ import annotations

import argparse
import logging

from src.config import settings
from src.evaluation.backtester import prepare_dataset, run_pipeline
from src.evaluation.config import PipelineSpec
from src.evaluation.reporting import print_threshold_table
from src.evaluation.simulation import BetReport, SimulationParams

logging.basicConfig(level=settings.app.log_level, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_simulation_params() -> SimulationParams:
    """Snapshot env-loaded ``EvaluationConfig`` into a frozen params bundle.

    Returns:
        Frozen ``SimulationParams`` with odds, stake, thresholds, bootstrap
        config sourced from ``settings.evaluation``.
    """
    cfg = settings.evaluation
    return SimulationParams(
        odds            = cfg.odds,
        stake           = cfg.flat_stake,
        thresholds      = cfg.prob_thresholds,
        bootstrap_iters = cfg.bootstrap_iters,
        bootstrap_seed  = cfg.bootstrap_seed,
        ci_alpha        = cfg.bootstrap_ci_alpha,
    )


def run_league_backtest(league_key: str) -> list[BetReport] | None:
    """Train and evaluate a per-league CatBoost classifier on one tournament.

    Loads the current-season dataset (season gate applied in ``load_data``),
    filters to ``tournament_name == league_key`` via ``PipelineSpec``,
    chrono-splits 80/20, trains, and runs the threshold sweep.

    Args:
        league_key: Value of ``matches.tournament_name`` to isolate
            (e.g. ``"ABA"``, ``"Israel"``, ``"NBA"``).

    Returns:
        One ``BetReport`` per probability threshold, or ``None`` if the
        subset falls below ``settings.evaluation.min_train_rows`` /
        ``min_test_rows`` guards.
    """
    df     = prepare_dataset()
    params = build_simulation_params()
    spec   = PipelineSpec(name=f"{league_key} — per-league", include=(league_key,))
    return run_pipeline(df, spec, params)


def _parse_cli() -> argparse.Namespace:
    """Parse ``--league`` (required) and optional ``--min-test-rows`` override."""
    parser = argparse.ArgumentParser(
        description="Per-league V6 backtester with bootstrap-CI95 threshold sweep.",
    )
    parser.add_argument(
        "--league", required=True,
        help="tournament_name in matches table (e.g. ABA, Israel, NBA)",
    )
    parser.add_argument(
        "--min-test-rows", type=int, default=None,
        help="Override settings.evaluation.min_test_rows for this run.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse CLI, run league backtest, print threshold table."""
    args = _parse_cli()

    if args.min_test_rows is not None:
        log.info(
            "Overriding settings.evaluation.min_test_rows: %d → %d",
            settings.evaluation.min_test_rows, args.min_test_rows,
        )
        settings.evaluation.min_test_rows = args.min_test_rows

    rows = run_league_backtest(args.league)
    if rows is None:
        log.warning(
            "No rows returned — verify tournament_name='%s' exists in DB and "
            "that the test set after chrono-split has ≥%d rows.",
            args.league, settings.evaluation.min_test_rows,
        )
        return

    params = build_simulation_params()
    print_threshold_table(
        f"{args.league} — per-league (Bootstrap CI95, {params.bootstrap_iters} iters):",
        rows,
    )


if __name__ == "__main__":
    main()
