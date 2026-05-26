"""ABA League pivot backtester — combined seasons 23/24 + 24/25 + 25/26.

Per-league V6 architecture: CatBoostClassifier trained on the train portion of
all ABA data, evaluated on the chronologically last 20% test slice. Reuses
the production pipeline (prepare_dataset / run_pipeline / threshold sweep /
bootstrap CI) — no duplication of the V6 logic, just a different
``PipelineSpec``.

The frozen V6 main entry-point (``src.evaluation.backtester``) stays unchanged
as the documented NBA baseline; this module is the strategic-pivot entry-point.

Run:
    python -m src.evaluation.backtest_aba
"""
from __future__ import annotations

import logging

from src.config import settings
from src.evaluation.backtester import prepare_dataset, run_pipeline
from src.evaluation.config import PipelineSpec
from src.evaluation.reporting import print_threshold_table
from src.evaluation.simulation import BetReport, SimulationParams

logging.basicConfig(level=settings.app.log_level, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ABA_LEAGUE_KEY: str = "ABA"
ABA_PIPELINE_NAME: str = "ABA combined (23/24 + 24/25 + 25/26) — per-league"


def _build_params() -> SimulationParams:
    """Snapshot the env-loaded EvaluationConfig into a frozen params bundle.

    Returns:
        Frozen SimulationParams with odds, stake, thresholds, bootstrap config
        sourced from ``settings.evaluation``.
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


def run_aba_backtest() -> list[BetReport] | None:
    """Train and evaluate a per-league CatBoost classifier on combined ABA data.

    Steps:
      1. Load all matches + features via the shared ``prepare_dataset``.
      2. Filter to ``tournament_name = "ABA"`` via ``PipelineSpec(include=...)``.
      3. Chrono-split 80/20 (per-league = whole ABA chronological order).
      4. Train CatBoostClassifier on the train portion.
      5. Threshold sweep + bootstrap CI95 (5000 iters from settings).

    Returns:
        One ``BetReport`` per probability threshold, or ``None`` if the dataset
        is below the configured ``min_train_rows`` / ``min_test_rows`` guards.
    """
    df = prepare_dataset()
    params = _build_params()
    spec = PipelineSpec(name=ABA_PIPELINE_NAME, include=(ABA_LEAGUE_KEY,))
    return run_pipeline(df, spec, params)


def main() -> None:
    """Entry point: run the ABA backtest and print the final threshold table."""
    rows = run_aba_backtest()
    if rows is None:
        log.warning(
            "ABA pipeline returned no rows — likely below min_train_rows / "
            "min_test_rows guard. Adjust settings.evaluation.min_test_rows.",
        )
        return

    params = _build_params()
    print_threshold_table(
        f"{ABA_PIPELINE_NAME} — Bootstrap CI95, {params.bootstrap_iters} iters:",
        rows,
    )


if __name__ == "__main__":
    main()
