"""Per-league backtester — orchestration entry point.

Trains an independent CatBoostClassifier per pipeline spec (see
``src.evaluation.config.PIPELINES``) and prints a threshold-rowed bootstrap-CI
report for each. All numerical knobs live in
``settings.evaluation`` (``EvaluationConfig``); pipeline definitions live in
``src.evaluation.config``.

Run:
    python -m src.evaluation.backtester
"""
from __future__ import annotations

import asyncio
import logging

import pandas as pd

from src.config import settings
from src.evaluation.config import BIN_TARGET, LINE_COL, PIPELINES, PipelineSpec
from src.evaluation.model import EXCLUDED_FEATURES, get_x, train_classifier
from src.evaluation.reporting import print_threshold_table
from src.evaluation.simulation import (
    BetReport,
    SimulationParams,
    run_threshold_sweep,
)
from src.features.score_features import ALL_FEAT, TARGET, build_features, load_data
from src.models.evaluation import chrono_split

logging.basicConfig(level=settings.app.log_level, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _build_simulation_params() -> SimulationParams:
    """Snapshot the env-loaded ``EvaluationConfig`` into a frozen params bundle."""
    cfg = settings.evaluation
    return SimulationParams(
        odds            = cfg.odds,
        stake           = cfg.flat_stake,
        thresholds      = cfg.prob_thresholds,
        bootstrap_iters = cfg.bootstrap_iters,
        bootstrap_seed  = cfg.bootstrap_seed,
        ci_alpha        = cfg.bootstrap_ci_alpha,
    )


def prepare_dataset() -> pd.DataFrame:
    """Load matches, build features, derive the binary target.

    Drops rows without ``TARGET`` (no actual game total to score against),
    without ``LINE_COL`` (can't train a P(total > line) target without the
    line), and push rows where ``game_total == line`` (ambiguous label).

    Returns:
        A DataFrame ready to feed into per-league pipelines. Includes all
        feature columns, the continuous ``TARGET``, the closing ``LINE_COL``,
        the categorical ``league``, and the binary ``BIN_TARGET``.
    """
    log.info("Loading data from DB…")
    matches, qs = asyncio.run(load_data())

    log.info("Building features…")
    df = build_features(matches, qs).dropna(subset=[TARGET]).reset_index(drop=True)
    log.info("  Rows after feature build: %d", len(df))

    before = len(df)
    df = df.dropna(subset=[LINE_COL]).reset_index(drop=True)
    log.info("  Dropped %d rows without %s → %d remain",
             before - len(df), LINE_COL, len(df))

    pushes = int((df[TARGET] == df[LINE_COL]).sum())
    if pushes:
        log.info("  Dropping %d push rows (game_total == line)", pushes)
        df = df[df[TARGET] != df[LINE_COL]].reset_index(drop=True)

    df[BIN_TARGET] = (df[TARGET] > df[LINE_COL]).astype(int)
    log.info("  Overall OVER rate: %.2f%%", 100 * df[BIN_TARGET].mean())

    feat_count = len([c for c in ALL_FEAT if c in df.columns and c not in EXCLUDED_FEATURES])
    log.info("Using %d features (hidden: %s)", feat_count, sorted(EXCLUDED_FEATURES))
    return df


def _apply_spec(df: pd.DataFrame, spec: PipelineSpec) -> pd.DataFrame:
    """Filter the dataset to the leagues described by ``spec``.

    Args:
        df: Output of ``prepare_dataset``.
        spec: Pipeline specification (which leagues to include/exclude).

    Returns:
        A reset-index subset; possibly empty if no league matches.
    """
    subset = df
    if spec.include:
        subset = subset[subset["league"].isin(spec.include)]
    if spec.exclude:
        subset = subset[~subset["league"].isin(spec.exclude)]
    return subset.reset_index(drop=True)


def run_pipeline(
    df: pd.DataFrame, spec: PipelineSpec, params: SimulationParams,
) -> list[BetReport] | None:
    """Train + evaluate one per-league pipeline.

    Applies ``spec``, splits chronologically, trains a CatBoost classifier,
    predicts on the held-out test portion, and runs the threshold sweep.

    Args:
        df: Output of ``prepare_dataset``.
        spec: Pipeline specification.
        params: Simulation knobs (odds, stake, thresholds, bootstrap …).

    Returns:
        A list of ``BetReport`` (one per threshold), or ``None`` if the subset
        is empty or below ``settings.evaluation.min_*_rows`` thresholds.
    """
    subset = _apply_spec(df, spec)
    if subset.empty:
        log.warning("[%s] empty subset — skipping", spec.name)
        return None

    leagues = sorted(subset["league"].unique().tolist())
    log.info("[%s] rows=%d  leagues=%s", spec.name, len(subset), leagues)

    train_df, test_dict = chrono_split(subset)
    test_df = pd.concat(test_dict.values(), ignore_index=True)
    over_rate = 100 * train_df[BIN_TARGET].mean() if len(train_df) else float("nan")
    log.info("[%s]  train=%d  test=%d  (OVER rate in train: %.2f%%)",
             spec.name, len(train_df), len(test_df), over_rate)

    guards = settings.evaluation
    if len(train_df) < guards.min_train_rows or len(test_df) < guards.min_test_rows:
        log.warning("[%s] too few rows (train=%d < %d, test=%d < %d) — skipping",
                    spec.name, len(train_df), guards.min_train_rows,
                    len(test_df),  guards.min_test_rows)
        return None

    # Per-league feature selection: single-league pipelines pass the league key
    # so the selector can drop signal-shape-mismatched groups (e.g. fatigue for
    # sparsely-scheduled leagues). Mixed-league pipelines pass None → default
    # rules (which currently means "no fatigue" — see feature_selector).
    league_key = spec.include[0] if len(spec.include) == 1 else None

    model   = train_classifier(train_df, target_col=BIN_TARGET, league_key=league_key)
    X_te    = get_x(test_df, league_key=league_key)
    probs   = model.predict_proba(X_te)[:, 1]
    actuals = test_df[TARGET].to_numpy()
    lines   = test_df[LINE_COL].to_numpy()

    return run_threshold_sweep(probs, actuals, lines, params)


def main() -> None:
    """Entry point: run every pipeline in ``PIPELINES`` and print results."""
    df = prepare_dataset()
    params = _build_simulation_params()

    results: dict[str, list[BetReport]] = {}
    for spec in PIPELINES:
        rows = run_pipeline(df, spec, params)
        if rows is not None:
            results[spec.name] = rows

    for name, rows in results.items():
        print_threshold_table(
            f"{name} — by probability threshold "
            f"(Bootstrap CI95, {params.bootstrap_iters} iters):",
            rows,
        )


if __name__ == "__main__":
    main()
