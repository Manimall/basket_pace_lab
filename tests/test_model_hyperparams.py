"""Tests for the typed CatBoost hyperparameter override (model search).

I/O-free: no training, just construction. Covers that the V6 baseline mirrors
``settings.model``, that overrides land on the CatBoost params, and that the
default leaves ``l2_leaf_reg`` unpinned (no magic-number regularisation).
"""
from __future__ import annotations

from dataclasses import replace

from src.config import settings
from src.evaluation.model import build_classifier, default_hyperparams


def test_default_hyperparams_mirror_settings() -> None:
    hp = default_hyperparams()
    assert hp.iterations == settings.model.catboost_iterations
    assert hp.learning_rate == settings.model.catboost_lr
    assert hp.depth == settings.model.catboost_depth
    assert hp.random_seed == settings.model.catboost_seed
    assert hp.l2_leaf_reg is None  # V6 not pinned to a magic l2 value


def test_build_classifier_applies_baseline() -> None:
    params = build_classifier(default_hyperparams()).get_params()
    assert params["depth"] == settings.model.catboost_depth
    assert params["iterations"] == settings.model.catboost_iterations
    assert params["learning_rate"] == settings.model.catboost_lr


def test_light_config_overrides_depth_and_iterations() -> None:
    light = replace(default_hyperparams(), depth=3, iterations=500)
    params = build_classifier(light).get_params()
    assert params["depth"] == 3
    assert params["iterations"] == 500


def test_conservative_config_sets_l2_and_lr() -> None:
    cons = replace(default_hyperparams(), l2_leaf_reg=10.0, learning_rate=0.01)
    params = build_classifier(cons).get_params()
    assert params["l2_leaf_reg"] == 10.0
    assert params["learning_rate"] == 0.01


def test_baseline_leaves_l2_unset() -> None:
    """No l2_leaf_reg passed → CatBoost keeps its own default (key absent)."""
    params = build_classifier(default_hyperparams()).get_params()
    assert "l2_leaf_reg" not in params


def test_hyperparams_is_frozen() -> None:
    hp = default_hyperparams()
    try:
        hp.depth = 9  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("ModelHyperparams must be frozen")
