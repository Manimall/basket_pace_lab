"""Sofascore ID finder package.

Reverse-maps Flashscore-sourced matches (``external_id = 'fs_…'``) to numeric
Sofascore event IDs so the Go scout can enrich them. See ``runner.run`` for the
orchestration entry point and ``scripts/sofascore_id_finder.py`` for the CLI.
"""
from __future__ import annotations

from src.data_collection.sofascore.id_finder.config import DEFAULT_THRESHOLD
from src.data_collection.sofascore.id_finder.runner import FinderOptions, run

__all__ = ["DEFAULT_THRESHOLD", "FinderOptions", "run"]
