"""Test the season-bucketing helper of the B.League walk-forward script.

The Aug rollover is the subtle bit: a July match belongs to the season that
started the PREVIOUS August, so mis-bucketing would silently leak a season
boundary into the walk-forward split. Loaded via importlib since ``scripts/``
is not a package.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "walk_forward_bleague.py"
_spec = importlib.util.spec_from_file_location("walk_forward_bleague", _SCRIPT)
assert _spec and _spec.loader
_wf = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _wf  # @dataclass needs the module registered before exec
_spec.loader.exec_module(_wf)


def test_season_of_maps_aug_rollover() -> None:
    dates = pd.Series(pd.to_datetime([
        "2024-10-03",  # start of 2024/25
        "2025-05-27",  # end   of 2024/25
        "2025-07-31",  # still 2024/25 (before Aug)
        "2025-08-01",  # start of 2025/26 (Aug boundary)
        "2025-10-03",  # 2025/26
        "2026-05-16",  # end of 2025/26
    ], utc=True))
    assert list(_wf._season_of(dates)) == [2024, 2024, 2024, 2025, 2025, 2025]
