"""Pure parsing helpers for the Flashscore results-page collector.

I/O-free string/date parsers extracted from ``collector_page.py`` to keep that
module under the 250-line limit and to make the date heuristics unit-testable.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone

# Defaults applied when Flashscore omits the time / year component.
_DEFAULT_HOUR:   int = 12   # noon — avoids day-boundary drift when tipoff time is absent
_DEFAULT_MINUTE: int = 0
# A bare "DD.MM." date is resolved to the most recent matching day within this
# look-back window (covers up to ~1.5 seasons of historical results pages).
_MAX_LOOKBACK_DAYS: int = 540
# Number of leading words combined into a team abbreviation (e.g. "Los Angeles
# Lakers" → "LAL").
_ABBREV_WORDS: int = 3
_ABBREV_FALLBACK_CHARS: int = 3

_FULL_DATE   = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})(?:\s+(\d{2}):(\d{2}))?")
_DATE_TIME   = re.compile(r"(\d{1,2})\.(\d{1,2})\.\s*(\d{2}):(\d{2})")
_DATE_ONLY   = re.compile(r"(\d{1,2})\.(\d{1,2})\.")


def abbrev(name: str) -> str:
    """Derive a short uppercase team code from a full team name."""
    words = name.split()
    if len(words) >= 2:
        return "".join(w[0] for w in words[:_ABBREV_WORDS]).upper()
    return name[:_ABBREV_FALLBACK_CHARS].upper()


def _make_utc(year: int, month: int, day: int, hour: int, minute: int) -> datetime | None:
    try:
        return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_fs_datetime(time_str: str) -> datetime | None:
    """Parse a Flashscore time string into a UTC datetime (or None).

    Handles three Flashscore formats: full ``DD.MM.YYYY [HH:MM]``,
    year-less ``DD.MM. HH:MM``, and bare ``DD.MM.`` — the latter two resolve
    the year to the most recent matching day within ``_MAX_LOOKBACK_DAYS``.
    """
    s = time_str.strip()

    full = _FULL_DATE.match(s)
    if full:
        hour   = int(full.group(4)) if full.group(4) else _DEFAULT_HOUR
        minute = int(full.group(5)) if full.group(5) else _DEFAULT_MINUTE
        return _make_utc(int(full.group(3)), int(full.group(2)), int(full.group(1)), hour, minute)

    dt = _DATE_TIME.match(s)
    if dt:
        day, month = int(dt.group(1)), int(dt.group(2))
        hour, minute = int(dt.group(3)), int(dt.group(4))
    else:
        only = _DATE_ONLY.match(s)
        if not only:
            return None
        day, month = int(only.group(1)), int(only.group(2))
        hour, minute = _DEFAULT_HOUR, _DEFAULT_MINUTE

    best = _resolve_recent_year(month, day)
    if best is None:
        return None
    return _make_utc(best.year, best.month, best.day, hour, minute)


def _resolve_recent_year(month: int, day: int) -> date | None:
    """Pick the most recent past date matching month/day within the look-back window."""
    today = date.today()
    best: date | None = None
    for year in (today.year, today.year - 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate <= today and (today - candidate).days <= _MAX_LOOKBACK_DAYS:
            if best is None or candidate > best:
                best = candidate
    return best
