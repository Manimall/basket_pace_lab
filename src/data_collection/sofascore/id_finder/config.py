"""Tunable constants for the Sofascore ID finder (zero hardcode at call sites)."""
from __future__ import annotations

from pathlib import Path

# cookies.json lives at the project root (same file the Go scout reads).
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
COOKIES_PATH: Path = _PROJECT_ROOT / "cookies.json"

BASE_URL: str = "https://api.sofascore.com/api/v1"
USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
IMPERSONATE_PROFILE: str = "chrome124"

BASE_HEADERS: dict[str, str] = {
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin":          "https://www.sofascore.com",
    "Referer":         "https://www.sofascore.com/",
    "Cache-Control":   "no-cache",
    "User-Agent":      USER_AGENT,
}

# HTTP / pagination
HTTP_TIMEOUT_SEC:   int   = 15
MAX_PAGES:          int   = 50
PAGE_DELAY_SEC:     float = 1.5     # between paginated requests within a season
INTER_SEASON_SLEEP: float = 3.0     # before each season fetch — eases WAF rate-limit
HTTP_OK:            int   = 200

# Matching
DEFAULT_THRESHOLD:  float = 0.50    # min fuzzy similarity to accept a match
DATE_WINDOW_DAYS:   int   = 1       # |db_date - sofa_date| ≤ this to be a candidate
HINT_WINDOW_DAYS:   int   = 3       # wider window when explaining an unmatched row
THRESHOLD_FLOOR:    float = 0.35    # lower bound for the "retry with lower threshold" tip
THRESHOLD_STEP:     float = 0.10    # suggested decrement for that tip
MAX_UNMATCHED_HINTS: int  = 8       # cap unmatched examples printed per league
