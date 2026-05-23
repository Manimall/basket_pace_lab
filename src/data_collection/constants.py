"""
Shared HTTP constants for all data-collection workers.

Centralised here so USER_AGENTS and API_HEADERS are never duplicated across
collector files. Scraping delays and thresholds live in CollectorConfig (settings.py).
"""
from __future__ import annotations

# User-Agent pool — rotated periodically to reduce bot-detection risk
USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]

# Used by ChinaNBL and collect_history.py scripts (single UA, no rotation needed)
DEFAULT_USER_AGENT = USER_AGENTS[0]

# Sofascore API headers (Playwright page.request)
SOFASCORE_API_HEADERS: dict[str, str] = {
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
}

# Flashscore base URL
FLASHSCORE_BASE_URL = "https://www.flashscore.com"

# Sofascore base URL
SOFASCORE_BASE_URL = "https://www.sofascore.com"
SOFASCORE_API_BASE = "https://api.sofascore.com/api/v1"
