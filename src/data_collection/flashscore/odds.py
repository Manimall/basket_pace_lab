"""
Flashscore odds scraper: extracts pre-match Over/Under total line.

Workflow per league:
  1. Navigate to the league results page and build {flashscore_id: pathname} index.
  2. For each match, navigate to match/#/odds-comparison/over-under and parse
     the Pinnacle row (fallback: bet365 → 1xbet → …).

Public API:
    build_href_index(page, league_path) -> dict[str, str]
    scrape_match_odds(page, flashscore_id, match_pathname) -> OddsRow | None
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from playwright.async_api import Page

from src.config import settings
from src.data_collection.constants import FLASHSCORE_BASE_URL

log = logging.getLogger(__name__)

# Sharp bookmaker first — closing Pinnacle line is the strongest market signal
BOOKMAKER_PRIORITY: tuple[str, ...] = (
    "pinnacle",
    "bet365",
    "1xbet",
    "bwin",
    "unibet",
    "william hill",
    "betway",
)

# Basketball Over/Under totals by league: PBA 160-195, Taiwan 150-185,
# LegaA/EL ~155-175, NBA 210-240 → safe range [130, 279]
_TOTAL_RE = re.compile(r"\b(1[3-9]\d|2[0-7]\d)(?:\.5)?\b")

_ODDS_HASH = "#/odds-comparison/over-under"

_PAGE_LOAD_TIMEOUT_MS: int = 20_000
_ODDS_LOAD_WAIT_SEC: float = 3.0
_NAV_WAIT_SEC: float = 2.0

# -----------------------------------------------------------------------
# JavaScript helpers (injected into Flashscore page context)
# -----------------------------------------------------------------------

_JS_HREF_INDEX = r"""
() => {
    // Return {matchId: pathname} from all basketball match links.
    // Flashscore match hrefs: /sport/country/league/team1-team2/ID/
    const idx = {};
    document.querySelectorAll('a[href*="/basketball/"]').forEach(a => {
        const m = a.pathname.match(/\/([A-Za-z0-9]{8,12})\/$/);
        if (m) idx[m[1]] = a.pathname;
    });
    return idx;
}
"""

_JS_EXTRACT_ODDS = r"""
(bookmakers) => {
    // Regex for basketball total lines (130–279, optionally ending in .5)
    const TOT_RE = /\b(1[3-9]\d|2[0-7]\d)(?:\.5)?\b/g;
    function extractTotals(text) {
        return [...text.matchAll(TOT_RE)].map(m => parseFloat(m[0]));
    }

    // Try Flashscore's common row selectors in priority order
    const ROW_SELECTORS = [
        '[class*="ui-table__row"]',
        '[class*="oddsRow"]',
        '[class*="odds__row"]',
        'tr',
    ];
    let rows = [];
    for (const sel of ROW_SELECTORS) {
        const found = Array.from(document.querySelectorAll(sel));
        if (found.length > 2) { rows = found; break; }
    }
    // Keep rows with meaningful content, skip bloated containers
    rows = rows.filter(r => {
        const len = r.textContent.trim().length;
        return len > 5 && len < 600;
    });

    for (const bm of bookmakers) {
        for (const row of rows) {
            if (!row.textContent.toLowerCase().includes(bm)) continue;
            const tots = extractTotals(row.textContent);
            if (tots.length === 0) continue;
            // Flashscore shows opening line first, closing line last in the same row
            return {
                total_open:  tots[0],
                total_close: tots[tots.length - 1],
                bookmaker:   bm,
            };
        }
    }
    return null;
}
"""


# -----------------------------------------------------------------------
# Public data types
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class OddsRow:
    total_open:  float | None   # opening O/U line
    total_close: float | None   # closing O/U line → stored as matches.total_line
    bookmaker:   str            # source bookmaker (e.g. "pinnacle")
    scraped_at:  datetime


# -----------------------------------------------------------------------
# Public functions
# -----------------------------------------------------------------------

async def build_href_index(page: Page, league_path: str) -> dict[str, str]:
    """
    Scrape the league results page and return {flashscore_id: pathname}.

    Clicks "show more" until exhausted so historical matches are indexed.
    """
    url = FLASHSCORE_BASE_URL + league_path + "results/"
    log.info("  Building href index: %s", url)
    await page.goto(url, wait_until="domcontentloaded", timeout=_PAGE_LOAD_TIMEOUT_MS)
    await asyncio.sleep(_NAV_WAIT_SEC)

    clicks = 0
    while True:
        btn = page.locator(
            'button:has-text("show more matches"), a:has-text("show more matches")'
        )
        try:
            if not await btn.first.is_visible(timeout=2_000):
                break
            await btn.first.click()
            await asyncio.sleep(settings.collector.show_more_delay)
            clicks += 1
        except Exception:
            break

    if clicks:
        log.debug("  'Show more' clicked %d times", clicks)

    idx: dict[str, str] = await page.evaluate(_JS_HREF_INDEX)
    log.info("  Href index: %d entries", len(idx))
    return idx


async def scrape_match_odds(
    page: Page,
    flashscore_id: str,
    match_pathname: str,
) -> OddsRow | None:
    """
    Navigate to the match Over/Under odds page and extract the best available line.

    Priority: pinnacle → bet365 → 1xbet → …
    Returns None when the page fails to load or no bookmaker row is found.
    """
    odds_url = f"{FLASHSCORE_BASE_URL}{match_pathname}{_ODDS_HASH}"
    try:
        await page.goto(
            odds_url, wait_until="domcontentloaded", timeout=_PAGE_LOAD_TIMEOUT_MS
        )
        await asyncio.sleep(_ODDS_LOAD_WAIT_SEC)
    except Exception as exc:
        log.warning("  [%s] Odds page load failed: %s", flashscore_id, exc)
        return None

    try:
        result: dict[str, Any] | None = await page.evaluate(
            _JS_EXTRACT_ODDS, list(BOOKMAKER_PRIORITY)
        )
    except Exception as exc:
        log.warning("  [%s] JS evaluation error: %s", flashscore_id, exc)
        return None

    if not result:
        log.debug("  [%s] No bookmaker row found on odds page", flashscore_id)
        return None

    total_open  = result.get("total_open")
    total_close = result.get("total_close")

    if total_open is None and total_close is None:
        log.debug("  [%s] Bookmaker row found but total line not extractable", flashscore_id)
        return None

    return OddsRow(
        total_open=total_open,
        total_close=total_close if total_close is not None else total_open,
        bookmaker=result["bookmaker"],
        scraped_at=datetime.now(tz=timezone.utc),
    )
