"""
Flashscore odds scraper: extracts pre-match Over/Under total line.

API-interception approach:
  1. Navigate to match odds page.
  2. Click the 'Over/Under' tab — triggers GraphQL calls on global.ds.lsapp.eu.
  3. Capture OVER_UNDER API responses (one per bookmaker, fired by the page).
  4. From all bookmaker opportunities, pick the handicap (total line) whose
     Over/Under payout odds are most symmetric (closest-to-even = sharpest line).

The previous HTML-parsing approach could not recover the actual handicap value:
Flashscore's O/U comparison page renders only decimal payout odds, not the line.
The total line is available only via their internal GraphQL API.

Public API:
    prepare_results_page(page, league_path) -> int
    get_match_url_via_click(page, flashscore_id) -> str | None
    scrape_match_odds(page, flashscore_id, match_url) -> OddsRow | None
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from playwright.async_api import Page

from src.config import settings
from src.data_collection.constants import FLASHSCORE_BASE_URL

log = logging.getLogger(__name__)

_ODDS_HASH            = "#/odds-comparison/over-under"
_PAGE_LOAD_TIMEOUT_MS : int   = 20_000
_CLICK_NAV_TIMEOUT_MS : int   = 10_000
_ODDS_LOAD_WAIT_SEC   : float = 3.5
_NAV_WAIT_SEC         : float = 2.0
_BACK_WAIT_SEC        : float = 1.0
_OU_TAB_WAIT_SEC      : float = 4.0   # wait for GraphQL responses after tab click


# -----------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class OddsRow:
    total_open:  float | None   # opening O/U line (currently unavailable from API)
    total_close: float | None   # closing O/U handicap → stored as matches.total_line
    bookmaker:   str            # Flashscore bookmaker_id (string)
    scraped_at:  datetime


# -----------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------

def _parse_best_opportunity(opportunities: list[dict]) -> tuple[float | None, float]:
    """
    From a list of Over/Under opportunities (different lines per bookmaker),
    pick the handicap whose payout odds are most symmetric.
    Returns (handicap, asymmetry_score).
    """
    best_handicap: float | None = None
    best_score:    float        = float("inf")
    for opp in opportunities:
        try:
            over_v   = float(opp["over"]["value"])
            under_v  = float(opp["under"]["value"])
            handicap = float(opp["handicap"]["value"])
        except (KeyError, TypeError, ValueError):
            continue
        score = abs(over_v - under_v) / (over_v + under_v)
        if score < best_score:
            best_score    = score
            best_handicap = handicap
    return best_handicap, best_score


def _extract_total_line(ou_responses: list[dict]) -> tuple[float | None, str]:
    """
    From captured OVER_UNDER API responses, select the bookmaker whose best
    opportunity has the most symmetric odds (= sharpest/most reliable line).
    Returns (total_close, bookmaker_id_str).
    """
    candidates: list[tuple[float, float, str]] = []  # (asymmetry, handicap, bm_id)
    for resp in ou_responses:
        bm_data = (resp.get("data") or {}).get("findPrematchOddsForBookmaker")
        if not bm_data:
            continue
        bm_id    = str(bm_data.get("bookmakerId", ""))
        opps     = bm_data.get("opportunities") or []
        handicap, score = _parse_best_opportunity(opps)
        if handicap is not None:
            candidates.append((score, handicap, bm_id))
    if not candidates:
        return None, ""
    candidates.sort()                      # smallest asymmetry first
    _, handicap, bm_id = candidates[0]
    return handicap, bm_id


# -----------------------------------------------------------------------
# Page helpers
# -----------------------------------------------------------------------

async def _click_show_more_loop(page: Page) -> int:
    """Click 'show more matches' until the button disappears. Returns click count."""
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
    return clicks


async def prepare_results_page(page: Page, league_path: str) -> int:
    """
    Navigate to the league results page and expand all matches.
    Returns the number of match elements visible after expansion.
    """
    url = FLASHSCORE_BASE_URL + league_path + "results/"
    log.info("  Loading: %s", url)
    await page.goto(url, wait_until="domcontentloaded", timeout=_PAGE_LOAD_TIMEOUT_MS)
    await asyncio.sleep(_NAV_WAIT_SEC)
    clicks = await _click_show_more_loop(page)
    count: int = await page.evaluate(
        "() => document.querySelectorAll('.event__match').length"
    )
    log.info("  %d matches loaded (%d show-more clicks)", count, clicks)
    return count


async def get_match_url_via_click(page: Page, flashscore_id: str) -> str | None:
    """
    Click the match element on the results page to resolve the full URL.
    Navigates back after capturing the URL. Returns None if not found.
    """
    await dismiss_overlays(page)   # lang selector / GDPR may reappear after go_back()
    selector = f'[id$="_{flashscore_id}"]'
    try:
        loc = page.locator(selector).first
        if not await loc.is_visible(timeout=2_000):
            log.debug("  [%s] Element not found on results page", flashscore_id)
            return None
        async with page.expect_navigation(
            wait_until="domcontentloaded",
            timeout=_CLICK_NAV_TIMEOUT_MS,
        ):
            await loc.click()
        match_url = page.url
        await page.go_back(wait_until="domcontentloaded", timeout=_PAGE_LOAD_TIMEOUT_MS)
        await asyncio.sleep(_BACK_WAIT_SEC)
        return match_url
    except Exception as exc:
        log.warning("  [%s] Click navigation failed: %s", flashscore_id, exc)
        return None


async def dismiss_overlays(page: Page) -> None:
    """
    Remove DOM elements that intercept pointer events on Flashscore pages:
      - language/country selector dialog (wcl-dialogWrapper) — removed entirely
        (clicking its confirm button may trigger a locale redirect)
      - OneTrust GDPR consent banner — accepted via JS click (safe, no redirect)
    """
    try:
        await page.evaluate("""() => {
            // Language/region dialog and GDPR overlay — make non-interactive without
            // touching React state (removing nodes breaks SPA routing).
            document.querySelectorAll(
                '[data-testid="wcl-dialog-wrapper"], [data-testid="wcl-dialog-overlay"], #onetrust-consent-sdk'
            ).forEach(el => {
                el.style.pointerEvents = 'none';
                el.style.display = 'none';
            });
        }""")
    except Exception:
        pass  # page may be mid-navigation; overlay dismissal is best-effort
    await asyncio.sleep(0.2)


async def _click_ou_tab(page: Page) -> bool:
    """Find and click the 'Over/Under' odds tab. Returns True if found."""
    return bool(await page.evaluate("""() => {
        const all = Array.from(document.querySelectorAll('button, [role="tab"]'));
        const btn = all.find(el => el.textContent.trim() === 'Over/Under');
        if (!btn) return false;
        btn.scrollIntoView({block: 'center'});
        btn.click();
        return true;
    }"""))


async def scrape_match_odds(
    page: Page,
    flashscore_id: str,
    match_url: str,
) -> OddsRow | None:
    """
    Navigate to match O/U odds page, click the Over/Under tab, intercept
    Flashscore's GraphQL API responses, and extract the closing total line.

    Uses a dedicated page so the caller's navigation state is unaffected.
    Returns None when the page fails or no O/U data is available.
    """
    ou_responses: list[dict] = []
    done = [False]  # mutable flag — avoids need for page.off() which is version-dependent

    async def _capture(resp: Any) -> None:
        if done[0]:
            return
        url = resp.url
        if "OVER_UNDER" in url and "lsapp.eu" in url and flashscore_id in url:
            try:
                ou_responses.append(await resp.json())
            except Exception:
                pass

    page.on("response", _capture)
    try:
        base = match_url.split("#")[0]
        try:
            await page.goto(
                base + _ODDS_HASH, wait_until="domcontentloaded",
                timeout=_PAGE_LOAD_TIMEOUT_MS,
            )
        except Exception as exc:
            log.warning("  [%s] Odds page load failed: %s", flashscore_id, exc)
            return None

        await asyncio.sleep(_ODDS_LOAD_WAIT_SEC)
        await dismiss_overlays(page)

        if not await _click_ou_tab(page):
            log.debug("  [%s] Over/Under tab not found", flashscore_id)
            return None

        await asyncio.sleep(_OU_TAB_WAIT_SEC)
    finally:
        done[0] = True

    if not ou_responses:
        log.debug("  [%s] No O/U API responses captured", flashscore_id)
        return None

    total_close, bm_id = _extract_total_line(ou_responses)
    if total_close is None:
        log.debug(
            "  [%s] Could not parse total line from %d responses",
            flashscore_id, len(ou_responses),
        )
        return None

    log.debug(
        "  [%s] total=%.1f  bm=%s  (%d responses)",
        flashscore_id, total_close, bm_id, len(ou_responses),
    )
    return OddsRow(
        total_open=None,
        total_close=total_close,
        bookmaker=bm_id,
        scraped_at=datetime.now(tz=timezone.utc),
    )
