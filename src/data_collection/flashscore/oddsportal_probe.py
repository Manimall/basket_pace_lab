"""
Quick probe — check if Oddsportal has VTB Over/Under data.

Navigates to the VTB results page, collects match URLs, then opens one
match and checks whether an O/U tab / handicap values are available.

Usage:
    python -m src.data_collection.flashscore.oddsportal_probe
"""
from __future__ import annotations

import asyncio
import logging
import random
import sys

from playwright.async_api import async_playwright

from src.data_collection.constants import USER_AGENTS

log = logging.getLogger("oddsportal_probe")

_BASE        = "https://www.oddsportal.com"
_RESULTS_URL = _BASE + "/basketball/russia/vtb-united-league/results/"


async def run() -> None:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ua      = random.choice(USER_AGENTS)
        ctx     = await browser.new_context(
            user_agent=ua,
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        page = await ctx.new_page()

        log.info("GET %s", _RESULTS_URL)
        try:
            await page.goto(_RESULTS_URL, wait_until="domcontentloaded", timeout=30_000)
        except Exception as exc:
            log.error("Failed to load results page: %s", exc)
            await browser.close()
            return

        await asyncio.sleep(4)

        title = await page.title()
        log.info("Page title: %s", title)

        # Check for Cloudflare block
        if "Just a moment" in title or "Cloudflare" in title:
            log.warning("Blocked by Cloudflare — Oddsportal not accessible headlessly")
            await browser.close()
            return

        # Try to find match rows
        match_links: list[str] = await page.evaluate("""() => {
            const anchors = Array.from(document.querySelectorAll('a[href*="/basketball/russia/vtb"]'));
            return anchors
                .map(a => a.href)
                .filter(h => h.includes('-') && !h.includes('results'))
                .slice(0, 5);
        }""")

        log.info("Match links found: %d", len(match_links))
        for url in match_links:
            log.info("  %s", url)

        if not match_links:
            # Try alternative selector
            html_sample = await page.evaluate(
                "() => document.body.innerHTML.slice(0, 2000)"
            )
            log.info("Page HTML sample:\n%s", html_sample)
            await browser.close()
            return

        # Navigate into first match and check O/U tab
        match_url = match_links[0]
        ou_url    = match_url.rstrip("/") + "/#over-under"
        log.info("Opening match O/U page: %s", ou_url)
        await page.goto(ou_url, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(4)

        # Look for handicap values (numbers like 160.5, 175.0 etc.)
        ou_data: dict = await page.evaluate("""() => {
            const rows = Array.from(document.querySelectorAll('tr, [data-ou]'));
            const handicaps = new Set();
            rows.forEach(r => {
                const t = r.textContent || '';
                const m = t.match(/\\b(1[3-9]\\d\\.5|2[0-3]\\d\\.5|\\d{3,4}\\.5)\\b/g);
                if (m) m.forEach(v => handicaps.add(v));
            });
            const title = document.title;
            const hasOUTab = !!document.querySelector('[href*="over-under"], [data-tab*="over"], button');
            return {
                title,
                hasOUTab,
                handicaps: Array.from(handicaps).slice(0, 10),
                bodyLen: document.body.innerText.length,
            };
        }""")

        log.info("Match page title: %s", ou_data.get("title"))
        log.info("Body text length: %d", ou_data.get("bodyLen", 0))
        log.info("O/U tab present: %s", ou_data.get("hasOUTab"))
        log.info("Handicap values found: %s", ou_data.get("handicaps"))

        if ou_data.get("handicaps"):
            log.info("✓ Oddsportal HAS O/U data for VTB — full scraper is viable")
        else:
            log.warning("✗ No handicap values found — VTB may not have O/U on Oddsportal")

        await browser.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        stream=sys.stdout,
    )
    asyncio.run(run())
