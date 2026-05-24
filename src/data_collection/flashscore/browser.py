"""Playwright session factory for Flashscore scrapers."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from playwright.async_api import Browser, Page

from src.data_collection.constants import FLASHSCORE_BASE_URL, USER_AGENTS
from src.data_collection.flashscore.odds import dismiss_overlays

log = logging.getLogger(__name__)


async def make_flashscore_session(
    pw: Any,
    warmup_path: str = "/basketball/",
) -> tuple[Browser, Page]:
    """Launch Chromium, warm up the session, return (browser, page).

    Caller owns the browser lifecycle — must call browser.close().
    """
    browser = await pw.chromium.launch(headless=True)
    ua = random.choice(USER_AGENTS)
    ctx = await browser.new_context(user_agent=ua)
    page = await ctx.new_page()
    log.info("Warming up Flashscore session (UA: %s…)", ua[:45])
    await page.goto(
        FLASHSCORE_BASE_URL + warmup_path,
        wait_until="domcontentloaded",
        timeout=20_000,
    )
    await asyncio.sleep(3)
    await dismiss_overlays(page)
    return browser, page
