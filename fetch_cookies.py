"""
Одноразовый скрипт: открывает Sofascore через Playwright,
ждёт JS-инициализации, сохраняет куки в cookies.json.

После этого run_parser_test.py подхватит куки автоматически.

Run:
    python fetch_cookies.py
"""

import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

COOKIES_PATH = Path(__file__).parent / "cookies.json"
TARGET_URL = "https://www.sofascore.com/basketball"
WAIT_SEC = 8


async def main() -> None:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
        )
        page = await context.new_page()

        print(f"Opening {TARGET_URL} ...")
        await page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=30_000)
        print(f"Waiting {WAIT_SEC}s for JS cookies to be set...")
        await asyncio.sleep(WAIT_SEC)

        cookies = await context.cookies()
        await browser.close()

    COOKIES_PATH.write_text(json.dumps(cookies, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(cookies)} cookies → {COOKIES_PATH}")
    for c in cookies:
        print(f"  {c['name']:<30} domain={c['domain']}")


if __name__ == "__main__":
    asyncio.run(main())
