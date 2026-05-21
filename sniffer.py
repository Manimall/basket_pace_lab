"""
Network sniffer for basketball API reverse engineering.

Usage:
    pip install playwright && playwright install chromium
    python sniffer.py

Меняй TARGET_URL на любую страницу матча нужного источника.
Скрипт перехватывает все JSON-ответы и выводит URL + превью тела
для эндпоинтов, похожих на статистику / box score.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from playwright.async_api import Browser, Page, Request, Response, async_playwright

# ---------------------------------------------------------------------------
# Config — меняй под нужный источник / матч
# ---------------------------------------------------------------------------

# Sofascore: любой завершённый матч NBA / VTB / EuroLeague
# Пример VTB: https://www.sofascore.com/cska-moscow-unics/xWsxWGsb#id:12345678
# Пример NBA: https://www.sofascore.com/boston-celtics-new-york-knicks/BdbdbGdb#id:12345678
TARGET_URL = "https://www.sofascore.com/cska-moscow-unics/xXsxXGsb#id:12571063"

# Альтернативные источники (раскомментируй нужный):
# TARGET_URL = "https://www.basketball-reference.com/boxscores/202506010BOS.html"
# TARGET_URL = "https://www.flashscore.com/match/basketball/XXXXXXXX/"
# TARGET_URL = "https://vtb-league.com/ru/game/XXXXXXX/"

PAGE_LOAD_WAIT_SEC = 12  # время ожидания после загрузки страницы

# Ключевые слова в URL — признак статистического эндпоинта
STAT_URL_KEYWORDS: list[str] = [
    "boxscore", "box-score", "box_score",
    "statistic", "statistics", "stats",
    "summary", "periods", "quarter", "period",
    "lineups", "lineup",
    "incident", "events",
    "game-detail", "game_detail",
    "score", "scores",
]

# Домены источников данных, которые точно не нужны (реклама, аналитика)
NOISE_DOMAINS: list[str] = [
    "google", "facebook", "doubleclick", "analytics",
    "hotjar", "sentry", "mixpanel", "amplitude",
    "cloudflare", "recaptcha", "braze", "onesignal",
]

PREVIEW_LENGTH = 400  # символов тела для превью

# ---------------------------------------------------------------------------
# Captured data
# ---------------------------------------------------------------------------

@dataclass
class CapturedEndpoint:
    url: str
    content_type: str
    is_stat: bool
    preview: str
    captured_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


captured: list[CapturedEndpoint] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_noise(url: str) -> bool:
    return any(domain in url for domain in NOISE_DOMAINS)


def _is_stat_url(url: str) -> bool:
    url_lower = url.lower()
    return any(kw in url_lower for kw in STAT_URL_KEYWORDS)


def _pretty_json_preview(raw: str, length: int = PREVIEW_LENGTH) -> str:
    """Пытается распарсить JSON и вернуть компактный превью."""
    try:
        parsed = json.loads(raw)
        compact = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        return compact[:length] + ("…" if len(compact) > length else "")
    except (json.JSONDecodeError, ValueError):
        return raw[:length]


# ---------------------------------------------------------------------------
# Response handler
# ---------------------------------------------------------------------------

async def on_response(response: Response) -> None:
    url = response.url
    status = response.status

    if _is_noise(url):
        return

    content_type: str = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        return

    is_stat = _is_stat_url(url)
    label = "  [STAT]" if is_stat else "        "
    print(f"{label} {status} {url}")

    preview = ""
    if is_stat:
        try:
            body = await response.text()
            preview = _pretty_json_preview(body)
            # Отступ для читаемости
            print(f"         └─ {preview}\n")
        except Exception as exc:
            print(f"         └─ [body read error: {exc}]\n")

    captured.append(
        CapturedEndpoint(
            url=url,
            content_type=content_type,
            is_stat=is_stat,
            preview=preview,
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_sniffer(target_url: str) -> None:
    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(
            headless=False,  # True для CI / headless-окружений
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
            locale="ru-RU",
            timezone_id="Europe/Moscow",
        )

        # Блокируем тяжёлые бинарные ресурсы — только HTML/JS/XHR нас интересует
        await context.route(
            re.compile(r"\.(png|jpg|jpeg|gif|svg|woff2?|ttf|eot|mp4|webm)(\?.*)?$"),
            lambda route, _: route.abort(),
        )

        page: Page = await context.new_page()
        page.on("response", on_response)

        print(f"\n{'='*70}")
        print(f"  Target : {target_url}")
        print(f"  Wait   : {PAGE_LOAD_WAIT_SEC}s after load")
        print(f"  Filter : JSON responses (stat endpoints marked [STAT])")
        print(f"{'='*70}\n")

        await page.goto(target_url, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(PAGE_LOAD_WAIT_SEC)

        await browser.close()

    # ---------------------------------------------------------------------------
    # Summary report
    # ---------------------------------------------------------------------------
    stat_endpoints = [e for e in captured if e.is_stat]
    all_json = [e for e in captured if not e.is_stat]

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total JSON responses : {len(captured)}")
    print(f"  Stat-like endpoints  : {len(stat_endpoints)}")

    if stat_endpoints:
        print(f"\n  --- STAT ENDPOINTS (copy these) ---")
        for ep in stat_endpoints:
            print(f"  {ep.url}")

    if all_json:
        print(f"\n  --- ALL OTHER JSON ENDPOINTS ---")
        for ep in all_json:
            print(f"  {ep.url}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(run_sniffer(TARGET_URL))
