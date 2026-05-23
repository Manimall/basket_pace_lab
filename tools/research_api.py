"""
Research script: Sofascore statistics API probe.

Задачи:
  1. Проверить, пропускает ли Sofascore прямые httpx-запросы (Cloudflare test).
  2. Распечатать полное дерево метрик по каждому периоду (ALL / 1ST / 2ND / …).

Run:
    python research_api.py [event_id]

    event_id по умолчанию — 12571063 (можно заменить на любой другой).
"""

import asyncio
import json
import sys
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EVENT_ID = "12571063"
BASE_URL = "https://api.sofascore.com/api/v1"
STATS_PATH = "/event/{event_id}/statistics"

# Периоды, которые выводим подробно (остальные — только заголовок)
VERBOSE_PERIODS: set[str] = {"ALL", "1ST", "2ND", "3RD", "4TH", "OT"}

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
    "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
}

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"


def _header(text: str) -> None:
    bar = "─" * 68
    print(f"\n{_c(CYAN, bar)}")
    print(f"  {_c(BOLD, text)}")
    print(f"{_c(CYAN, bar)}")


def _ok(text: str) -> None:
    print(f"  {_c(GREEN, '✓')} {text}")


def _warn(text: str) -> None:
    print(f"  {_c(YELLOW, '!')} {text}")


def _err(text: str) -> None:
    print(f"  {_c(RED, '✗')} {text}")


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def _analyze_error(response: httpx.Response) -> None:
    """Подробный разбор не-200 ответа."""
    _err(f"HTTP {response.status_code}  {response.url}")

    _header("Response headers")
    for k, v in response.headers.items():
        print(f"    {_c(DIM, k + ':'):<40} {v}")

    # Cloudflare-специфичные признаки
    cf_markers = {
        "cf-ray": "Cloudflare Ray ID — трафик проходит через CF",
        "cf-cache-status": "Cloudflare кеш-статус",
        "cf-mitigated": "Cloudflare явно заблокировал запрос",
        "x-frame-options": "Дополнительная защита фрейма",
    }
    detected: list[str] = []
    for marker, description in cf_markers.items():
        if marker in response.headers:
            detected.append(f"{marker}: {response.headers[marker]}  →  {description}")

    if detected:
        _header("Cloudflare signatures detected")
        for d in detected:
            _warn(d)

        if response.status_code == 403:
            print()
            _warn("Вердикт: Cloudflare блокирует прямые httpx-запросы.")
            _warn("Решения:")
            print("    1. Playwright (браузерные cookies + JS-challenge) — уже есть sniffer.py")
            print("    2. curl_cffi (имитация TLS fingerprint Chrome) — pip install curl_cffi")
            print("    3. Прокси / rotating residential proxies")
    else:
        _warn("Cloudflare-заголовков нет. Возможно, блокировка на уровне приложения.")

    # Тело ответа (может содержать подсказку)
    try:
        body = response.text[:600]
        if body.strip():
            _header("Response body (first 600 chars)")
            print(f"    {body}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Statistics tree printer
# ---------------------------------------------------------------------------

def _print_stat_value(name: str, home: Any, away: Any, indent: int = 6) -> None:
    pad = " " * indent
    h_str = str(home) if home is not None else _c(DIM, "—")
    a_str = str(away) if away is not None else _c(DIM, "—")
    print(f"{pad}{_c(DIM, '·')} {name:<40}  home: {_c(GREEN, h_str):<20}  away: {_c(YELLOW, a_str)}")


def _print_period(period_data: dict[str, Any], verbose: bool) -> None:
    period = period_data.get("period", "?")
    groups: list[dict[str, Any]] = period_data.get("groups", [])
    total_metrics = sum(len(g.get("statisticsItems", [])) for g in groups)

    label = _c(BOLD, f"Period: {period}") + f"  ({len(groups)} groups, {total_metrics} metrics)"
    print(f"\n  {label}")

    if not verbose:
        for g in groups:
            group_name = g.get("groupName", "?")
            items = g.get("statisticsItems", [])
            print(f"    {_c(DIM, group_name)}: {len(items)} metrics  "
                  f"[{', '.join(i.get('name','?') for i in items[:4])}{'…' if len(items) > 4 else ''}]")
        return

    for group in groups:
        group_name = group.get("groupName", "Unknown group")
        items: list[dict[str, Any]] = group.get("statisticsItems", [])
        print(f"\n    {_c(CYAN, '▸')} {_c(BOLD, group_name)}")

        for item in items:
            name: str = item.get("name", "?")
            home_val = item.get("home")
            away_val = item.get("away")
            home_pct = item.get("homePercentage")
            away_pct = item.get("awayPercentage")

            _print_stat_value(name, home_val, away_val)

            # Процентные поля (FG%, 3P%, FT% и т.д.) — если есть
            if home_pct is not None or away_pct is not None:
                _print_stat_value(
                    f"{name} %",
                    f"{home_pct:.1f}%" if home_pct is not None else None,
                    f"{away_pct:.1f}%" if away_pct is not None else None,
                    indent=8,
                )


def _print_statistics(data: dict[str, Any]) -> None:
    statistics: list[dict[str, Any]] = data.get("statistics", [])

    if not statistics:
        _warn("Ключ 'statistics' отсутствует или пуст. Полный JSON:")
        print(json.dumps(data, ensure_ascii=False, indent=2)[:1500])
        return

    _header(f"Statistics tree  ({len(statistics)} periods found)")

    for period_data in statistics:
        period = period_data.get("period", "")
        verbose = period.upper() in VERBOSE_PERIODS
        _print_period(period_data, verbose=verbose)

    # Pace-релевантные метрики — резюме
    _header("Pace-relevant metrics scan")
    pace_keywords = [
        "turnovers", "offensive rebounds", "field goals", "free throws",
        "three point", "assists", "blocks", "steals", "possessions", "pace",
        "attempted", "made",
    ]
    found: dict[str, list[str]] = {}

    all_period = next(
        (p for p in statistics if p.get("period", "").upper() == "ALL"), None
    )
    if all_period:
        for group in all_period.get("groups", []):
            for item in group.get("statisticsItems", []):
                name: str = item.get("name", "").lower()
                for kw in pace_keywords:
                    if kw in name:
                        found.setdefault(kw, []).append(item.get("name", "?"))

    if found:
        for kw, names in sorted(found.items()):
            _ok(f"{kw:<30} → {', '.join(names)}")
    else:
        _warn("Pace-метрики не найдены автоматически. Проверь дерево выше вручную.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def probe(event_id: str) -> None:
    url = BASE_URL + STATS_PATH.format(event_id=event_id)

    _header(f"Sofascore API Probe  —  event_id: {event_id}")
    print(f"  URL: {_c(CYAN, url)}\n")

    async with httpx.AsyncClient(
        headers=HEADERS,
        follow_redirects=True,
        timeout=15.0,
    ) as client:
        print(f"  Sending GET request…")
        try:
            response = await client.get(url)
        except httpx.ConnectError as exc:
            _err(f"Connection error: {exc}")
            return
        except httpx.TimeoutException:
            _err("Request timed out (15s)")
            return

        print(f"  Status: {_c(GREEN if response.status_code == 200 else RED, str(response.status_code))}")
        print(f"  Content-Type: {response.headers.get('content-type', '—')}")
        print(f"  Content-Length: {response.headers.get('content-length', '—')} bytes")

        if response.status_code != 200:
            _analyze_error(response)
            return

        _ok("Request passed through — Cloudflare не заблокировал httpx!")

        try:
            data: dict[str, Any] = response.json()
        except json.JSONDecodeError as exc:
            _err(f"Failed to parse JSON: {exc}")
            print(response.text[:500])
            return

        _print_statistics(data)

    print()


if __name__ == "__main__":
    event_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVENT_ID
    asyncio.run(probe(event_id))
