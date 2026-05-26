"""
Flashscore page-scraping helpers for the enricher.

Responsibilities:
  - Build per-league results index from the Flashscore results page
  - Fuzzy-match DB records to index entries
  - Build QuarterStatRow list from matched entry
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import date
from typing import Any

from playwright.async_api import Page

from src.config import settings
from src.data_collection.constants import FLASHSCORE_BASE_URL
from src.data_collection.flashscore.norm import FsMatch, QScore, _norm, _sim
from src.data_collection.flashscore.odds import dismiss_overlays
from src.database.crud import QuarterStatRow
from src.database.models import PeriodType

log = logging.getLogger(__name__)

# tournament_name (DB) → Flashscore URL path
LEAGUE_PATHS: dict[str, str] = {
    "NBA":            "/basketball/usa/nba/",
    "EuroLeague":     "/basketball/europe/euroleague/",
    "VTB":            "/basketball/russia/vtb-united-league/",
    "ACB":            "/basketball/spain/acb/",
    "LegaA":          "/basketball/italy/lega-a/",
    "BBL":            "/basketball/germany/bbl/",
    "LNB":            "/basketball/france/lnb/",
    "NBL":            "/basketball/australia/nbl/",
    "CBA":            "/basketball/china/cba/",
    "BLeague":        "/basketball/japan/b-league/",
    "LNBP":           "/basketball/mexico/lnbp/",
    "PBA":            "/basketball/philippines/pba-philippine-cup/",
    "PBA_Comm":       "/basketball/philippines/pba-commissioner-s-cup/",
    "PBA_Gov":        "/basketball/philippines/pba-governors-cup/",
    "Taiwan_PLeague": "/basketball/taiwan/p-league/",
    "Taiwan_TPBL":    "/basketball/taiwan/tpbl/",
    # Adriatic League (sponsor name on Flashscore: "AdmiralBet ABA League")
    "ABA":            "/basketball/europe/admiralbet-aba-league/",
}

_JS_EXTRACT_MATCHES = r"""
() => {
    const rows = [];
    document.querySelectorAll('.event__match').forEach(el => {
        const homeEl = el.querySelector('.event__homeParticipant');
        const awayEl = el.querySelector('.event__awayParticipant');
        const timeEl = el.querySelector('.event__time');
        const firstName = (node) => {
            if (!node) return '';
            for (const child of node.childNodes) {
                if (child.nodeType === Node.TEXT_NODE) { const t = child.textContent.trim(); if (t) return t; }
                if (child.nodeType === Node.ELEMENT_NODE) { const t = child.textContent.trim().split('\n')[0].trim(); if (t) return t; }
            }
            return node.textContent.trim().split('\n')[0].trim();
        };
        const home = firstName(homeEl), away = firstName(awayEl);
        const timeStr = timeEl ? timeEl.textContent.trim() : '';
        let elId = el.id || el.getAttribute('data-id') || '';
        const idM = elId.match(/([A-Za-z0-9]{8,})$/);
        const fsId = idM ? idM[1] : '';
        const qScores = [];
        for (let q = 1; q <= 4; q++) {
            const hEl = el.querySelector('.event__part--home.event__part--' + q);
            const aEl = el.querySelector('.event__part--away.event__part--' + q);
            qScores.push([hEl ? (parseInt(hEl.textContent.trim()) || null) : null,
                          aEl ? (parseInt(aEl.textContent.trim()) || null) : null]);
        }
        if (home && away && fsId) rows.push({ timeStr, home, away, fsId, qScores });
    });
    return rows;
}
"""


def _parse_fs_date(
    time_str: str,
    season_years: tuple[int, int] | None = None,
) -> date | None:
    m = re.match(r"(\d{1,2})\.(\d{1,2})\.", time_str.strip())
    if not m:
        return None
    day, month = int(m.group(1)), int(m.group(2))

    if season_years:
        # Sports seasons span two calendar years: months Jul-Dec belong to year[0],
        # months Jan-Jun belong to year[1].
        year = season_years[0] if month >= 7 else season_years[1]
        try:
            return date(year, month, day)
        except ValueError:
            return None

    today = date.today()
    best: date | None = None
    for year in [today.year, today.year - 1]:
        try:
            d = date(year, month, day)
            if d <= today and abs((today - d).days) <= 540:
                if best is None or d > best:
                    best = d
        except ValueError:
            pass
    return best


def _infer_season_years(url: str) -> tuple[int, int] | None:
    """Extract (year1, year2) from URLs like '.../nba-2024-2025/...'."""
    m = re.search(r"-(\d{4})-(\d{4})/", url)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


async def build_index_from_url(page: Page, url: str) -> list[FsMatch]:
    """Build a Flashscore match index from an arbitrary results page URL.

    Returns an empty list (with a warning) if the page fails to load.
    """
    log.info("  Building index: %s", url)
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=25_000)
    except Exception as exc:
        log.warning("  Failed to load %s: %s", url, exc)
        return []
    await asyncio.sleep(2)
    # Per-page dismissal: gambling-ads / lang-selector dialogs reappear after
    # navigation and intercept the "Show more matches" click. Discovered on
    # the AdmiralBet ABA League page; same fix applies to any results page.
    await dismiss_overlays(page)

    clicks = 0
    while True:
        btn = page.locator('button:has-text("show more matches"), a:has-text("show more matches")')
        try:
            if not await btn.first.is_visible(timeout=2_000):
                break
            await btn.first.click()
            await asyncio.sleep(settings.collector.show_more_delay)
            clicks += 1
        except Exception:
            break

    log.info("  'Show more' clicked %d times", clicks)
    raw_rows: list[dict[str, Any]] = await page.evaluate(_JS_EXTRACT_MATCHES)
    log.info("  Raw rows from JS: %d", len(raw_rows))

    season_years = _infer_season_years(url)
    entries: list[FsMatch] = []
    for r in raw_rows:
        if not r.get("fsId"):
            continue
        entries.append(FsMatch(
            fs_id=r["fsId"], date_str=r["timeStr"],
            match_date=_parse_fs_date(r["timeStr"], season_years),
            home_raw=r["home"], away_raw=r["away"],
            home_norm=_norm(r["home"]), away_norm=_norm(r["away"]),
            q_scores=[(q[0], q[1]) for q in r["qScores"]],
        ))

    has_scores = sum(1 for e in entries if any(q[0] is not None for q in e.q_scores))
    log.info("  Index built: %d entries (%d have quarter scores)", len(entries), has_scores)
    return entries


async def build_league_index(page: Page, league_key: str) -> list[FsMatch]:
    """Build index for a known league key (looks up URL from LEAGUE_PATHS)."""
    path = LEAGUE_PATHS[league_key]
    return await build_index_from_url(page, FLASHSCORE_BASE_URL + path + "results/")


def find_match(
    index: list[FsMatch],
    target_date: date,
    home_name: str,
    away_name: str,
    threshold: float | None = None,
) -> FsMatch | None:
    thr = threshold if threshold is not None else settings.collector.match_threshold
    h_norm, a_norm = _norm(home_name), _norm(away_name)
    best: FsMatch | None = None
    best_score = 0.0
    for entry in index:
        if entry.match_date and abs((entry.match_date - target_date).days) > 1:
            continue
        combined = (_sim(h_norm, entry.home_norm) + _sim(a_norm, entry.away_norm)) / 2
        if combined > best_score:
            best_score, best = combined, entry
    return best if best_score >= thr else None


def build_quarter_rows(fs_match: FsMatch) -> list[QuarterStatRow] | None:
    if len(fs_match.q_scores) < 4 or all(q[0] is None for q in fs_match.q_scores):
        return None
    return [
        QuarterStatRow(
            period_number=p, period_type=PeriodType.QUARTER,
            home_score=h, away_score=a,
            home_fga=None, away_fga=None, home_fta=None, away_fta=None,
            home_off_reb=None, away_off_reb=None, home_turnovers=None, away_turnovers=None,
            home_possessions=None, away_possessions=None, home_pace=None, away_pace=None,
        )
        for p, (h, a) in enumerate(fs_match.q_scores[:4], 1)
    ]
