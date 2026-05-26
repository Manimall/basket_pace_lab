"""
Flashscore collector page-scraping helpers.

Data structures, JS extractor, date parser, and DB persistence for
FlashscoreCollector. Isolated here to keep collector.py under 250 lines.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from playwright.async_api import Page

from src.config import settings
from src.data_collection.constants import FLASHSCORE_BASE_URL
from src.data_collection.flashscore.odds import dismiss_overlays
from src.database.crud import QuarterStatRow, get_or_create_team, save_quarter_stats, upsert_match
from src.database.models import MatchStatus, PeriodType, SeasonType

log = logging.getLogger(__name__)

QScore = tuple[int | None, int | None]

# league_key → config
COLLECTOR_LEAGUES: dict[str, dict] = {
    "LegaA": {
        "path":            "/basketball/italy/lega-a/",
        "tournament_name": "LegaA",
        "season":          "Lega A 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "PBA_PhilCup": {
        "path":            "/basketball/philippines/pba-philippine-cup/",
        "tournament_name": "PBA_PhilCup",
        "season":          "PBA Philippine Cup 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "PBA_CommCup": {
        "path":            "/basketball/philippines/pba-commissioner-s-cup/",
        "tournament_name": "PBA_CommCup",
        "season":          "PBA Commissioner's Cup 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "PBA_GovCup": {
        "path":            "/basketball/philippines/pba-governors-cup/",
        "tournament_name": "PBA_GovCup",
        "season":          "PBA Governors' Cup 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "Taiwan_PLeague": {
        "path":            "/basketball/taiwan/p-league/",
        "tournament_name": "Taiwan_PLeague",
        "season":          "P.League+ 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "Taiwan_TPBL": {
        "path":            "/basketball/taiwan/tpbl/",
        "tournament_name": "Taiwan_TPBL",
        "season":          "TPBL 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    # Adriatic League — pivot target after V6 (NBA market efficiency ceiling).
    # Sponsor naming on Flashscore: "AdmiralBet ABA League".
    # All three seasons share tournament_name="ABA" → data merges in one bucket.
    "ABA": {
        "path":            "/basketball/europe/admiralbet-aba-league/",
        "tournament_name": "ABA",
        "season":          "ABA League 25/26",
        "season_type":     SeasonType.REGULAR,
    },
    "ABA_2425": {
        "path":            "/basketball/europe/admiralbet-aba-league-2024-2025/",
        "tournament_name": "ABA",
        "season":          "ABA League 24/25",
        "season_type":     SeasonType.REGULAR,
    },
    "ABA_2324": {
        "path":            "/basketball/europe/admiralbet-aba-league-2023-2024/",
        "tournament_name": "ABA",
        "season":          "ABA League 23/24",
        "season_type":     SeasonType.REGULAR,
    },
}

_JS_COLLECT_MATCHES = r"""
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
        const fsHomeEl = el.querySelector('.event__score--home');
        const fsAwayEl = el.querySelector('.event__score--away');
        const finalHome = fsHomeEl ? (parseInt(fsHomeEl.textContent.trim()) || null) : null;
        const finalAway = fsAwayEl ? (parseInt(fsAwayEl.textContent.trim()) || null) : null;
        const qScores = [];
        for (let q = 1; q <= 4; q++) {
            const hEl = el.querySelector('.event__part--home.event__part--' + q);
            const aEl = el.querySelector('.event__part--away.event__part--' + q);
            qScores.push([hEl ? (parseInt(hEl.textContent.trim()) || null) : null,
                          aEl ? (parseInt(aEl.textContent.trim()) || null) : null]);
        }
        const hasOT = !!el.querySelector('.event__part--home.event__part--5');
        if (home && away && fsId) rows.push({ timeStr, home, away, fsId, finalHome, finalAway, qScores, hasOT });
    });
    return rows;
}
"""


@dataclass
class FsCollectedMatch:
    fs_id:      str
    time_str:   str
    match_dt:   datetime | None
    home_raw:   str
    away_raw:   str
    final_home: int | None
    final_away: int | None
    has_ot:     bool
    q_scores:   list[QScore] = field(default_factory=list)


def _abbrev(name: str) -> str:
    words = name.split()
    if len(words) >= 2:
        return "".join(w[0] for w in words[:3]).upper()
    return name[:3].upper()


def _parse_fs_datetime(time_str: str) -> datetime | None:
    s = time_str.strip()
    m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})(?:\s+(\d{2}):(\d{2}))?", s)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hour   = int(m.group(4)) if m.group(4) else 12
        minute = int(m.group(5)) if m.group(5) else 0
        try:
            return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
        except ValueError:
            return None
    m2 = re.match(r"(\d{1,2})\.(\d{1,2})\.\s*(\d{2}):(\d{2})", s)
    if m2:
        day, month = int(m2.group(1)), int(m2.group(2))
        hour, minute = int(m2.group(3)), int(m2.group(4))
    else:
        m3 = re.match(r"(\d{1,2})\.(\d{1,2})\.", s)
        if not m3:
            return None
        day, month = int(m3.group(1)), int(m3.group(2))
        hour, minute = 12, 0
    today = date.today()
    best = None
    for year in [today.year, today.year - 1]:
        try:
            d = date(year, month, day)
            if d <= today and abs((today - d).days) <= 540:
                if best is None or d > best:
                    best = d
        except ValueError:
            pass
    if best is None:
        return None
    return datetime(best.year, best.month, best.day, hour, minute, tzinfo=timezone.utc)


async def scrape_results_page(page: Page, path: str) -> list[FsCollectedMatch]:
    url = FLASHSCORE_BASE_URL + path + "results/"
    log.info("  Scraping: %s", url)
    await page.goto(url, wait_until="domcontentloaded", timeout=25_000)
    await asyncio.sleep(2)
    # Re-dismiss gambling-ads / lang-selector dialogs that reappear after
    # per-page navigation and intercept the "Show more matches" click.
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
    raw_rows: list[dict[str, Any]] = await page.evaluate(_JS_COLLECT_MATCHES)
    log.info("  Raw rows: %d", len(raw_rows))

    results: list[FsCollectedMatch] = []
    for r in raw_rows:
        if not r.get("fsId") or (r.get("finalHome") is None and r.get("finalAway") is None):
            continue
        results.append(FsCollectedMatch(
            fs_id=r["fsId"], time_str=r["timeStr"],
            match_dt=_parse_fs_datetime(r["timeStr"]),
            home_raw=r["home"], away_raw=r["away"],
            final_home=r.get("finalHome"), final_away=r.get("finalAway"),
            has_ot=r.get("hasOT", False),
            q_scores=[(q[0], q[1]) for q in r["qScores"]],
        ))
    log.info("  Collected: %d finished matches", len(results))
    return results


async def save_match(
    session_factory: Any,
    league_key: str,
    config: dict,
    m: FsCollectedMatch,
) -> bool:
    if m.match_dt is None:
        return False
    external_id = f"fs_{m.fs_id}"
    home_ext = f"fs_{config['tournament_name']}_{m.home_raw[:30]}"
    away_ext = f"fs_{config['tournament_name']}_{m.away_raw[:30]}"

    async with session_factory() as db:
        async with db.begin():
            home_team = await get_or_create_team(db, home_ext, m.home_raw, _abbrev(m.home_raw))
            away_team = await get_or_create_team(db, away_ext, m.away_raw, _abbrev(m.away_raw))
            match = await upsert_match(
                db, external_id=external_id,
                home_team_id=home_team.id, away_team_id=away_team.id,
                scheduled_at=m.match_dt, season=config["season"],
                tournament_name=config["tournament_name"], season_type=config["season_type"],
                status=MatchStatus.FINISHED,
                home_score_final=m.final_home, away_score_final=m.final_away,
                went_to_overtime=m.has_ot, has_quarter_breakdown=True, flashscore_id=m.fs_id,
            )
            quarter_rows = [
                QuarterStatRow(
                    period_number=period, period_type=PeriodType.QUARTER,
                    home_score=h_score, away_score=a_score,
                    home_fga=None, away_fga=None, home_fta=None, away_fta=None,
                    home_off_reb=None, away_off_reb=None, home_turnovers=None, away_turnovers=None,
                    home_possessions=None, away_possessions=None, home_pace=None, away_pace=None,
                )
                for period, (h_score, a_score) in enumerate(m.q_scores[:4], 1)
            ]
            if quarter_rows:
                await save_quarter_stats(db, match.id, quarter_rows)
    return True
