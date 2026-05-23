"""
Flashscore collector — collects matches from scratch for leagues not on Sofascore.

For each configured league:
  1. Scrape results page (click 'show more' until all loaded)
  2. Extract: teams, date/time, final score, Q1-Q4 scores, flashscore_id
  3. Upsert teams → upsert match → insert 4 quarter_stats rows
  4. Sets has_quarter_breakdown=True and flashscore_id from the start

Usage:
    python -m src.data_collection.flashscore_collector
    python -m src.data_collection.flashscore_collector --leagues LegaA PBA_PhilCup
    python -m src.data_collection.flashscore_collector --dry-run
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from playwright.async_api import Page, async_playwright

from src.config import settings
from src.database.crud import QuarterStatRow, get_or_create_team, save_quarter_stats, upsert_match
from src.database.engine import dispose_engine, get_session_factory
from src.database.models import MatchStatus, PeriodType, SeasonType

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("flashscore_collector")

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL = "https://www.flashscore.com"

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

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
}

SHOW_MORE_DELAY = 1.2

# ── Data structure ────────────────────────────────────────────────────────────

QScore = tuple[int | None, int | None]


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

# ── JavaScript ────────────────────────────────────────────────────────────────

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
                if (child.nodeType === Node.TEXT_NODE) {
                    const t = child.textContent.trim();
                    if (t) return t;
                }
                if (child.nodeType === Node.ELEMENT_NODE) {
                    const t = child.textContent.trim().split('\n')[0].trim();
                    if (t) return t;
                }
            }
            return node.textContent.trim().split('\n')[0].trim();
        };

        const home = firstName(homeEl);
        const away = firstName(awayEl);
        const timeStr = timeEl ? timeEl.textContent.trim() : '';

        // Match ID from element id, e.g. "g_3_A78Lrpel" → "A78Lrpel"
        let elId = el.id || el.getAttribute('data-id') || '';
        const idM = elId.match(/([A-Za-z0-9]{8,})$/);
        const fsId = idM ? idM[1] : '';

        // Final scores
        const fsHomeEl = el.querySelector('.event__score--home');
        const fsAwayEl = el.querySelector('.event__score--away');
        const finalHome = fsHomeEl ? (parseInt(fsHomeEl.textContent.trim()) || null) : null;
        const finalAway = fsAwayEl ? (parseInt(fsAwayEl.textContent.trim()) || null) : null;

        // Per-quarter scores Q1-Q4
        const qScores = [];
        for (let q = 1; q <= 4; q++) {
            const hEl = el.querySelector('.event__part--home.event__part--' + q);
            const aEl = el.querySelector('.event__part--away.event__part--' + q);
            const h = hEl ? (parseInt(hEl.textContent.trim()) || null) : null;
            const a = aEl ? (parseInt(aEl.textContent.trim()) || null) : null;
            qScores.push([h, a]);
        }

        // OT: check if period 5 exists
        const ot1Home = el.querySelector('.event__part--home.event__part--5');
        const hasOT = !!ot1Home;

        if (home && away && fsId) {
            rows.push({ timeStr, home, away, fsId, finalHome, finalAway, qScores, hasOT });
        }
    });

    return rows;
}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[^a-z0-9 ]")


def _abbrev(name: str) -> str:
    """Derive 3-letter abbreviation from team name."""
    words = name.split()
    if len(words) >= 2:
        return "".join(w[0] for w in words[:3]).upper()
    return name[:3].upper()


def _parse_fs_datetime(time_str: str) -> datetime | None:
    """
    Parse Flashscore date strings → datetime UTC.
    Handles: '22.05. 21:00', '08.11.2024', '08.11.2024 14:30'.
    When no year is given, picks the most recent past date ≤ today within 18 months.
    """
    s = time_str.strip()

    # Case 1: DD.MM.YYYY HH:MM or DD.MM.YYYY
    m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})(?:\s+(\d{2}):(\d{2}))?", s)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hour   = int(m.group(4)) if m.group(4) else 12
        minute = int(m.group(5)) if m.group(5) else 0
        try:
            return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
        except ValueError:
            return None

    # Case 2: DD.MM. HH:MM (no year)
    m2 = re.match(r"(\d{1,2})\.(\d{1,2})\.\s*(\d{2}):(\d{2})", s)
    if m2:
        day, month = int(m2.group(1)), int(m2.group(2))
        hour, minute = int(m2.group(3)), int(m2.group(4))
    else:
        # Case 3: DD.MM. (no time, no year)
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

# ── Results page scraping ─────────────────────────────────────────────────────


async def _scrape_results_page(page: Page, path: str) -> list[FsCollectedMatch]:
    url = BASE_URL + path + "results/"
    log.info("  Scraping: %s", url)

    await page.goto(url, wait_until="domcontentloaded", timeout=25_000)
    await asyncio.sleep(2)

    # Click 'show more matches' until it disappears
    clicks = 0
    while True:
        btn = page.locator(
            'button:has-text("show more matches"), a:has-text("show more matches")'
        )
        try:
            if not await btn.first.is_visible(timeout=2_000):
                break
            await btn.first.click()
            await asyncio.sleep(SHOW_MORE_DELAY)
            clicks += 1
        except Exception:
            break

    log.info("  'Show more' clicked %d times", clicks)

    raw_rows: list[dict] = await page.evaluate(_JS_COLLECT_MATCHES)
    log.info("  Raw rows: %d", len(raw_rows))

    results: list[FsCollectedMatch] = []
    for r in raw_rows:
        if not r.get("fsId"):
            continue
        # Skip matches without final score (live/scheduled)
        if r.get("finalHome") is None and r.get("finalAway") is None:
            continue
        match_dt = _parse_fs_datetime(r["timeStr"])
        results.append(FsCollectedMatch(
            fs_id=r["fsId"],
            time_str=r["timeStr"],
            match_dt=match_dt,
            home_raw=r["home"],
            away_raw=r["away"],
            final_home=r.get("finalHome"),
            final_away=r.get("finalAway"),
            has_ot=r.get("hasOT", False),
            q_scores=[(q[0], q[1]) for q in r["qScores"]],
        ))

    log.info("  Collected: %d finished matches", len(results))
    return results

# ── DB persistence ────────────────────────────────────────────────────────────


async def _save_match(
    session_factory: Any,
    league_key: str,
    config: dict,
    m: FsCollectedMatch,
) -> bool:
    """Upsert one match with teams and quarter stats. Returns True on success."""
    if m.match_dt is None:
        return False

    external_id = f"fs_{m.fs_id}"
    home_ext = f"fs_{config['tournament_name']}_{m.home_raw[:30]}"
    away_ext = f"fs_{config['tournament_name']}_{m.away_raw[:30]}"

    async with session_factory() as db:
        async with db.begin():
            home_team = await get_or_create_team(
                db, home_ext, m.home_raw, _abbrev(m.home_raw)
            )
            away_team = await get_or_create_team(
                db, away_ext, m.away_raw, _abbrev(m.away_raw)
            )

            match = await upsert_match(
                db,
                external_id=external_id,
                home_team_id=home_team.id,
                away_team_id=away_team.id,
                scheduled_at=m.match_dt,
                season=config["season"],
                tournament_name=config["tournament_name"],
                season_type=config["season_type"],
                status=MatchStatus.FINISHED,
                home_score_final=m.final_home,
                away_score_final=m.final_away,
                went_to_overtime=m.has_ot,
                has_quarter_breakdown=True,
                flashscore_id=m.fs_id,
            )

            # Build 4 quarter rows
            quarter_rows = []
            for period, (h_score, a_score) in enumerate(m.q_scores[:4], 1):
                quarter_rows.append(QuarterStatRow(
                    period_number=period,
                    period_type=PeriodType.QUARTER,
                    home_score=h_score,
                    away_score=a_score,
                    home_fga=None, away_fga=None,
                    home_fta=None, away_fta=None,
                    home_off_reb=None, away_off_reb=None,
                    home_turnovers=None, away_turnovers=None,
                    home_possessions=None, away_possessions=None,
                    home_pace=None, away_pace=None,
                ))

            if quarter_rows:
                await save_quarter_stats(db, match.id, quarter_rows)

    return True

# ── Main orchestrator ─────────────────────────────────────────────────────────


class FlashscoreCollector:
    def __init__(
        self,
        session_factory: Any = None,
        leagues: list[str] | None = None,
        dry_run: bool = False,
    ) -> None:
        self._sf      = session_factory or get_session_factory()
        self._leagues = leagues
        self._dry_run = dry_run

    async def run(self) -> None:
        log.info("DB: %s:%s/%s", settings.db.host, settings.db.port, settings.db.name)

        targets = {
            k: v for k, v in COLLECTOR_LEAGUES.items()
            if self._leagues is None or k in self._leagues
        }
        log.info("Leagues to collect: %s", list(targets.keys()))

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            ua  = random.choice(USER_AGENTS)
            ctx = await browser.new_context(user_agent=ua)
            page = await ctx.new_page()

            log.info("Establishing Flashscore session (UA: %s…)", ua[:45])
            await page.goto(BASE_URL + "/basketball/", wait_until="domcontentloaded", timeout=20_000)
            await asyncio.sleep(3)

            grand_ok = grand_skip = grand_err = 0
            t_total  = time.monotonic()

            for league_key, config in targets.items():
                log.info("━" * 60)
                log.info("[%s]  Collecting from Flashscore…", league_key)

                try:
                    matches = await _scrape_results_page(page, config["path"])
                except Exception as exc:
                    log.error("[%s] Failed to scrape: %s", league_key, exc)
                    continue

                if self._dry_run:
                    for m in matches[:5]:
                        log.info(
                            "  [dry] %s | %s vs %s | final %s-%s | Q1-4: %s",
                            m.time_str, m.home_raw, m.away_raw,
                            m.final_home, m.final_away,
                            [(qs[0], qs[1]) for qs in m.q_scores],
                        )
                    log.info("  [dry] Total: %d matches found for %s", len(matches), league_key)
                    continue

                ok = skip = err = 0
                t0 = time.monotonic()

                for i, m in enumerate(matches, 1):
                    try:
                        success = await _save_match(self._sf, league_key, config, m)
                        if success:
                            ok += 1
                        else:
                            skip += 1
                    except Exception as exc:
                        log.warning("  err [%d]: %s vs %s → %s", i, m.home_raw, m.away_raw, str(exc)[:100])
                        err += 1

                    if i % 50 == 0 or i == len(matches):
                        elapsed = time.monotonic() - t0
                        log.info(
                            "[%s]  %d/%d | ok=%d skip=%d err=%d | %.0fs",
                            league_key, i, len(matches), ok, skip, err, elapsed,
                        )

                log.info("[%s]  DONE  ok=%d  skip=%d  err=%d", league_key, ok, skip, err)
                grand_ok  += ok
                grand_skip += skip
                grand_err  += err

            await browser.close()

        elapsed = time.monotonic() - t_total
        log.info("━" * 60)
        log.info(
            "ALL DONE  ok=%d  skip=%d  err=%d  |  %.0f min",
            grand_ok, grand_skip, grand_err, elapsed / 60,
        )
        await dispose_engine()

# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> dict:
    args = sys.argv[1:]
    opts: dict = {"leagues": None, "dry_run": False}
    i = 0
    while i < len(args):
        if args[i] == "--leagues" and i + 1 < len(args):
            vals, i = [], i + 1
            while i < len(args) and not args[i].startswith("--"):
                vals.append(args[i]); i += 1
            opts["leagues"] = vals
        elif args[i] == "--dry-run":
            opts["dry_run"] = True; i += 1
        else:
            i += 1
    return opts


if __name__ == "__main__":
    opts = _parse_args()
    collector = FlashscoreCollector(
        leagues=opts["leagues"],
        dry_run=opts["dry_run"],
    )
    asyncio.run(collector.run())
