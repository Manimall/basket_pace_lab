"""
Flashscore enricher — replaces game-fallback quarter_stats with real per-quarter data.

Strategy:
  - Build an in-memory index from the Flashscore results page for each league.
    Each entry already contains Q1-Q4 scores (no per-match navigation needed).
  - Fuzzy-match DB matches to Flashscore entries by date (±1 day) + team names.
  - Replace the single GAME row with 4 QUARTER rows; set has_quarter_breakdown=True.

Note: Flashscore only exposes per-quarter SCORES on the results list.
      FGA/FTA/OffReb/TO remain NULL (not available at quarter level on Flashscore).

Usage:
    python -m src.data_collection.flashscore_enricher
    python -m src.data_collection.flashscore_enricher --leagues EuroLeague VTB BBL
    python -m src.data_collection.flashscore_enricher --leagues LegaA ACB --seasons 2526 2425
    python -m src.data_collection.flashscore_enricher --leagues EuroLeague --limit 20
    python -m src.data_collection.flashscore_enricher --dry-run
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
from datetime import date, datetime
from typing import Any

from playwright.async_api import Page, async_playwright
from sqlalchemy import text

from src.config import settings
from src.database.crud import QuarterStatRow, save_quarter_stats
from src.database.engine import dispose_engine, get_session_factory
from src.database.models import PeriodType

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("flashscore_enricher")

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL = "https://www.flashscore.com"

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

# tournament_name (DB) → Flashscore URL path
LEAGUE_PATHS: dict[str, str] = {
    "EuroLeague":  "/basketball/europe/euroleague/",
    "VTB":         "/basketball/russia/vtb-united-league/",
    "ACB":         "/basketball/spain/acb/",
    "LegaA":       "/basketball/italy/lega-a/",
    "BBL":         "/basketball/germany/bbl/",
    "LNB":         "/basketball/france/lnb/",
    "NBL":         "/basketball/australia/nbl/",
    "CBA":         "/basketball/china/cba/",
    "BLeague":     "/basketball/japan/b-league/",
    "LNBP":        "/basketball/mexico/lnbp/",
    # PBA Philippines — 3 separate conferences on Flashscore
    "PBA":         "/basketball/philippines/pba-philippine-cup/",
    "PBA_Comm":    "/basketball/philippines/pba-commissioner-s-cup/",
    "PBA_Gov":     "/basketball/philippines/pba-governors-cup/",
    # Taiwan P. League+
    "Taiwan_PLeague": "/basketball/taiwan/p-league/",
    "Taiwan_TPBL":    "/basketball/taiwan/tpbl/",
}

# Season code → SQL LIKE pattern against matches.season column
SEASON_FILTERS: dict[str, str] = {
    "2526": "%25/26%",
    "2425": "%24/25%",
    "2324": "%23/24%",
}

MATCH_THRESHOLD = 0.40   # min similarity to accept a team name match
SHOW_MORE_DELAY = 1.2    # seconds to wait after each "show more" click

# ── Data structures ───────────────────────────────────────────────────────────

QScore = tuple[int | None, int | None]   # (home, away) for one quarter


@dataclass
class FsMatch:
    """One match entry from the Flashscore results index."""
    fs_id:      str
    date_str:   str             # raw time string from Flashscore (for debugging)
    match_date: date | None
    home_raw:   str
    away_raw:   str
    home_norm:  str
    away_norm:  str
    q_scores:   list[QScore] = field(default_factory=list)  # [(h_q1,a_q1), …, (h_q4,a_q4)]


@dataclass
class DbMatch:
    match_id:   int
    ext_id:     str
    league:     str
    match_date: date
    home_team:  str
    away_team:  str

# ── Team name normalisation ───────────────────────────────────────────────────

_STOP = re.compile(
    r"\b(basketball|club|bc|bk|fc|sk|ak|kk|as|ss|bball|city|team|"
    r"real|istanbul|moscow|milan|milano|munchen|munich|london|"
    r"koszykowki|baloncesto|pallacanestro|basket|baskets|sport|brose|s|"
    # Common sponsor/generic role words (NOT identifying city/team names)
    r"telekom|ewe|fraport|skyliners|mhp|mlp|ratiopharm|"
    r"fitness|first|gladiators|towers|seawolves|riesen|lowen|lions|"
    r"academics|bv|rasta|"
    # EuroLeague sponsor words (removed so team's traditional name survives)
    r"ea7|emporio|armani|ldlc|beko|meridianbet|mozzart|bet)\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^a-z0-9 ]")
_DIGITS_ONLY = re.compile(r"\b\d+\b")

_SPLIT_COMPOUNDS = [
    # Flashscore splits these; Sofascore joins them
    ("sunrockers", "sun rockers"),
    ("neozone", "neo zone"),
]

# Explicit pre-normalization aliases (applied before stop-word removal).
# Key: lowercase, diacritics stripped. Value: the canonical replacement.
_NAME_ALIASES: dict[str, str] = {
    # EuroLeague
    "ea7 emporio armani milano": "olimpia",
    "barca basket": "barcelona",
    "barca": "barcelona",
    "crvena zvezda": "crvena zvezda meridianbet",  # ensure token overlap
    "partizan": "partizan mozzart bet",
    # CBA — Sofascore uses English nicknames; Flashscore uses sponsor/city names
    "nanjing monkey kings": "nanjing tongxi",
    "zhejiang golden bulls": "zhejiang guangsha",
}


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    # Apply explicit aliases first
    if name in _NAME_ALIASES:
        name = _NAME_ALIASES[name]
    for compound, expanded in _SPLIT_COMPOUNDS:
        name = name.replace(compound, expanded)
    name = _STOP.sub(" ", name)
    name = _DIGITS_ONLY.sub(" ", name)
    name = _PUNCT.sub(" ", name)
    return " ".join(name.split())


def _sim(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / max(len(ta), len(tb))
    # Boost when the shorter name's tokens are all contained in the longer name.
    # Handles the common case where Flashscore uses a short city name ("bonn")
    # and Sofascore uses the full name with sponsors ("Telekom Baskets Bonn").
    big, small = (ta, tb) if len(ta) >= len(tb) else (tb, ta)
    if small and small.issubset(big):
        containment = 0.5 + 0.25 * (len(small) / len(big))
        overlap = max(overlap, containment)
    return overlap

# ── JavaScript helpers ────────────────────────────────────────────────────────

# Extracts all match rows from the loaded results page, including per-quarter scores.
# Uses querySelector with class fragments that are stable across builds.
_JS_EXTRACT_MATCHES = r"""
() => {
    const rows = [];

    document.querySelectorAll('.event__match').forEach(el => {
        // Team names — first text node only (strips "Advancing to next round:..." etc.)
        const homeEl = el.querySelector('.event__homeParticipant');
        const awayEl = el.querySelector('.event__awayParticipant');
        const timeEl = el.querySelector('.event__time');

        const firstName = (node) => {
            if (!node) return '';
            // Walk child nodes and take the first non-empty text
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

        // Per-quarter scores: classes like "event__part--home event__part--1"
        const qScores = [];
        for (let q = 1; q <= 4; q++) {
            const hEl = el.querySelector('.event__part--home.event__part--' + q);
            const aEl = el.querySelector('.event__part--away.event__part--' + q);
            const h = hEl ? (parseInt(hEl.textContent.trim()) || null) : null;
            const a = aEl ? (parseInt(aEl.textContent.trim()) || null) : null;
            qScores.push([h, a]);
        }

        if (home && away && fsId) {
            rows.push({ timeStr, home, away, fsId, qScores });
        }
    });

    return rows;
}
"""


def _parse_fs_date(time_str: str) -> date | None:
    """
    Parse Flashscore time strings like '22.05. 21:00' → date.
    No year is given. Returns the most recent PAST date (≤ today) within 18 months.
    This correctly handles Oct-Nov start-of-season dates (e.g., '01.10.' → 2025, not 2026).
    """
    m = re.match(r"(\d{1,2})\.(\d{1,2})\.", time_str.strip())
    if not m:
        return None
    day, month = int(m.group(1)), int(m.group(2))
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

# ── Results page scraping ─────────────────────────────────────────────────────


async def _build_league_index(page: Page, league_key: str) -> list[FsMatch]:
    """
    Navigate to the league results page, click 'show more matches' until all
    results are loaded, then extract all match entries (with per-quarter scores).
    """
    path = LEAGUE_PATHS[league_key]
    url  = BASE_URL + path + "results/"
    log.info("  Building index: %s", url)

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

    raw_rows: list[dict] = await page.evaluate(_JS_EXTRACT_MATCHES)
    log.info("  Raw rows from JS: %d", len(raw_rows))

    entries: list[FsMatch] = []
    for r in raw_rows:
        if not r.get("fsId"):
            continue
        match_date = _parse_fs_date(r["timeStr"])
        # Filter: skip if Q1-Q4 all None (match in progress or postponed)
        q_scores: list[QScore] = [(q[0], q[1]) for q in r["qScores"]]
        entries.append(FsMatch(
            fs_id=r["fsId"],
            date_str=r["timeStr"],
            match_date=match_date,
            home_raw=r["home"],
            away_raw=r["away"],
            home_norm=_norm(r["home"]),
            away_norm=_norm(r["away"]),
            q_scores=q_scores,
        ))

    has_scores = sum(1 for e in entries if any(q[0] is not None for q in e.q_scores))
    log.info("  Index built: %d entries (%d have quarter scores)", len(entries), has_scores)
    return entries


def _find_match(
    index: list[FsMatch],
    target_date: date,
    home_name: str,
    away_name: str,
) -> FsMatch | None:
    h_norm = _norm(home_name)
    a_norm = _norm(away_name)

    best: FsMatch | None = None
    best_score = 0.0

    for entry in index:
        if entry.match_date:
            delta = abs((entry.match_date - target_date).days)
            if delta > 1:
                continue

        h_score = _sim(h_norm, entry.home_norm)
        a_score = _sim(a_norm, entry.away_norm)
        combined = (h_score + a_score) / 2

        if combined > best_score:
            best_score = combined
            best = entry

    return best if best_score >= MATCH_THRESHOLD else None


def _build_quarter_rows(fs_match: FsMatch) -> list[QuarterStatRow] | None:
    """Build 4 QuarterStatRow objects from per-quarter scores on FsMatch."""
    if len(fs_match.q_scores) < 4:
        return None
    if all(q[0] is None for q in fs_match.q_scores):
        return None

    rows = []
    for period, (h_score, a_score) in enumerate(fs_match.q_scores[:4], 1):
        rows.append(QuarterStatRow(
            period_number=period,
            period_type=PeriodType.QUARTER,
            home_score=h_score,
            away_score=a_score,
            # Box-score stats (FGA/FTA/OffReb/TO) not available per-quarter on FS
            home_fga=None, away_fga=None,
            home_fta=None, away_fta=None,
            home_off_reb=None, away_off_reb=None,
            home_turnovers=None, away_turnovers=None,
            home_possessions=None, away_possessions=None,
            home_pace=None, away_pace=None,
        ))
    return rows

# ── DB helpers ────────────────────────────────────────────────────────────────


async def _save_enriched(
    db_session_factory: Any,
    match_id: int,
    fs_id: str,
    quarter_rows: list[QuarterStatRow],
) -> None:
    async with db_session_factory() as db:
        async with db.begin():
            await save_quarter_stats(db, match_id, quarter_rows)
            await db.execute(
                text(
                    "UPDATE matches "
                    "SET has_quarter_breakdown = TRUE, flashscore_id = :fid "
                    "WHERE id = :mid"
                ),
                {"fid": fs_id, "mid": match_id},
            )


async def _load_db_matches(
    db_session_factory: Any,
    leagues: list[str] | None,
    season_codes: list[str] | None,
    limit: int | None,
) -> list[DbMatch]:
    conditions = ["m.has_quarter_breakdown = FALSE", "m.home_score_final IS NOT NULL"]
    params: dict = {}

    if leagues:
        conditions.append("m.tournament_name = ANY(:leagues)")
        params["leagues"] = leagues

    if season_codes:
        season_likes = [SEASON_FILTERS[c] for c in season_codes if c in SEASON_FILTERS]
        if season_likes:
            season_clauses = " OR ".join(
                f"m.season LIKE :season_{i}" for i in range(len(season_likes))
            )
            conditions.append(f"({season_clauses})")
            for i, pat in enumerate(season_likes):
                params[f"season_{i}"] = pat

    where = " AND ".join(conditions)
    lim   = f"LIMIT {int(limit)}" if limit else ""

    sql = f"""
        SELECT m.id, m.external_id, m.tournament_name,
               m.scheduled_at::date AS match_date,
               ht.name AS home_team, at.name AS away_team
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE {where}
        ORDER BY m.scheduled_at DESC
        {lim}
    """

    async with db_session_factory() as db:
        rows = (await db.execute(text(sql), params)).mappings().all()

    return [
        DbMatch(
            match_id=r["id"],
            ext_id=r["external_id"] or "",
            league=r["tournament_name"] or "",
            match_date=r["match_date"],
            home_team=r["home_team"],
            away_team=r["away_team"],
        )
        for r in rows
    ]


async def _ensure_flashscore_id_column(db_session_factory: Any) -> None:
    async with db_session_factory() as db:
        async with db.begin():
            await db.execute(text(
                "ALTER TABLE matches ADD COLUMN IF NOT EXISTS flashscore_id VARCHAR(16);"
            ))

# ── Main orchestrator ─────────────────────────────────────────────────────────


class FlashscoreEnricher:
    def __init__(
        self,
        session_factory: Any = None,
        leagues: list[str] | None = None,
        season_codes: list[str] | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> None:
        self._sf           = session_factory or get_session_factory()
        self._leagues      = leagues
        self._season_codes = season_codes
        self._limit        = limit
        self._dry_run      = dry_run

    async def run(self) -> None:
        await _ensure_flashscore_id_column(self._sf)
        log.info("DB: %s:%s/%s", settings.db.host, settings.db.port, settings.db.name)

        db_matches = await _load_db_matches(
            self._sf, self._leagues, self._season_codes, self._limit
        )
        log.info("Matches to enrich: %d", len(db_matches))

        if self._dry_run:
            for m in db_matches[:20]:
                log.info("  [dry] %s | %s vs %s | %s", m.league, m.home_team, m.away_team, m.match_date)
            return

        if not db_matches:
            log.info("Nothing to enrich.")
            return

        # Group by league
        by_league: dict[str, list[DbMatch]] = {}
        for m in db_matches:
            by_league.setdefault(m.league, []).append(m)

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            ua  = random.choice(USER_AGENTS)
            ctx = await browser.new_context(user_agent=ua)
            page = await ctx.new_page()

            log.info("Establishing Flashscore session (UA: %s…)", ua[:45])
            await page.goto(BASE_URL + "/basketball/", wait_until="domcontentloaded", timeout=20_000)
            await asyncio.sleep(3)

            grand_ok = grand_nomatch = grand_noscores = grand_err = 0
            t_total  = time.monotonic()

            for league_key, matches in by_league.items():
                if league_key not in LEAGUE_PATHS:
                    log.warning("[%s] No Flashscore URL mapping — skipping", league_key)
                    grand_nomatch += len(matches)
                    continue

                log.info("━" * 60)
                log.info("[%s]  Building results index…", league_key)

                try:
                    index = await _build_league_index(page, league_key)
                except Exception as exc:
                    log.error("[%s] Failed to build index: %s", league_key, exc)
                    grand_err += len(matches)
                    continue

                ok = nomatch = noscores = err = 0
                t0 = time.monotonic()

                for i, m in enumerate(matches, 1):
                    try:
                        fs_entry = _find_match(index, m.match_date, m.home_team, m.away_team)
                        if fs_entry is None:
                            nomatch += 1
                            log.debug("  no-match: %s vs %s (%s)", m.home_team, m.away_team, m.match_date)
                            continue

                        quarter_rows = _build_quarter_rows(fs_entry)
                        if quarter_rows is None:
                            noscores += 1
                            log.debug("  no-scores: %s → %s", m.match_date, fs_entry.fs_id)
                        else:
                            await _save_enriched(self._sf, m.match_id, fs_entry.fs_id, quarter_rows)
                            ok += 1

                    except Exception as exc:
                        log.warning("  err: %s vs %s → %s", m.home_team, m.away_team, str(exc)[:100])
                        err += 1

                    if i % 50 == 0 or i == len(matches):
                        elapsed = time.monotonic() - t0
                        log.info(
                            "[%s]  %d/%d | ok=%d nomatch=%d noscores=%d err=%d | %.0fs elapsed",
                            league_key, i, len(matches), ok, nomatch, noscores, err, elapsed,
                        )

                grand_ok      += ok
                grand_nomatch += nomatch
                grand_noscores += noscores
                grand_err     += err

                log.info(
                    "[%s]  DONE  ok=%d  nomatch=%d  noscores=%d  err=%d",
                    league_key, ok, nomatch, noscores, err,
                )

            await browser.close()

        elapsed = time.monotonic() - t_total
        log.info("━" * 60)
        log.info(
            "ALL DONE  ok=%d  nomatch=%d  noscores=%d  err=%d  |  %.0f min",
            grand_ok, grand_nomatch, grand_noscores, grand_err, elapsed / 60,
        )

        async with self._sf() as db:
            enriched = (await db.execute(
                text("SELECT COUNT(*) FROM matches WHERE flashscore_id IS NOT NULL")
            )).scalar()
        log.info("Matches with quarter data (Flashscore): %d", enriched)
        await dispose_engine()

# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> dict:
    args = sys.argv[1:]
    opts: dict = {"leagues": None, "seasons": None, "limit": None, "dry_run": False}
    i = 0
    while i < len(args):
        if args[i] == "--leagues" and i + 1 < len(args):
            vals, i = [], i + 1
            while i < len(args) and not args[i].startswith("--"):
                vals.append(args[i]); i += 1
            opts["leagues"] = vals
        elif args[i] == "--seasons" and i + 1 < len(args):
            vals, i = [], i + 1
            while i < len(args) and not args[i].startswith("--"):
                vals.append(args[i]); i += 1
            opts["seasons"] = vals
        elif args[i] == "--limit" and i + 1 < len(args):
            opts["limit"] = int(args[i + 1]); i += 2
        elif args[i] == "--dry-run":
            opts["dry_run"] = True; i += 1
        else:
            i += 1
    return opts


if __name__ == "__main__":
    opts = _parse_args()
    enricher = FlashscoreEnricher(
        leagues=opts["leagues"],
        season_codes=opts["seasons"],
        limit=opts["limit"],
        dry_run=opts["dry_run"],
    )
    asyncio.run(enricher.run())
