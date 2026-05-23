"""
Integration test: fetch → parse → save → verify.

Requires running PostgreSQL (docker compose up postgres -d).
Reads DB settings from .env (see .env.example).

Run:
    python run_parser_test.py            # live API
    python run_parser_test.py --mock     # local fixtures (no internet)
    python run_parser_test.py 12345678   # different event_id, live
"""

import asyncio
import logging
import sys

from sqlalchemy import select, text

from src.config import settings
from src.database.engine import create_tables, dispose_engine, get_session_factory
from src.database.models import Match, QuarterStats, Team
from src.data_collection.sofascore_client import SofascoreClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("run_parser_test")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = [a for a in sys.argv[1:] if a.startswith("--")]
MOCK_MODE = "--mock" in flags
TEST_EVENT_ID = int(args[0]) if args else 12571063

# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"

def _row(label: str, value: str) -> str:
    return f"  {CYAN}{label:<28}{RESET} {value}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    log.info("DB: %s:%s/%s", settings.db.host, settings.db.port, settings.db.name)

    # ── Init tables ──────────────────────────────────────────────────────────
    log.info("Initialising tables …")
    await create_tables()
    log.info("Tables OK")

    # ── Fetch + parse + save ─────────────────────────────────────────────────
    session_factory = get_session_factory()

    mode_label = "MOCK (local fixtures)" if MOCK_MODE else "LIVE (Sofascore API)"
    log.info("Mode: %s", mode_label)

    async with SofascoreClient(mock=MOCK_MODE) as client:
        async with session_factory() as session:
            async with session.begin():
                match = await client.process_and_save_match(TEST_EVENT_ID, session)

    # ── Verification SELECT ──────────────────────────────────────────────────
    log.info("Running verification SELECT …")

    async with session_factory() as session:
        # Reload match with teams
        result = await session.execute(
            select(Match).where(Match.id == match.id)
        )
        saved_match: Match = result.scalar_one()

        home_team_result = await session.execute(
            select(Team).where(Team.id == saved_match.home_team_id)
        )
        away_team_result = await session.execute(
            select(Team).where(Team.id == saved_match.away_team_id)
        )
        home_team = home_team_result.scalar_one()
        away_team = away_team_result.scalar_one()

        qs_result = await session.execute(
            select(QuarterStats)
            .where(QuarterStats.match_id == match.id)
            .order_by(QuarterStats.period_type, QuarterStats.period_number)
        )
        quarter_rows = qs_result.scalars().all()

        # DB version sanity check
        ver_result = await session.execute(text("SELECT version()"))
        pg_version: str = ver_result.scalar_one().split(",")[0]

    # ── Output ───────────────────────────────────────────────────────────────
    sep = f"{CYAN}{'─' * 64}{RESET}"
    print(f"\n{sep}")
    print(f"  {BOLD}VERIFICATION REPORT{RESET}  event_id={TEST_EVENT_ID}")
    print(sep)

    print(f"\n{BOLD}  Match{RESET}")
    print(_row("Internal match_id:", str(saved_match.id)))
    print(_row("External event_id:", saved_match.external_id or "—"))
    print(_row("Home team:", f"{home_team.name} (id={home_team.id})"))
    print(_row("Away team:", f"{away_team.name} (id={away_team.id})"))
    print(_row("Scheduled at:", str(saved_match.scheduled_at)))
    print(_row("Season:", saved_match.season))
    print(_row("Status:", saved_match.status.value))
    print(_row("Final score:", f"{saved_match.home_score_final} – {saved_match.away_score_final}"))
    print(_row("Regulation score:", f"{saved_match.home_score_regulation} – {saved_match.away_score_regulation}"))
    print(_row("Went to OT:", str(saved_match.went_to_overtime)))

    print(f"\n{BOLD}  QuarterStats  ({len(quarter_rows)} rows){RESET}")

    header = (
        f"  {'Period':<8} {'Type':<10} "
        f"{'Score H-A':<12} "
        f"{'FGA H/A':<12} "
        f"{'FTA H/A':<12} "
        f"{'OffReb H/A':<14} "
        f"{'TO H/A':<10} "
        f"{'Poss H/A':<16} "
        f"{'Pace H/A'}"
    )
    print(f"\n{CYAN}{header}{RESET}")
    print(f"  {'─' * 110}")

    def _fmt(v: int | float | None) -> str:
        if v is None:
            return "—"
        return str(int(v)) if isinstance(v, int) else f"{v:.1f}"

    for qs in quarter_rows:
        period_label = (
            f"Q{qs.period_number}" if qs.period_type.value == "quarter"
            else f"OT{qs.period_number}"
        )
        print(
            f"  {period_label:<8} {qs.period_type.value:<10} "
            f"{_fmt(qs.home_score)}-{_fmt(qs.away_score):<9} "
            f"{_fmt(qs.home_fga)}/{_fmt(qs.away_fga):<9} "
            f"{_fmt(qs.home_fta)}/{_fmt(qs.away_fta):<9} "
            f"{_fmt(qs.home_off_reb)}/{_fmt(qs.away_off_reb):<11} "
            f"{_fmt(qs.home_turnovers)}/{_fmt(qs.away_turnovers):<7} "
            f"{_fmt(qs.home_possessions)}/{_fmt(qs.away_possessions):<13} "
            f"{_fmt(qs.home_pace)}/{_fmt(qs.away_pace)}"
        )

    print(f"\n  {GREEN}✓ {len(quarter_rows)} rows successfully written to quarter_stats{RESET}")
    print(f"  {GREEN}✓ {pg_version}{RESET}")
    print(f"\n{sep}\n")

    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
