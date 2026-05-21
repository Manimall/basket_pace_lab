import asyncio
import logging
import sys

from sqlalchemy import text

from src.config import settings
from src.database.engine import create_tables, dispose_engine, get_session_factory

logging.basicConfig(
    level=getattr(logging, settings.app.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("basket_pace_lab")


async def check_db_connection() -> str:
    """Verify connectivity and return server version string."""
    async with get_session_factory()() as session:
        result = await session.execute(text("SELECT version()"))
        version: str = result.scalar_one()
    return version


async def main() -> None:
    log.info("Starting basket_pace_lab …")
    log.info("DB host: %s:%s / db: %s", settings.db.host, settings.db.port, settings.db.name)

    log.info("Creating tables (if not exist) …")
    await create_tables()
    log.info("Tables OK")

    pg_version = await check_db_connection()
    log.info("PostgreSQL: %s", pg_version)

    log.info("System ready.")

    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
