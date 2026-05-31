"""
Canonical async database connection management.

Provides a lazy singleton engine + session factory; all other modules should
import from here (or from the engine.py backward-compat shim).
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import settings
from src.database.models import Base

log = logging.getLogger(__name__)

# Canonical type for the async session factory returned by get_session_factory().
# Importing this alias (rather than re-deriving async_sessionmaker[AsyncSession]
# at each call site) keeps session-factory parameters strictly typed and avoids
# clashing with curl_cffi's same-named AsyncSession in scraper modules.
SessionFactory = async_sessionmaker[AsyncSession]

# Connection-pool sizing for the async engine.
_POOL_SIZE:    int = 10
_MAX_OVERFLOW: int = 20

_engine: AsyncEngine | None = None
_session_factory: SessionFactory | None = None


def get_engine() -> AsyncEngine:
    """Return the lazily-created singleton async engine.

    Returns:
        The process-wide ``AsyncEngine``, created on first call.
    """
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.db.async_dsn,
            echo=settings.app.debug,
            pool_size=_POOL_SIZE,
            max_overflow=_MAX_OVERFLOW,
            pool_pre_ping=True,
        )
        log.debug("AsyncEngine created: %s", settings.db.host)
    return _engine


def get_session_factory() -> SessionFactory:
    """Return the lazily-created singleton async session factory.

    Returns:
        A process-wide ``async_sessionmaker`` bound to the singleton engine.
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency-injection compatible async session generator."""
    async with get_session_factory()() as session:
        yield session


async def create_tables() -> None:
    """Create all tables. Prefer Alembic migrations in production."""
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Tables created (or already exist).")


async def dispose_engine() -> None:
    """Dispose the singleton engine and reset the cached factory.

    Safe to call when nothing was created. Used by scripts on shutdown so
    asyncpg connections close cleanly.
    """
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        log.debug("AsyncEngine disposed.")
    _engine = None
    _session_factory = None
