# Backward-compat shim — real implementation lives in connection.py
from src.database.connection import (  # noqa: F401
    SessionFactory,
    create_tables,
    dispose_engine,
    get_engine,
    get_session,
    get_session_factory,
)

__all__ = [
    "SessionFactory",
    "get_engine",
    "get_session_factory",
    "get_session",
    "create_tables",
    "dispose_engine",
]
