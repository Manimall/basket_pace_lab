"""Shim — content moved to src.data_collection.sofascore.event."""
from src.data_collection.sofascore.event import (  # noqa: F401
    api_get, fetch_event_ids, load_existing_external_ids, process_event,
)

__all__ = ["api_get", "fetch_event_ids", "load_existing_external_ids", "process_event"]
