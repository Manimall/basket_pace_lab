"""Sofascore data-collection sub-package."""
from src.data_collection.sofascore.collector import MassScheduler
from src.data_collection.sofascore.client import SofascoreClient

__all__ = ["MassScheduler", "SofascoreClient"]
