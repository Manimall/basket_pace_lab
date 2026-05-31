"""Sofascore data-collection sub-package."""
from src.data_collection.sofascore.client import SofascoreClient
from src.data_collection.sofascore.collector import MassScheduler

__all__ = ["MassScheduler", "SofascoreClient"]
