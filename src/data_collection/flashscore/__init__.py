"""Flashscore data-collection sub-package."""
from src.data_collection.flashscore.collector import FlashscoreCollector
from src.data_collection.flashscore.enricher import FlashscoreEnricher

__all__ = ["FlashscoreCollector", "FlashscoreEnricher"]
