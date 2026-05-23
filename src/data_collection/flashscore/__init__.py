"""Flashscore data-collection sub-package."""
from src.data_collection.flashscore.collector import FlashscoreCollector
from src.data_collection.flashscore.enricher import FlashscoreEnricher
from src.data_collection.flashscore.odds_enricher import OddsEnricher

__all__ = ["FlashscoreCollector", "FlashscoreEnricher", "OddsEnricher"]
