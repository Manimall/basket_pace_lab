"""
Base class for all data-collection workers.

Every source (Sofascore, Flashscore, ChinaNBL, …) exposes the same interface:
    await worker.run()   — async, does the actual work
    worker.run_sync()    — blocking wrapper for use from scripts / CLI

Subclasses only need to implement `async def run(self)`.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Abstract base for all data-collection workers."""

    @abstractmethod
    async def run(self) -> None:
        """Execute the full collection pipeline (async)."""
        ...

    def run_sync(self) -> None:
        """Blocking convenience wrapper for scripts and CLI entry points."""
        asyncio.run(self.run())
