"""Provider abstractions."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ai_consensus_core.models.config import ProviderSettings
from ai_consensus_core.models.contracts import UnifiedProviderResponse


class AIProvider(ABC):
    """Abstract provider contract."""

    def __init__(
        self,
        *,
        name: str,
        settings: ProviderSettings,
        logger: logging.Logger | None = None,
    ) -> None:
        self.name = name
        self.settings = settings
        self.logger = logger or logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> UnifiedProviderResponse:
        """Call provider and return normalized result."""
