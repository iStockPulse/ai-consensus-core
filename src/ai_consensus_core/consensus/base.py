"""Consensus strategy interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ai_consensus_core.models.config import PackageConfig
from ai_consensus_core.models.contracts import ConsensusEstimate, UnifiedProviderResponse


class ConsensusStrategy(ABC):
    """Consensus strategy interface."""

    @abstractmethod
    def compute(
        self,
        *,
        provider_results: list[UnifiedProviderResponse],
        config: PackageConfig,
    ) -> ConsensusEstimate:
        """Compute consensus output from provider results."""
