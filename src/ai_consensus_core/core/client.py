"""Unified AI client orchestrating providers and consensus."""

from __future__ import annotations

import logging
from typing import Any

from ai_consensus_core.config.loader import load_package_config
from ai_consensus_core.consensus.base import ConsensusStrategy
from ai_consensus_core.consensus.weighted import WeightedMeanConsensus
from ai_consensus_core.logging.artifacts import ArtifactLogger
from ai_consensus_core.models.config import PackageConfig
from ai_consensus_core.models.contracts import (
    AIOrchestrationResult,
    InvestigationRequest,
)
from ai_consensus_core.prompts.registry import PromptRegistry
from ai_consensus_core.providers.factory import (
    create_providers,
    query_all_providers,
)
from ai_consensus_core.providers.http_providers import attach_field_estimations


class UnifiedAIClient:
    """High-level entry point for provider orchestration and consensus."""

    def __init__(
        self,
        *,
        config_path: str | None = None,
        config_yaml: str | None = None,
        runtime_overrides: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.config: PackageConfig = load_package_config(
            config_path=config_path,
            config_yaml=config_yaml,
            runtime_overrides=runtime_overrides,
        )
        self.prompt_registry = PromptRegistry(self.config)
        self.providers = create_providers(self.config, logger=self.logger)
        self.consensus_strategy = self._build_consensus_strategy(self.config)
        self.artifact_logger = ArtifactLogger(
            output_path=self.config.artifacts_log_file, logger=self.logger
        )

    @staticmethod
    def _build_consensus_strategy(config: PackageConfig) -> ConsensusStrategy:
        if config.consensus.strategy == "weighted_mean":
            return WeightedMeanConsensus()
        raise ValueError(
            f"Unknown consensus strategy '{config.consensus.strategy}'."
        )

    def run(self, request: InvestigationRequest) -> AIOrchestrationResult:
        """Run all providers and compute consensus."""
        prompts_system: dict[str, str] = {}
        prompts_user: dict[str, str] = {}
        for provider, settings in self.providers:
            runtime_provider_settings = request.runtime_provider_overrides.get(
                provider.name, {}
            )
            merged_provider_settings = {
                "default_system_prompt_path": (
                    runtime_provider_settings.get("default_system_prompt_path")
                    or settings.default_system_prompt_path
                ),
                "default_user_prompt_path": (
                    runtime_provider_settings.get("default_user_prompt_path")
                    or settings.default_user_prompt_path
                ),
            }
            bundle = self.prompt_registry.resolve(
                provider_name=provider.name,
                request=request,
                provider_settings=merged_provider_settings,
            )
            prompts_system[provider.name] = bundle.system_prompt
            prompts_user[provider.name] = bundle.user_prompt

        provider_results = query_all_providers(
            providers=self.providers,
            system_prompt_by_provider=prompts_system,
            user_prompt_by_provider=prompts_user,
            schema=request.output_schema,
            logger=self.logger,
        )
        provider_results = attach_field_estimations(
            provider_results, request.estimated_fields
        )
        consensus = self.consensus_strategy.compute(
            provider_results=provider_results, config=self.config
        )
        output = AIOrchestrationResult(
            request=request,
            provider_results=provider_results,
            consensus=consensus,
        )
        self.artifact_logger.write(output)
        return output
