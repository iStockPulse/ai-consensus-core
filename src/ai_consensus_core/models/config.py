"""Typed package configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderSettings:
    """Configuration for a specific provider."""

    enabled: bool = True
    model: str = ""
    api_base: str = ""
    api_key_env: str = ""
    weight: float = 0.25
    max_tokens: int = 4096
    context_window: int = 200000
    temperature: float = 0.1
    timeout_seconds: int = 90
    structured_output_method: str = "json_schema"
    supports_reasoning: bool = False
    supports_extended_thinking: bool = False
    reasoning_effort: str | None = None
    thinking_budget: int | None = None
    extended_thinking_budget: int | None = None
    max_retries: int = 3
    retry_delay_seconds: int = 2
    default_system_prompt_path: str | None = None
    default_user_prompt_path: str | None = None


@dataclass
class PromptSettings:
    """Prompt source settings for defaults and provider overrides."""

    default_system_prompt_path: str | None = None
    default_user_prompt_path: str | None = None
    provider_system_prompt_paths: dict[str, str] = field(default_factory=dict)
    provider_user_prompt_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class ConsensusSettings:
    """Consensus engine settings."""

    strategy: str = "weighted_mean"
    confidence_weight: float = 0.5
    min_probability: float = 0.0
    max_probability: float = 1.0


@dataclass
class PackageConfig:
    """Top-level package configuration."""

    language: str = "en"
    providers: dict[str, ProviderSettings] = field(default_factory=dict)
    prompts: PromptSettings = field(default_factory=PromptSettings)
    consensus: ConsensusSettings = field(default_factory=ConsensusSettings)
    artifacts_log_file: str = "logs/ai_consensus_artifacts.jsonl"
    default_investigation_instructions: str = (
        "Investigate the supplied context, produce structured output, "
        "and quantify uncertainty where requested."
    )
    additional: dict[str, Any] = field(default_factory=dict)
