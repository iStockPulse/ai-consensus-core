"""Unified request/response and consensus contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldEstimation:
    """Single field estimation with probability."""

    value: Any
    probability: float
    rationale: str = ""


@dataclass
class InvestigationRequest:
    """Unified request sent to all providers."""

    context_text: str
    output_schema: dict[str, Any]
    investigation_instructions: str = ""
    estimated_fields: list[str] = field(default_factory=list)
    runtime_system_prompt_by_provider: dict[str, str] = field(default_factory=dict)
    runtime_user_prompt_by_provider: dict[str, str] = field(default_factory=dict)
    runtime_provider_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedProviderResponse:
    """Normalized provider result payload."""

    provider_name: str
    model: str
    success: bool
    raw_response: str
    parsed_payload: dict[str, Any] | None
    field_estimations: dict[str, FieldEstimation] = field(default_factory=dict)
    confidence: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class ConsensusEstimate:
    """Consensus outcome and explainability artifacts."""

    success: bool
    consensus_payload: dict[str, Any]
    field_probabilities: dict[str, float]
    provider_contributions: dict[str, float]
    disagreement_metrics: dict[str, float]
    errors: list[str] = field(default_factory=list)


@dataclass
class AIOrchestrationResult:
    """Complete output of one orchestration run."""

    request: InvestigationRequest
    provider_results: list[UnifiedProviderResponse]
    consensus: ConsensusEstimate
