"""Weighted confidence-aware consensus strategy."""

from __future__ import annotations

import math
from typing import Any

from ai_consensus_core.consensus.base import ConsensusStrategy
from ai_consensus_core.models.config import PackageConfig
from ai_consensus_core.models.contracts import ConsensusEstimate, UnifiedProviderResponse


def _numeric(value: Any) -> float | None:
    try:
        if isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _flatten_numeric_fields(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    fields: dict[str, float] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            fields.update(_flatten_numeric_fields(value, path))
            continue
        numeric_value = _numeric(value)
        if numeric_value is not None:
            fields[path] = numeric_value
    return fields


class WeightedMeanConsensus(ConsensusStrategy):
    """Weighted mean aggregation with confidence-aware provider scaling."""

    def compute(
        self,
        *,
        provider_results: list[UnifiedProviderResponse],
        config: PackageConfig,
    ) -> ConsensusEstimate:
        successful = [result for result in provider_results if result.success and result.parsed_payload]
        if not successful:
            return ConsensusEstimate(
                success=False,
                consensus_payload={},
                field_probabilities={},
                provider_contributions={},
                disagreement_metrics={},
                errors=[result.error or "unknown error" for result in provider_results if result.error],
            )

        confidence_weight = config.consensus.confidence_weight
        field_accumulator: dict[str, float] = {}
        weight_accumulator: dict[str, float] = {}
        provider_contributions: dict[str, float] = {}

        for result in successful:
            provider_cfg = config.providers.get(result.provider_name)
            base_weight = provider_cfg.weight if provider_cfg else 1.0
            effective_weight = base_weight * ((1.0 - confidence_weight) + confidence_weight * result.confidence)
            provider_contributions[result.provider_name] = round(effective_weight, 6)
            numeric_fields = _flatten_numeric_fields(result.parsed_payload or {})
            for field_path, value in numeric_fields.items():
                field_accumulator[field_path] = field_accumulator.get(field_path, 0.0) + (value * effective_weight)
                weight_accumulator[field_path] = weight_accumulator.get(field_path, 0.0) + effective_weight

        consensus_payload: dict[str, Any] = {}
        field_probabilities: dict[str, float] = {}
        disagreement_metrics: dict[str, float] = {}

        for field_path, weighted_sum in field_accumulator.items():
            total_weight = weight_accumulator.get(field_path, 0.0)
            if total_weight <= 0:
                continue
            mean_value = weighted_sum / total_weight
            field_probabilities[field_path] = max(
                config.consensus.min_probability,
                min(config.consensus.max_probability, mean_value if 0.0 <= mean_value <= 1.0 else 0.0),
            )
            self._set_nested(consensus_payload, field_path, round(mean_value, 6))

            sample_values = []
            for result in successful:
                values = _flatten_numeric_fields(result.parsed_payload or {})
                if field_path in values:
                    sample_values.append(values[field_path])
            if sample_values:
                variance = sum((item - mean_value) ** 2 for item in sample_values) / len(sample_values)
                disagreement_metrics[field_path] = round(math.sqrt(variance), 6)

        return ConsensusEstimate(
            success=True,
            consensus_payload=consensus_payload,
            field_probabilities=field_probabilities,
            provider_contributions=provider_contributions,
            disagreement_metrics=disagreement_metrics,
            errors=[result.error or "" for result in provider_results if result.error],
        )

    @staticmethod
    def _set_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        cursor = target
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
