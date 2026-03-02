from __future__ import annotations

from ai_consensus_core import InvestigationRequest, UnifiedAIClient


def main() -> None:
    client = UnifiedAIClient(config_path="configs/config.yaml")
    request = InvestigationRequest(
        context_text=(
            "Factory line machine telemetry summary:\n"
            "- Motor vibration RMS increased 22% over baseline in 10 days\n"
            "- Bearing temperature now averages 81C (baseline 70C)\n"
            "- Unplanned downtime incidents in last 60 days: 2\n"
            "- Maintenance window available in 6 days\n"
            "- Spare bearing stock: 1 unit\n"
        ),
        output_schema={
            "type": "object",
            "properties": {
                "failure_probability_next_14d": {"type": "number"},
                "recommended_maintenance_date_offset_days": {"type": "number"},
                "downtime_impact_hours_if_delayed": {"type": "number"},
                "failure_probability_next_14d_probability": {"type": "number"},
                "recommended_maintenance_date_offset_days_probability": {
                    "type": "number"
                },
                "justification": {"type": "string"},
            },
            "required": [
                "failure_probability_next_14d",
                "recommended_maintenance_date_offset_days",
                "downtime_impact_hours_if_delayed",
            ],
            "additionalProperties": False,
        },
        investigation_instructions=(
            "Estimate near-term failure probability and the optimal "
            "maintenance timing to reduce expected downtime."
        ),
        estimated_fields=[
            "failure_probability_next_14d",
            "recommended_maintenance_date_offset_days",
        ],
    )
    result = client.run(request)
    print(result.consensus.consensus_payload)
    print(result.consensus.field_probabilities)


if __name__ == "__main__":
    main()
