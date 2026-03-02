from __future__ import annotations

from ai_consensus_core import InvestigationRequest, UnifiedAIClient


def main() -> None:
    client = UnifiedAIClient(config_path="configs/config.yaml")
    request = InvestigationRequest(
        context_text=(
            "SaaS account portfolio summary:\n"
            "- Segment: SMB annual contracts\n"
            "- Average product usage down 14% in the last 30 days\n"
            "- Support response SLA misses increased from 2% to 11%\n"
            "- Competitor introduced lower-cost tier with migration tools\n"
            "- NPS dropped from 41 to 29 across top 50 at-risk accounts\n"
        ),
        output_schema={
            "type": "object",
            "properties": {
                "churn_probability_90d": {"type": "number"},
                "retention_lift_if_action": {"type": "number"},
                "top_retention_action": {"type": "string"},
                "churn_probability_90d_probability": {"type": "number"},
                "retention_lift_if_action_probability": {"type": "number"},
                "explanation": {"type": "string"},
            },
            "required": [
                "churn_probability_90d",
                "retention_lift_if_action",
                "top_retention_action",
            ],
            "additionalProperties": False,
        },
        investigation_instructions=(
            "Estimate 90-day churn probability and expected retention lift "
            "from the most effective immediate action."
        ),
        estimated_fields=[
            "churn_probability_90d",
            "retention_lift_if_action",
        ],
    )
    result = client.run(request)
    print(result.consensus.consensus_payload)
    print(result.consensus.provider_contributions)


if __name__ == "__main__":
    main()
