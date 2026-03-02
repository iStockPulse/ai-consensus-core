from __future__ import annotations

from ai_consensus_core import InvestigationRequest, UnifiedAIClient


def main() -> None:
    client = UnifiedAIClient(config_path="configs/config.yaml")
    request = InvestigationRequest(
        context_text=(
            "Hiring pipeline snapshot for senior backend role:\n"
            "- Candidates in final loop: 12\n"
            "- Offer acceptance in last quarter: 58%\n"
            "- Time to fill target: 45 days, current forecast: 57 days\n"
            "- Compensation band sits at market median\n"
            "- Two competing employers are offering signing bonuses\n"
        ),
        output_schema={
            "type": "object",
            "properties": {
                "fill_within_45_days_probability": {"type": "number"},
                "expected_offers_needed": {"type": "number"},
                "best_pipeline_intervention": {"type": "string"},
                "expected_offers_needed_probability": {"type": "number"},
                "confidence_notes": {"type": "string"},
            },
            "required": [
                "fill_within_45_days_probability",
                "expected_offers_needed",
                "best_pipeline_intervention",
            ],
            "additionalProperties": False,
        },
        investigation_instructions=(
            "Investigate pipeline bottlenecks and estimate probability of "
            "hitting time-to-fill target and offers needed."
        ),
        estimated_fields=["expected_offers_needed"],
    )
    result = client.run(request)
    print(result.consensus.consensus_payload)
    print(result.consensus.disagreement_metrics)


if __name__ == "__main__":
    main()
