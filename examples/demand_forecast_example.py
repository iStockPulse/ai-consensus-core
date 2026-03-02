from __future__ import annotations

from ai_consensus_core import InvestigationRequest, UnifiedAIClient


def main() -> None:
    client = UnifiedAIClient(config_path="configs/config.yaml")
    request = InvestigationRequest(
        context_text=(
            "E-commerce weekly snapshot:\n"
            "- Last 6 weeks unit sales: 1200, 1280, 1310, 1405, 1490, 1575\n"
            "- Promo calendar: spring campaign starts in 5 days\n"
            "- Inventory on hand: 4200 units\n"
            "- Lead time from supplier: 14 days\n"
            "- Competitor launched 10% discount on similar SKU line\n"
        ),
        output_schema={
            "type": "object",
            "properties": {
                "next_week_demand_units": {"type": "number"},
                "stockout_probability": {"type": "number"},
                "recommended_reorder_units": {"type": "number"},
                "next_week_demand_units_probability": {"type": "number"},
                "recommended_reorder_units_probability": {"type": "number"},
                "notes": {"type": "string"},
            },
            "required": [
                "next_week_demand_units",
                "stockout_probability",
                "recommended_reorder_units",
            ],
            "additionalProperties": False,
        },
        investigation_instructions=(
            "Investigate demand drivers and estimate next week demand, "
            "stockout probability, and reorder recommendation."
        ),
        estimated_fields=[
            "next_week_demand_units",
            "recommended_reorder_units",
        ],
    )
    result = client.run(request)
    print(result.consensus.consensus_payload)
    print(result.consensus.field_probabilities)


if __name__ == "__main__":
    main()
