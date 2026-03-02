from __future__ import annotations

from ai_consensus_core import InvestigationRequest, UnifiedAIClient


def main() -> None:
    client = UnifiedAIClient(config_path="configs/config.yaml")
    request = InvestigationRequest(
        context_text="Example context payload",
        output_schema={
            "type": "object",
            "properties": {
                "risk_probability": {"type": "number"},
                "risk_probability_probability": {"type": "number"},
            },
            "required": ["risk_probability"],
            "additionalProperties": True,
        },
        investigation_instructions="Estimate risk and uncertainty.",
        estimated_fields=["risk_probability"],
    )
    result = client.run(request)
    print(result.consensus.consensus_payload)


if __name__ == "__main__":
    main()
