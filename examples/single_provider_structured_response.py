from __future__ import annotations

from pathlib import Path

from ai_consensus_core import (  # type: ignore[import-not-found]
    InvestigationRequest,
    UnifiedAIClient,
)


def main() -> None:
    prompts_dir = Path(__file__).parent / "prompts"
    default_system = (prompts_dir / "default_system.md").resolve()
    default_user = (prompts_dir / "default_user.md").resolve()
    alt_system = (prompts_dir / "openai_system_alt.md").resolve()
    alt_user = (prompts_dir / "openai_user_alt.md").resolve()

    client = UnifiedAIClient(
        config_yaml=f"""
prompts:
  default_system_prompt_path: "{default_system}"
  default_user_prompt_path: "{default_user}"
  provider_system_prompt_paths:
    openai: "{alt_system}"
  provider_user_prompt_paths:
    openai: "{alt_user}"
default_investigation_instructions: >
  Investigate the context and return strict JSON with calibrated confidence.
ai_providers:
  openai:
    enabled: true
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-5.3"
    api_base: "https://api.openai.com/v1"
    max_tokens: 1024
    timeout_seconds: 90
    temperature: 0.1
"""
    )

    request = InvestigationRequest(
        context_text=(
            "Customer feedback snippets:\n"
            "- Checkout is fast but promo code validation fails often.\n"
            "- Mobile app crashes on profile edit on Android 14.\n"
            "- Search quality improved and users find products faster.\n"
        ),
        output_schema={
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "overall_sentiment": {"type": "string"},
                "action_items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["topics", "overall_sentiment", "action_items"],
            "additionalProperties": False,
        },
        investigation_instructions=(
            "Extract key feedback topics and provide actionable summary."
        ),
        runtime_system_prompt_by_provider={
            "openai": (
                "You are a product analytics assistant. Return strict JSON only."
            )
        },
        runtime_user_prompt_by_provider={
            "openai": (
                "Task: {investigation_instructions}\n\n"
                "Context:\n{context_text}\n\n"
                "Return JSON with fields topics, overall_sentiment, action_items."
            )
        },
        metadata={"example": "single_provider_structured_response"},
    )

    result = client.run(request)
    single = next(
        response for response in result.provider_results if response.success
    )

    print("Provider:", single.provider_name)
    print("Model:", single.model)
    print("Structured payload:")
    print(single.parsed_payload)


if __name__ == "__main__":
    main()
