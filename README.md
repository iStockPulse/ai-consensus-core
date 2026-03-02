# ai-consensus-core

Reusable Python package for structured AI investigations across multiple
providers with unified request/response models, probability estimation,
and consensus aggregation.

The project is designed for open source reuse under the MIT License.

## What It Does

- Query OpenAI, Claude, Gemini, and Grok using one interface.
- Normalize provider responses into one model.
- Request probability estimates for specific output fields.
- Aggregate multiple provider outputs with a consensus strategy.
- Persist full run artifacts (request, provider outputs, consensus diagnostics).

## Installation

### Option 1: requirements.txt (public repo usage)

Add one of these lines to `requirements.txt`:

```txt
# when published on PyPI
ai-consensus-core>=0.1.0

# or install directly from GitHub (before/without PyPI)
git+https://github.com/iStockPulse/ai-consensus-core.git
```

Then:

```bash
pip install -r requirements.txt
```

### Option 2: uv project

```bash
uv add ai-consensus-core
```

Or from GitHub:

```bash
uv add git+https://github.com/iStockPulse/ai-consensus-core.git
```

### Option 3: local editable dependency (monorepo)

In your root `pyproject.toml`:

```toml
[project]
dependencies = ["ai-consensus-core>=0.1.0"]

[tool.uv.sources]
ai-consensus-core = { path = "packages/ai_consensus_core", editable = true }
```

Then:

```bash
uv sync
```

## Required Environment Variables

Set provider keys for the providers you enable in config:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `XAI_API_KEY`

## Quick Start

```python
from ai_consensus_core import InvestigationRequest, UnifiedAIClient

client = UnifiedAIClient(config_path="config.yaml")

request = InvestigationRequest(
    context_text="Weekly sales and support metrics...",
    output_schema={
        "type": "object",
        "properties": {
            "next_week_demand": {"type": "number"},
            "churn_probability": {"type": "number"},
            "next_week_demand_probability": {"type": "number"},
        },
        "required": ["next_week_demand", "churn_probability"],
        "additionalProperties": False,
    },
    investigation_instructions=(
        "Investigate demand and churn signals and return JSON only."
    ),
    estimated_fields=["next_week_demand", "churn_probability"],
)

result = client.run(request)
print(result.consensus.consensus_payload)
print(result.consensus.field_probabilities)
```

## Configuration Inputs

`UnifiedAIClient` accepts:

- `config_path`: YAML file path
- `config_yaml`: YAML content string
- `runtime_overrides`: in-memory override mapping merged on top

`language` is optional in YAML. If omitted (or empty/null), it defaults to `en`.

## Minimal Config Example

```yaml
prompts:
  default_system_prompt_path: "prompts/default_system.md"
  default_user_prompt_path: "prompts/default_user.md"
  provider_system_prompt_paths: {}
  provider_user_prompt_paths: {}

consensus:
  strategy: "weighted_mean"
  confidence_weight: 0.5
  min_probability: 0.0
  max_probability: 1.0

default_investigation_instructions: >
  Investigate the supplied context, return strict JSON matching
  the schema, and provide calibrated uncertainty.

ai_providers:
  openai:
    enabled: true
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-5.3"
    api_base: "https://api.openai.com/v1"
    weight: 0.28
    max_tokens: 4096
    timeout_seconds: 90
    temperature: 0.1
  claude:
    enabled: true
    api_key_env: "ANTHROPIC_API_KEY"
    model: "claude-sonnet-4-20250514"
    api_base: "https://api.anthropic.com/v1"
    weight: 0.27
    max_tokens: 8192
    timeout_seconds: 90
    temperature: 0.1
  gemini:
    enabled: true
    api_key_env: "GOOGLE_API_KEY"
    model: "gemini-2.5-pro"
    api_base: "https://generativelanguage.googleapis.com/v1beta"
    weight: 0.25
    max_tokens: 16384
    timeout_seconds: 90
    temperature: 0.1
  grok:
    enabled: true
    api_key_env: "XAI_API_KEY"
    model: "grok-3"
    api_base: "https://api.x.ai/v1"
    weight: 0.20
    max_tokens: 16000
    timeout_seconds: 90
    temperature: 0.1
```

## Single Provider vs Multi-Provider Consensus

### Use a single AI provider

Set your target provider `enabled: true`.
Any provider missing from config is treated as disabled automatically.
You still get the same normalized response model and artifact logging.

If you only need one structured provider response, read the first successful
item from `provider_results`:

```python
from pathlib import Path

from ai_consensus_core import InvestigationRequest, UnifiedAIClient

prompts_dir = Path("examples/prompts").resolve()

client = UnifiedAIClient(
    config_yaml=f"""
prompts:
  default_system_prompt_path: "{prompts_dir / 'default_system.md'}"
  default_user_prompt_path: "{prompts_dir / 'default_user.md'}"
  provider_system_prompt_paths:
    openai: "{prompts_dir / 'openai_system_alt.md'}"
  provider_user_prompt_paths:
    openai: "{prompts_dir / 'openai_user_alt.md'}"
default_investigation_instructions: >
  Investigate the context and return strict JSON.
ai_providers:
  openai:
    enabled: true
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-5.3"
    api_base: "https://api.openai.com/v1"
"""
)

request = InvestigationRequest(
    context_text="Summarize product feedback by topic and sentiment.",
    output_schema={
        "type": "object",
        "properties": {
            "topics": {"type": "array", "items": {"type": "string"}},
            "overall_sentiment": {"type": "string"},
        },
        "required": ["topics", "overall_sentiment"],
        "additionalProperties": False,
    },
    investigation_instructions="Return strict JSON only.",
    runtime_system_prompt_by_provider={
        "openai": "You are a product analyst. Return strict JSON only.",
    },
    runtime_user_prompt_by_provider={
        "openai": (
            "Task: {investigation_instructions}\n\n"
            "Context:\n{context_text}\n\n"
            "Return JSON matching schema exactly with fields:\n"
            "- topics: array of strings\n"
            "- overall_sentiment: string"
        )
    },
)

result = client.run(request)
single = next(r for r in result.provider_results if r.success)
print(single.provider_name)
print(single.parsed_payload)
```

Note: if you do not provide runtime prompts, the client uses configured prompt
files (if set) or built-in fallback prompts.

In `runtime_user_prompt_by_provider`, placeholders are resolved from request:

- `{investigation_instructions}` -> value from
  `InvestigationRequest.investigation_instructions`
  (or config default if empty)
- `{context_text}` -> value from `InvestigationRequest.context_text`

### Use multiple providers with consensus

Enable two or more providers. The package will:

1. run provider calls in parallel,
2. normalize outputs,
3. compute consensus with provider weights and confidence,
4. expose consensus diagnostics (`provider_contributions`,
   `disagreement_metrics`, `field_probabilities`).

## Runtime Prompt Overrides (Per Provider)

You can override prompt text per provider at request time:

```python
request = InvestigationRequest(
    context_text="...",
    output_schema={"type": "object", "properties": {}, "additionalProperties": True},
    runtime_system_prompt_by_provider={
        "openai": "You are an operations analyst. Return JSON only.",
        "claude": "Focus on causal reasoning and confidence calibration.",
    },
    runtime_user_prompt_by_provider={
        "gemini": "Analyze this context and produce strict JSON schema output.",
    },
)
```

## InvestigationRequest Runtime Controls (Deep Dive)

These fields let you control behavior per request without changing base YAML.

- `investigation_instructions: str`
  - Purpose: task-level instruction for this specific run.
  - Used by prompt rendering as `{investigation_instructions}`.
  - If empty, the client uses `default_investigation_instructions` from config.
  - Typical use: "Focus on leading indicators and return strict JSON only."

- `estimated_fields: list[str]`
  - Purpose: declare fields where uncertainty/probability is important.
  - Used by prompt rendering as `{estimated_fields}` and for post-processing
    field estimations in normalized provider results.
  - Typical use: `["risk_level", "probability", "eta_days"]`.

- `runtime_system_prompt_by_provider: dict[str, str]`
  - Purpose: replace system prompt per provider for this run.
  - Keys must be provider names (`openai`, `claude`, `gemini`, `grok`).
  - Highest-priority source for system prompt resolution.
  - Typical use: tune style/behavior for one provider without touching files.

- `runtime_user_prompt_by_provider: dict[str, str]`
  - Purpose: replace user prompt template/content per provider for this run.
  - Also highest-priority for user prompt resolution.
  - Use this when one provider needs extra formatting/schema guidance.

- `runtime_provider_overrides: dict[str, dict[str, Any]]`
  - Purpose: per-provider runtime settings overrides.
  - Current supported keys:
    - `default_system_prompt_path`
    - `default_user_prompt_path`
  - These are used when runtime prompt text is not directly provided.
  - Typical use: switch prompt files for one request (A/B prompt testing).

- `metadata: dict[str, Any]`
  - Purpose: custom run metadata for observability and traceability.
  - Stored in artifacts under `request.metadata`.
  - Typical use: `run_id`, `scenario`, `tenant`, `experiment`.

Prompt precedence per provider:

1. `runtime_system_prompt_by_provider` / `runtime_user_prompt_by_provider`
2. Prompt files from `runtime_provider_overrides` (if provided)
3. Provider prompt paths from config (`prompts.provider_*_prompt_paths`)
4. Provider-level default prompt paths in provider settings
5. Global defaults (`prompts.default_*_prompt_path`), then built-in fallback

Example using all runtime fields together:

```python
from ai_consensus_core import InvestigationRequest

request = InvestigationRequest(
    context_text="Q2 incidents increased in one region; staffing unchanged.",
    output_schema={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "risk_level": {"type": "number"},
            "eta_days": {"type": "number"},
        },
        "required": ["summary", "risk_level", "eta_days"],
        "additionalProperties": False,
    },
    investigation_instructions=(
        "Identify likely causes, quantify operational risk, and give ETA."
    ),
    estimated_fields=["risk_level", "eta_days"],
    runtime_system_prompt_by_provider={
        "openai": "You are an SRE analyst. Return strict JSON only.",
    },
    runtime_user_prompt_by_provider={
        "gemini": (
            "Assess incident trend from the context and return JSON "
            "matching schema exactly."
        ),
    },
    runtime_provider_overrides={
        "claude": {
            "default_system_prompt_path": "prompts/claude_system_alt.md",
            "default_user_prompt_path": "prompts/claude_user_alt.md",
        }
    },
    metadata={
        "run_id": "incident-q2-001",
        "scenario": "incident_triage",
        "experiment": "prompt-v2",
    },
)
```

## Output Model Overview

`client.run(...)` returns `AIOrchestrationResult` with:

- `request`: normalized request envelope
- `provider_results`: list of `UnifiedProviderResponse`
- `consensus`: `ConsensusEstimate`

`UnifiedProviderResponse` contains:

- `provider_name`, `model`, `success`
- `raw_response`, `parsed_payload`, `error`
- `confidence`, `latency_ms`
- `field_estimations`

`ConsensusEstimate` contains:

- `consensus_payload`
- `field_probabilities`
- `provider_contributions`
- `disagreement_metrics`
- `errors`

## Examples

See `examples/`:

- `basic_usage.py`
- `single_provider_structured_response.py`
- `demand_forecast_example.py`
- `customer_churn_example.py`
- `hiring_pipeline_example.py`
- `equipment_maintenance_example.py`

Run an example:

```bash
python examples/demand_forecast_example.py
```

## Logging and Artifacts

Each run writes JSONL artifacts (default:
`logs/ai_consensus_artifacts.jsonl`) including:

- full request payload
- all provider raw and parsed outputs
- consensus output and diagnostics
- provider errors (if any)

This makes post-run debugging and auditing straightforward.

## License

MIT. See `LICENSE`.
