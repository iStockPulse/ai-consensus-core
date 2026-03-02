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
git+https://github.com/<org>/<repo>.git
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
uv add git+https://github.com/<org>/<repo>.git
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

## Minimal Config Example

```yaml
language: "en"

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

Set one provider `enabled: true`, all others `enabled: false`.
You still get the same normalized response model and artifact logging.

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
