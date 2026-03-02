"""Config loaders supporting path or YAML string."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ai_consensus_core.models.config import (
    ConsensusSettings,
    PackageConfig,
    PromptSettings,
    ProviderSettings,
)


def _merge_dicts(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _provider_from_raw(raw: dict[str, Any]) -> ProviderSettings:
    return ProviderSettings(
        enabled=bool(raw.get("enabled", True)),
        model=str(raw.get("model", "")),
        api_base=str(raw.get("api_base", "")),
        api_key_env=str(raw.get("api_key_env", "")),
        weight=float(raw.get("weight", 0.25)),
        max_tokens=int(raw.get("max_tokens", 4096)),
        context_window=int(raw.get("context_window", 200000)),
        temperature=float(raw.get("temperature", 0.1)),
        timeout_seconds=int(raw.get("timeout_seconds", 90)),
        structured_output_method=str(
            raw.get("structured_output_method", "json_schema")
        ),
        supports_reasoning=bool(raw.get("supports_reasoning", False)),
        supports_extended_thinking=bool(
            raw.get("supports_extended_thinking", False)
        ),
        reasoning_effort=raw.get("reasoning_effort"),
        thinking_budget=raw.get("thinking_budget"),
        extended_thinking_budget=raw.get("extended_thinking_budget"),
        max_retries=int(raw.get("max_retries", 3)),
        retry_delay_seconds=int(raw.get("retry_delay_seconds", 2)),
        default_system_prompt_path=raw.get("default_system_prompt_path"),
        default_user_prompt_path=raw.get("default_user_prompt_path"),
    )


def _normalize_raw_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize both legacy and package-native config shapes."""
    providers = raw.get("providers")
    if providers is None:
        providers = raw.get("ai_providers", {})

    prompts = raw.get("prompts", {})
    consensus = raw.get("consensus", {})
    language_raw = raw.get("language")
    if language_raw is None:
        language_raw = raw.get("assessment", {}).get("language")
    if isinstance(language_raw, str):
        language = language_raw.strip().lower() or "en"
    else:
        language = "en"

    artifacts_log_file = raw.get("artifacts_log_file")
    if artifacts_log_file is None:
        artifacts_log_file = "logs/ai_consensus_artifacts.jsonl"

    return {
        "language": language,
        "providers": providers,
        "prompts": prompts,
        "consensus": consensus,
        "artifacts_log_file": artifacts_log_file,
        "default_investigation_instructions": raw.get(
            "default_investigation_instructions",
            (
                "Investigate the supplied context, produce structured output, "
                "and quantify uncertainty where requested."
            ),
        ),
        "additional": {
            k: v
            for k, v in raw.items()
            if k
            not in {
                "language",
                "providers",
                "ai_providers",
                "prompts",
                "consensus",
                "artifacts_log_file",
                "default_investigation_instructions",
            }
        },
    }


def load_package_config(
    *,
    config_path: str | Path | None = None,
    config_yaml: str | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> PackageConfig:
    """
    Load package config from path or YAML text.

    Precedence:
    defaults < config source < runtime_overrides
    """
    sources_set = sum(
        1 for candidate in (config_path, config_yaml) if candidate is not None
    )
    if sources_set > 1:
        raise ValueError("Provide only one of config_path or config_yaml.")

    source_data: dict[str, Any] = {}
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise ValueError("YAML config must be a mapping at top level.")
        source_data = parsed
    elif config_yaml is not None:
        parsed = yaml.safe_load(config_yaml)
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise ValueError(
                "YAML config text must be a mapping at top level."
            )
        source_data = parsed
    normalized = _normalize_raw_config(source_data)
    if runtime_overrides:
        normalized = _merge_dicts(normalized, runtime_overrides)

    providers_raw = normalized.get("providers", {}) or {}
    prompts_raw = normalized.get("prompts", {}) or {}
    consensus_raw = normalized.get("consensus", {}) or {}

    providers: dict[str, ProviderSettings] = {
        name: _provider_from_raw(raw_cfg if isinstance(raw_cfg, dict) else {})
        for name, raw_cfg in providers_raw.items()
    }

    prompts = PromptSettings(
        default_system_prompt_path=prompts_raw.get(
            "default_system_prompt_path"
        ),
        default_user_prompt_path=prompts_raw.get("default_user_prompt_path"),
        provider_system_prompt_paths=dict(
            prompts_raw.get("provider_system_prompt_paths", {})
        ),
        provider_user_prompt_paths=dict(
            prompts_raw.get("provider_user_prompt_paths", {})
        ),
    )
    consensus = ConsensusSettings(
        strategy=str(consensus_raw.get("strategy", "weighted_mean")),
        confidence_weight=float(
            consensus_raw.get("confidence_weight", 0.5)
        ),
        min_probability=float(consensus_raw.get("min_probability", 0.0)),
        max_probability=float(consensus_raw.get("max_probability", 1.0)),
    )

    return PackageConfig(
        language=str(normalized.get("language") or "en"),
        providers=providers,
        prompts=prompts,
        consensus=consensus,
        artifacts_log_file=str(
            normalized.get(
                "artifacts_log_file", "logs/ai_consensus_artifacts.jsonl"
            )
        ),
        default_investigation_instructions=str(
            normalized.get("default_investigation_instructions", "")
        ),
        additional=dict(normalized.get("additional", {})),
    )
