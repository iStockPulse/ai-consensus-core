"""Prompt registry with default and per-provider templates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_consensus_core.models.config import PackageConfig
from ai_consensus_core.models.contracts import InvestigationRequest


@dataclass
class PromptBundle:
    """Resolved prompt pair for a provider call."""

    system_prompt: str
    user_prompt: str


def _read_markdown(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


class PromptRegistry:
    """Resolves prompts from defaults, provider and runtime overrides."""

    def __init__(self, config: PackageConfig) -> None:
        self._config = config
        self._default_system_prompt = _read_markdown(
            config.prompts.default_system_prompt_path
        )
        self._default_user_prompt = _read_markdown(
            config.prompts.default_user_prompt_path
        )

        self._provider_system_prompts: dict[str, str] = {}
        self._provider_user_prompts: dict[str, str] = {}

        for (
            provider,
            path,
        ) in config.prompts.provider_system_prompt_paths.items():
            self._provider_system_prompts[provider] = _read_markdown(path) or ""
        for (
            provider,
            path,
        ) in config.prompts.provider_user_prompt_paths.items():
            self._provider_user_prompts[provider] = _read_markdown(path) or ""

    def resolve(
        self,
        *,
        provider_name: str,
        request: InvestigationRequest,
        provider_settings: dict[str, Any] | None = None,
    ) -> PromptBundle:
        """Resolve system and user prompts for the provider."""
        system_prompt = request.runtime_system_prompt_by_provider.get(provider_name)
        user_prompt = request.runtime_user_prompt_by_provider.get(provider_name)

        if not system_prompt:
            system_prompt = self._provider_system_prompts.get(provider_name)
        if not user_prompt:
            user_prompt = self._provider_user_prompts.get(provider_name)

        if provider_settings:
            system_prompt = system_prompt or _read_markdown(
                provider_settings.get("default_system_prompt_path")
            )
            user_prompt = user_prompt or _read_markdown(
                provider_settings.get("default_user_prompt_path")
            )

        if not system_prompt:
            system_prompt = self._default_system_prompt or (
                "You are a careful analyst. Follow instructions exactly "
                "and return valid JSON only."
            )
        if not user_prompt:
            user_prompt = self._default_user_prompt or (
                "{investigation_instructions}\n\nContext:\n{context_text}\n\n"
                "If estimated_fields are provided, include probabilities "
                "between 0 and 1."
            )

        rendered_user_prompt = self._render_user_prompt(user_prompt, request)
        return PromptBundle(
            system_prompt=system_prompt,
            user_prompt=rendered_user_prompt,
        )

    def _render_user_prompt(
        self, template: str, request: InvestigationRequest
    ) -> str:
        """
        Render known placeholders without calling str.format().

        This avoids accidental formatting of literal JSON braces in runtime
        prompts that are already fully rendered by upstream layers.
        """
        investigation_instructions = (
            request.investigation_instructions.strip()
            if request.investigation_instructions
            else self._config.default_investigation_instructions
        )
        replacements = {
            "{investigation_instructions}": investigation_instructions,
            "{context_text}": request.context_text,
            "{estimated_fields}": ", ".join(request.estimated_fields),
        }
        rendered = template
        for token, value in replacements.items():
            rendered = rendered.replace(token, value)
        return rendered
