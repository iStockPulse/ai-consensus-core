"""Provider factory and concurrent query helpers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from typing import Any, cast

from ai_consensus_core.models.config import PackageConfig, ProviderSettings
from ai_consensus_core.models.contracts import UnifiedProviderResponse
from ai_consensus_core.providers.base import AIProvider
from ai_consensus_core.providers.http_providers import PROVIDER_CLASS_MAP


def create_providers(
    config: PackageConfig,
    *,
    logger: logging.Logger | None = None,
) -> list[tuple[AIProvider, ProviderSettings]]:
    """Create enabled providers from typed package config."""
    resolved_logger = logger or logging.getLogger(__name__)
    providers: list[tuple[AIProvider, ProviderSettings]] = []

    # Missing provider config is treated as disabled.
    for provider_name in PROVIDER_CLASS_MAP:
        settings = config.providers.get(
            provider_name, ProviderSettings(enabled=False)
        )
        if not settings.enabled:
            resolved_logger.info(
                "Provider %s disabled; skipping.", provider_name
            )
            continue
        provider_cls = PROVIDER_CLASS_MAP.get(provider_name)
        if provider_cls is None:
            resolved_logger.warning(
                "Unknown provider '%s'; skipping.", provider_name
            )
            continue
        api_key = os.getenv(settings.api_key_env, "")
        if not api_key:
            resolved_logger.warning(
                "Provider %s missing env var %s; skipping.",
                provider_name,
                settings.api_key_env,
            )
            continue
        concrete_cls = cast(type[Any], provider_cls)
        providers.append(
            (
                concrete_cls(
                    name=provider_name,
                    settings=settings,
                    logger=resolved_logger,
                ),
                settings,
            )
        )

    for configured_name in config.providers:
        if configured_name not in PROVIDER_CLASS_MAP:
            resolved_logger.warning(
                "Unknown provider '%s'; skipping.", configured_name
            )

    if not providers:
        raise RuntimeError(
            "No AI providers available after configuration and env checks."
        )
    return providers


def _call_provider_in_thread(
    provider: AIProvider,
    *,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
) -> UnifiedProviderResponse:
    """Run async provider call in a worker thread."""
    return asyncio.run(
        provider.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
        )
    )


def query_all_providers(
    *,
    providers: list[tuple[AIProvider, ProviderSettings]],
    system_prompt_by_provider: dict[str, str],
    user_prompt_by_provider: dict[str, str],
    schema: dict[str, Any],
    logger: logging.Logger | None = None,
) -> list[UnifiedProviderResponse]:
    """Run provider calls in ThreadPoolExecutor and normalize exceptions."""
    resolved_logger = logger or logging.getLogger(__name__)
    names: list[str] = []
    models: list[str] = []
    index_by_future: dict[
        concurrent.futures.Future[UnifiedProviderResponse],
        int,
    ] = {}
    max_workers = max(1, len(providers))

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        for idx, (provider, settings) in enumerate(providers):
            names.append(provider.name)
            models.append(settings.model)
            future = executor.submit(
                _call_provider_in_thread,
                provider,
                system_prompt=system_prompt_by_provider[provider.name],
                user_prompt=user_prompt_by_provider[provider.name],
                schema=schema,
            )
            index_by_future[future] = idx

        ordered_results: list[UnifiedProviderResponse | None] = [
            None
        ] * len(providers)
        for future in concurrent.futures.as_completed(index_by_future):
            idx = index_by_future[future]
            try:
                ordered_results[idx] = future.result()
            except Exception as exc:  # noqa: BLE001
                ordered_results[idx] = UnifiedProviderResponse(
                    provider_name=names[idx],
                    model=models[idx],
                    success=False,
                    raw_response="",
                    parsed_payload=None,
                    confidence=0.0,
                    latency_ms=0.0,
                    error=f"Provider call failed: {exc}",
                )

    final_results = [item for item in ordered_results if item is not None]
    resolved_logger.info(
        "Provider calls completed: %d total, %d successful",
        len(final_results),
        sum(1 for result in final_results if result.success),
    )
    return final_results
