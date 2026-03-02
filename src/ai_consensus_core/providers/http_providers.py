"""HTTP provider implementations for OpenAI, Claude, Gemini, and Grok."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx

from ai_consensus_core.models.contracts import FieldEstimation, UnifiedProviderResponse
from ai_consensus_core.providers.base import AIProvider


def _parse_json_response(raw_text: str) -> tuple[dict[str, Any] | None, float]:
    try:
        return json.loads(raw_text), 1.0
    except json.JSONDecodeError:
        pass

    if "```json" in raw_text:
        try:
            start = raw_text.index("```json") + 7
            end = raw_text.index("```", start)
            return json.loads(raw_text[start:end].strip()), 0.95
        except (ValueError, json.JSONDecodeError):
            pass

    if "```" in raw_text:
        try:
            start = raw_text.index("```") + 3
            end = raw_text.index("```", start)
            return json.loads(raw_text[start:end].strip()), 0.9
        except (ValueError, json.JSONDecodeError):
            pass

    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}") + 1
        return json.loads(raw_text[start:end]), 0.85
    except (ValueError, json.JSONDecodeError):
        return None, 0.0


def _sanitize_error_text(text: str) -> str:
    return re.sub(r"([?&]key=)[^&'\"\s]+", r"\1***", text)


def _strip_schema_keys(value: Any, keys_to_strip: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_schema_keys(item, keys_to_strip)
            for key, item in value.items()
            if key not in keys_to_strip
        }
    if isinstance(value, list):
        return [_strip_schema_keys(item, keys_to_strip) for item in value]
    return value


def _extract_openai_responses_text(data: dict[str, Any]) -> str:
    """Extract text from OpenAI Responses API payload."""
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if isinstance(block.get("text"), str):
                            return block["text"]
                        if (
                            block.get("type") == "output_text"
                            and isinstance(block.get("text"), str)
                        ):
                            return block["text"]
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    return ""


def _extract_field_estimations(
    parsed_payload: dict[str, Any] | None,
    estimated_fields: list[str],
) -> dict[str, FieldEstimation]:
    estimations: dict[str, FieldEstimation] = {}
    if not parsed_payload or not estimated_fields:
        return estimations

    for field_name in estimated_fields:
        value = parsed_payload.get(field_name)
        probability = 0.0
        rationale = ""
        probability_key = f"{field_name}_probability"
        rationale_key = f"{field_name}_rationale"
        if probability_key in parsed_payload:
            try:
                probability = float(parsed_payload[probability_key])
            except (TypeError, ValueError):
                probability = 0.0
        if rationale_key in parsed_payload:
            rationale = str(parsed_payload[rationale_key])
        estimations[field_name] = FieldEstimation(
            value=value,
            probability=max(0.0, min(1.0, probability)),
            rationale=rationale,
        )
    return estimations


def _build_error_response(
    *,
    provider_name: str,
    model: str,
    start_time: float,
    error_text: str,
) -> UnifiedProviderResponse:
    return UnifiedProviderResponse(
        provider_name=provider_name,
        model=model,
        success=False,
        raw_response="",
        parsed_payload=None,
        confidence=0.0,
        latency_ms=(time.time() - start_time) * 1000,
        error=error_text,
    )


class OpenAIProvider(AIProvider):
    """OpenAI API provider with JSON schema response format."""

    async def call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> UnifiedProviderResponse:
        provider_name = "openai"
        start_time = time.time()
        try:
            api_key = os.getenv(self.settings.api_key_env)
            if not api_key:
                raise ValueError(f"{self.settings.api_key_env} is not set")

            url = f"{self.settings.api_base}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            timeout = httpx.Timeout(self.settings.timeout_seconds)
            fallback_models = [
                self.settings.model,
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4.1",
                "o3",
                "gpt-5",
            ]
            models_to_try: list[str] = []
            for model_name in fallback_models:
                if model_name and model_name not in models_to_try:
                    models_to_try.append(model_name)

            last_error = ""
            async with httpx.AsyncClient(timeout=timeout) as client:
                for model_name in models_to_try:
                    payload: dict[str, Any] = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "result_schema",
                                "schema": schema,
                                "strict": True,
                            },
                        },
                    }

                    if model_name.startswith("gpt-5"):
                        payload["max_completion_tokens"] = self.settings.max_tokens
                    else:
                        payload["max_tokens"] = self.settings.max_tokens
                        payload["temperature"] = self.settings.temperature

                    if self.settings.reasoning_effort and (
                        model_name.startswith("gpt-5")
                        or model_name.startswith("o")
                    ):
                        payload["reasoning_effort"] = self.settings.reasoning_effort

                    try:
                        response = await client.post(
                            url, json=payload, headers=headers
                        )
                    except httpx.TimeoutException:
                        last_error = (
                            f"model={model_name} timeout after "
                            f"{self.settings.timeout_seconds}s"
                        )
                        continue
                    except Exception as exc:  # noqa: BLE001
                        exc_text = _sanitize_error_text(str(exc))
                        if not exc_text:
                            exc_text = type(exc).__name__
                        last_error = f"model={model_name} error={exc_text}"
                        continue

                    if response.status_code >= 400:
                        status = response.status_code
                        body_snippet = response.text[:500]
                        last_error = (
                            f"model={model_name} status={status} "
                            f"body={_sanitize_error_text(body_snippet)}"
                        )
                        # Retry with fallback model on model/compatibility errors.
                        if status in {400, 404, 422, 429, 500, 502, 503, 504}:
                            continue
                        response.raise_for_status()

                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed, confidence = _parse_json_response(content)
                    return UnifiedProviderResponse(
                        provider_name=provider_name,
                        model=model_name,
                        success=parsed is not None,
                        raw_response=content,
                        parsed_payload=parsed,
                        confidence=confidence,
                        latency_ms=(time.time() - start_time) * 1000,
                        error=(
                            None
                            if parsed is not None
                            else "Failed to parse JSON response"
                        ),
                    )

            raise RuntimeError(
                "OpenAI failed across all fallback models. "
                f"Last error: {last_error}"
            )
        except RuntimeError:
            # Fallback to Responses API for models not available on chat/completions.
            try:
                responses_url = f"{self.settings.api_base}/responses"
                responses_payload: dict[str, Any] = {
                    "model": self.settings.model,
                    "input": [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": system_prompt,
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": user_prompt,
                                }
                            ],
                        },
                    ],
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "result_schema",
                            "schema": schema,
                            "strict": True,
                        }
                    },
                    "max_output_tokens": self.settings.max_tokens,
                }
                timeout = httpx.Timeout(self.settings.timeout_seconds)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        responses_url,
                        json=responses_payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()
                content = _extract_openai_responses_text(data)
                parsed, confidence = _parse_json_response(content)
                return UnifiedProviderResponse(
                    provider_name=provider_name,
                    model=self.settings.model,
                    success=parsed is not None,
                    raw_response=content,
                    parsed_payload=parsed,
                    confidence=confidence,
                    latency_ms=(time.time() - start_time) * 1000,
                    error=(
                        None
                        if parsed is not None
                        else "Failed to parse JSON response"
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                exc_text = _sanitize_error_text(str(exc))
                if not exc_text:
                    exc_text = type(exc).__name__
                return _build_error_response(
                    provider_name=provider_name,
                    model=self.settings.model,
                    start_time=start_time,
                    error_text=f"OpenAI error: {exc_text}",
                )
        except httpx.HTTPStatusError as exc:
            text = _sanitize_error_text(str(exc))
            return _build_error_response(
                provider_name=provider_name,
                model=self.settings.model,
                start_time=start_time,
                error_text=f"OpenAI HTTP error: {text}",
            )
        except Exception as exc:  # noqa: BLE001
            exc_text = _sanitize_error_text(str(exc))
            if not exc_text:
                exc_text = type(exc).__name__
            return _build_error_response(
                provider_name=provider_name,
                model=self.settings.model,
                start_time=start_time,
                error_text=f"OpenAI error: {exc_text}",
            )


class ClaudeProvider(AIProvider):
    """Anthropic API provider using tool_use for structured output."""

    async def call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> UnifiedProviderResponse:
        provider_name = "claude"
        start_time = time.time()
        try:
            api_key = os.getenv(self.settings.api_key_env)
            if not api_key:
                raise ValueError(f"{self.settings.api_key_env} is not set")

            payload: dict[str, Any] = {
                "model": self.settings.model,
                "max_tokens": self.settings.max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "tools": [{"name": "submit_result", "description": "Return final JSON", "input_schema": schema}],
                "tool_choice": {"type": "tool", "name": "submit_result"},
            }
            url = f"{self.settings.api_base}/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

            timeout = httpx.Timeout(self.settings.timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            blocks = data.get("content", [])
            tool_input: dict[str, Any] | None = None
            for block in blocks:
                if block.get("type") == "tool_use" and isinstance(block.get("input"), dict):
                    tool_input = block["input"]
                    break
            raw = json.dumps(tool_input or {}, ensure_ascii=False)
            parsed, confidence = _parse_json_response(raw)
            return UnifiedProviderResponse(
                provider_name=provider_name,
                model=self.settings.model,
                success=parsed is not None,
                raw_response=raw,
                parsed_payload=parsed,
                confidence=confidence,
                latency_ms=(time.time() - start_time) * 1000,
                error=None if parsed is not None else "Failed to parse tool output",
            )
        except Exception as exc:  # noqa: BLE001
            return _build_error_response(
                provider_name=provider_name,
                model=self.settings.model,
                start_time=start_time,
                error_text=f"Claude error: {_sanitize_error_text(str(exc))}",
            )


class GeminiProvider(AIProvider):
    """Google Gemini provider using response schema and JSON mime type."""

    async def call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> UnifiedProviderResponse:
        provider_name = "gemini"
        start_time = time.time()
        try:
            api_key = os.getenv(self.settings.api_key_env)
            if not api_key:
                raise ValueError(f"{self.settings.api_key_env} is not set")

            sanitized_schema = _strip_schema_keys(
                schema,
                {
                    "additionalProperties",
                    "$schema",
                    "strict",
                },
            )

            url = (
                f"{self.settings.api_base}/models/{self.settings.model}:generateContent"
                f"?key={api_key}"
            )
            timeout = httpx.Timeout(self.settings.timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                base_generation_cfg: dict[str, Any] = {
                    "responseMimeType": "application/json",
                    "temperature": self.settings.temperature,
                    "maxOutputTokens": self.settings.max_tokens,
                }
                if self.settings.thinking_budget:
                    base_generation_cfg["thinkingConfig"] = {
                        "thinkingBudget": self.settings.thinking_budget
                    }

                payload_with_schema: dict[str, Any] = {
                    "contents": [
                        {
                            "parts": [
                                {"text": f"{system_prompt}\n\n{user_prompt}"}
                            ]
                        }
                    ],
                    "generationConfig": dict(
                        base_generation_cfg,
                        responseSchema=sanitized_schema,
                    ),
                }

                response = await client.post(
                    url,
                    json=payload_with_schema,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 400:
                    # Retry without schema and thinking config when API rejects structure.
                    fallback_cfg = dict(base_generation_cfg)
                    fallback_cfg.pop("thinkingConfig", None)
                    payload_no_schema: dict[str, Any] = {
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "text": (
                                            f"{system_prompt}\n\n{user_prompt}\n\n"
                                            "Return valid JSON only."
                                        )
                                    }
                                ]
                            }
                        ],
                        "generationConfig": fallback_cfg,
                    }
                    response = await client.post(
                        url,
                        json=payload_no_schema,
                        headers={"Content-Type": "application/json"},
                    )

                response.raise_for_status()
                data = response.json()

            candidates = data.get("candidates", [])
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "")
            parsed, confidence = _parse_json_response(text)
            return UnifiedProviderResponse(
                provider_name=provider_name,
                model=self.settings.model,
                success=parsed is not None,
                raw_response=text,
                parsed_payload=parsed,
                confidence=confidence,
                latency_ms=(time.time() - start_time) * 1000,
                error=None if parsed is not None else "Failed to parse JSON response",
            )
        except Exception as exc:  # noqa: BLE001
            return _build_error_response(
                provider_name=provider_name,
                model=self.settings.model,
                start_time=start_time,
                error_text=f"Gemini error: {_sanitize_error_text(str(exc))}",
            )


class GrokProvider(AIProvider):
    """xAI Grok provider with OpenAI-compatible endpoint."""

    async def call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> UnifiedProviderResponse:
        provider_name = "grok"
        start_time = time.time()
        try:
            api_key = os.getenv(self.settings.api_key_env)
            if not api_key:
                raise ValueError(f"{self.settings.api_key_env} is not set")
            schema_text = json.dumps(schema, ensure_ascii=False)
            payload: dict[str, Any] = {
                "model": self.settings.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"{system_prompt}\n\nReturn only valid JSON matching this schema:\n"
                            f"{schema_text}"
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.settings.temperature,
                "max_tokens": self.settings.max_tokens,
            }
            url = f"{self.settings.api_base}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            timeout = httpx.Timeout(self.settings.timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            content = data["choices"][0]["message"]["content"]
            parsed, confidence = _parse_json_response(content)
            return UnifiedProviderResponse(
                provider_name=provider_name,
                model=self.settings.model,
                success=parsed is not None,
                raw_response=content,
                parsed_payload=parsed,
                confidence=confidence,
                latency_ms=(time.time() - start_time) * 1000,
                error=None if parsed is not None else "Failed to parse JSON response",
            )
        except Exception as exc:  # noqa: BLE001
            return _build_error_response(
                provider_name=provider_name,
                model=self.settings.model,
                start_time=start_time,
                error_text=f"Grok error: {_sanitize_error_text(str(exc))}",
            )


PROVIDER_CLASS_MAP = {
    "openai": OpenAIProvider,
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
    "grok": GrokProvider,
}


def attach_field_estimations(
    responses: list[UnifiedProviderResponse], estimated_fields: list[str]
) -> list[UnifiedProviderResponse]:
    """Attach field estimation data on normalized provider responses."""
    for response in responses:
        response.field_estimations = _extract_field_estimations(
            response.parsed_payload, estimated_fields
        )
    return responses
