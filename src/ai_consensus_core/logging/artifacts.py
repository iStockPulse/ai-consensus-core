"""Run artifact logging utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_consensus_core.models.contracts import AIOrchestrationResult


def _redact_secrets(payload: Any) -> Any:
    if isinstance(payload, dict):
        redacted: dict[str, Any] = {}
        for key, value in payload.items():
            key_lower = key.lower()
            if any(token in key_lower for token in ("api_key", "token", "secret", "password")):
                redacted[key] = "***"
            else:
                redacted[key] = _redact_secrets(value)
        return redacted
    if isinstance(payload, list):
        return [_redact_secrets(value) for value in payload]
    return payload


class ArtifactLogger:
    """Persist orchestration artifacts as JSONL."""

    def __init__(self, *, output_path: str | Path, logger: logging.Logger | None = None) -> None:
        self.output_path = Path(output_path)
        self.logger = logger or logging.getLogger(__name__)

    def write(self, result: AIOrchestrationResult) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request": asdict(result.request),
            "provider_results": [asdict(item) for item in result.provider_results],
            "consensus": asdict(result.consensus),
        }
        safe_row = _redact_secrets(row)
        with self.output_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(safe_row, ensure_ascii=False) + "\n")
        self.logger.info("Artifact entry written to %s", self.output_path)
