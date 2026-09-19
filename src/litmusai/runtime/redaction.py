"""Local supported-secret detection before durable storage or external classification."""

from __future__ import annotations

import re

from pydantic import JsonValue

from litmusai.runtime.config import ThreatPolicy
from litmusai.runtime.models import CapturedEvent, RuntimeEvent, SensitiveSignal


class Redactor:
    """Bounded format matching; deliberately does not claim general PII recognition."""

    def __init__(self, policy: ThreatPolicy) -> None:
        self.patterns: list[tuple[str, re.Pattern[str]]] = [
            (
                "github_token",
                re.compile(
                    r"(?:gh[pousr]_[A-Za-z0-9]{30,255}|"
                    r"github_pat_[A-Za-z0-9_]{30,255})"
                ),
            ),
            ("aws_access_key", re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}")),
            (
                "private_key",
                re.compile(
                    r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"
                    r"[\s\S]*?(?:-----END (?:[A-Z]+ )?PRIVATE KEY-----|$)"
                ),
            ),
        ]
        for pattern in policy.protected_patterns:
            expression = (
                re.escape(pattern.prefix)
                + "["
                + re.escape(pattern.alphabet)
                + "]{"
                + str(pattern.min_suffix)
                + ","
                + str(pattern.max_suffix)
                + "}"
            )
            self.patterns.append((pattern.name, re.compile(expression)))

    def text(self, value: str) -> tuple[str, list[str]]:
        """Replace supported values and return data types, never matched values."""
        kinds: list[str] = []
        for name, pattern in self.patterns:
            value, count = pattern.subn("[REDACTED]", value)
            if count:
                kinds.append(name)
        return value, kinds

    def capture(self, event: RuntimeEvent) -> CapturedEvent:
        """Sanitize payload recursively; reject protected values in correlation metadata."""
        event.validate_boundary()
        data = event.model_dump(mode="json")
        metadata = {k: v for k, v in data.items() if k != "payload"}
        for value in metadata.values():
            if isinstance(value, str) and self.text(value)[1]:
                raise ValueError("protected value in event metadata")
        signals: list[SensitiveSignal] = []

        def scrub(value: JsonValue, location: str, depth: int = 0) -> JsonValue:
            if depth > 16:
                raise ValueError("payload nesting exceeds limit")
            if isinstance(value, str):
                redacted, kinds = self.text(value)
                signals.extend(SensitiveSignal(data_type=k, location=location) for k in kinds)
                return redacted
            if isinstance(value, list):
                return [scrub(item, location, depth + 1) for item in value]
            if isinstance(value, dict):
                return {
                    str(scrub(k, location, depth + 1)): scrub(v, location, depth + 1)
                    for k, v in value.items()
                }
            return value

        payload = data["payload"]
        for field, value in payload.items():
            # Structural fields must remain valid, and cannot contain protected values.
            if field in {"kind", "name", "role", "error_type"}:
                if isinstance(value, str) and self.text(value)[1]:
                    raise ValueError("protected value in structural field")
            else:
                payload[field] = scrub(value, field)
        return CapturedEvent(event=RuntimeEvent.model_validate(data), signals=signals[:100])
