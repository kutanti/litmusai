"""Local policy checks and an injectable semantic-classifier contract."""

from __future__ import annotations

import json
from email.utils import parsedate_to_datetime
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Literal, Protocol

import httpx
from pydantic import Field

from litmusai.runtime.config import (
    ClassifierConfig,
    ConversationPolicy,
    ThreatPolicy,
    ToolUsagePolicy,
    secret,
)
from litmusai.runtime.models import (
    CapturedEvent,
    Contract,
    DetectionResult,
    Message,
    RiskScore,
    Stage,
    ToolActivity,
    utcnow,
)

RESPONSE_LIMIT_BYTES = 16384


def result(
    captured: CapturedEvent,
    policy: ThreatPolicy | ToolUsagePolicy | ConversationPolicy,
    detector: str,
    **values: object,
) -> DetectionResult:
    """Construct a versioned finding without exposing the raw event payload."""
    event = captured.event
    stage: Stage = (
        "attempt"
        if detector in {"prompt_injection", "prompt_injection_review"}
        else "observed"
        if event.event_type in {"tool.completed", "response.completed"}
        else "requested"
        if event.event_type.startswith("tool.")
        else "attempt"
    )
    return DetectionResult.model_validate(
        dict(
            event_id=event.event_id,
            detector=detector,
            policy_id=policy.policy_id,
            policy_version=policy.version,
            severity=policy.severity,
            stage=stage,
            source_event_ids=[event.event_id],
            context_incomplete=captured.capture_gap,
            **values,
        )
    )


def tool_policy(captured: CapturedEvent, policy: ThreatPolicy) -> list[DetectionResult]:
    """Check exact tool/destination allowlists supplied by trusted configuration."""
    payload = captured.event.payload
    if not isinstance(payload, ToolActivity):
        return [
            result(captured, policy, "tool_policy", outcome="skipped", reason="not a tool event")
        ]
    evidence = [f"tool={payload.name}"]
    if payload.destination is not None:
        evidence.append(f"destination={payload.destination}")
    findings = [
        result(
            captured,
            policy,
            "tool_policy",
            category="unauthorized_tool",
            evidence=evidence,
            outcome=(
                "insufficient_context"
                if policy.allowed_tools is None
                else "clear"
                if payload.name in policy.allowed_tools
                else "detected"
            ),
            reason=(
                "tool allowlist unavailable"
                if policy.allowed_tools is None
                else "tool allowed"
                if payload.name in policy.allowed_tools
                else "tool outside allowlist"
            ),
        )
    ]
    if payload.destination is not None or payload.name in policy.destination_tools:
        missing = payload.destination is None or policy.allowed_destinations is None
        forbidden = not missing and payload.destination not in (policy.allowed_destinations or [])
        findings.append(
            result(
                captured,
                policy,
                "tool_policy",
                category="forbidden_destination",
                evidence=evidence,
                outcome="insufficient_context" if missing else "detected" if forbidden else "clear",
                reason=(
                    "destination or authorization unavailable"
                    if missing
                    else "destination outside allowlist"
                    if forbidden
                    else "destination allowed"
                ),
            )
        )
    return findings


def sensitive_data(captured: CapturedEvent, policy: ThreatPolicy) -> list[DetectionResult]:
    """Only outbound arguments/responses support an exposure finding, not inbound results."""
    payload = captured.event.payload
    eligible = (
        isinstance(payload, ToolActivity) or captured.event.event_type == "response.completed"
    )
    signals = [s for s in captured.signals if s.location in {"arguments", "text"}]
    if not eligible or not signals:
        return [
            result(
                captured,
                policy,
                "sensitive_data",
                outcome="skipped" if not eligible else "clear",
                reason="no supported outbound sensitive-data signal",
            )
        ]
    destination = payload.destination if isinstance(payload, (Message, ToolActivity)) else None
    missing = destination is None or policy.allowed_destinations is None
    forbidden = not missing and destination not in (policy.allowed_destinations or [])
    return [
        result(
            captured,
            policy,
            "sensitive_data",
            category="sensitive_data",
            outcome="insufficient_context" if missing else "detected" if forbidden else "clear",
            reason=(
                "potential exposure; destination or authorization unavailable"
                if missing
                else "protected data supplied toward a forbidden destination"
                if forbidden
                else "protected data used with an allowed destination"
            ),
            evidence=[f"data_type={s.data_type}; location={s.location}" for s in signals]
            + ([f"destination={destination}"] if destination else []),
        )
    ]


class VerdictBase(Contract):
    """Fields shared by injection-classifier and conversation-policy verdicts."""

    outcome: Literal["detected", "clear", "insufficient_context", "needs_review"]
    reason: str = Field(min_length=1, max_length=1000)
    context_incomplete: bool = Field(default=False, strict=True)
    reported_cost_usd: float | None = Field(default=None, ge=0, strict=True)
    estimated_cost_usd: float | None = Field(default=None, ge=0, strict=True)
    input_tokens: int | None = Field(default=None, ge=0, strict=True)
    output_tokens: int | None = Field(default=None, ge=0, strict=True)
    detector_version: str | None = Field(
        default=None, min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:+=/-]+$"
    )


class ClassifierVerdict(VerdictBase):
    """Minimal strict semantic response; severity remains an operator policy decision.

    ``reported_cost_usd`` is what the provider reported. ``estimated_cost_usd`` is
    calculated by an adapter from token counts and an operator-configured price.
    ``detector_version`` identifies the provider model and question wording; it is
    appended to the configured classifier version.
    """

    risk_score: RiskScore | None = None


def versioned(configured: str, provider: str | None) -> str:
    """Append a provider-reported model and wording version to the configured version."""
    return f"{configured}+{provider}" if provider else configured


def check_risk_score(verdict: ClassifierVerdict) -> None:
    """Reject a score that contradicts the outcome it accompanies."""
    score = verdict.risk_score
    if score is None or verdict.outcome == "insufficient_context":
        return
    if (score.value >= score.threshold) != (verdict.outcome == "detected"):
        raise ValueError("classifier score contradicts its outcome")


class ProviderRateLimitError(Exception):
    """A provider refused or deferred a call for rate limiting.

    The engine records a skipped finding with degraded coverage; it is never clear.
    """

    def __init__(
        self, retry_after_seconds: float | None = None, *, provider_called: bool = True
    ) -> None:
        super().__init__("provider rate limited")
        self.retry_after_seconds = retry_after_seconds
        self.provider_called = provider_called


def retry_after_seconds(value: str | None) -> float | None:
    """Parse a Retry-After header given in seconds or as an HTTP date."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - utcnow()).total_seconds()
        except (TypeError, ValueError):
            return None
    return max(0.0, seconds) if seconds == seconds else None


def new_client(*, max_connections: int = 10) -> httpx.AsyncClient:
    """HTTP client without redirects, environment proxies, or stored cookies."""
    return httpx.AsyncClient(
        follow_redirects=False,
        trust_env=False,
        cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])),
        limits=httpx.Limits(
            max_connections=max_connections, max_keepalive_connections=max_connections
        ),
    )


async def post_json(
    client: httpx.AsyncClient | None,
    url: str,
    body: dict[str, object],
    *,
    headers: dict[str, str],
    timeout: float,
    rate_limited_statuses: frozenset[int] = frozenset(),
) -> bytes:
    """POST once and return a bounded response body; there are no retries.

    With ``client`` set, the shared pool is reused; otherwise a client is created
    for this request. Statuses in ``rate_limited_statuses`` raise
    :class:`ProviderRateLimitError`; every other non-2xx status raises.
    """
    if client is None:
        async with new_client(max_connections=1) as owned:
            return await post_json(
                owned,
                url,
                body,
                headers=headers,
                timeout=timeout,
                rate_limited_statuses=rate_limited_statuses,
            )
    async with client.stream("POST", url, json=body, headers=headers, timeout=timeout) as response:
        if response.status_code in rate_limited_statuses:
            raise ProviderRateLimitError(
                retry_after_seconds(response.headers.get("retry-after"))
            )
        response.raise_for_status()
        data = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=8192):
            data.extend(chunk)
            if len(data) > RESPONSE_LIMIT_BYTES:
                raise ValueError("classifier response exceeds limit")
    return bytes(data)


class InjectionClassifier(Protocol):
    """Implement this interface to use a client-selected injection classifier."""

    async def classify(
        self,
        captured: CapturedEvent,
        context: list[CapturedEvent],
        incomplete: bool,
    ) -> ClassifierVerdict:
        """Classify sanitized, source-labelled context without invoking the agent."""
        ...


class HTTPInjectionClassifier:
    """Adapter for a client-owned classifier endpoint implementing the documented contract.

    This adapter supplies no built-in LLM or vendor model. Enable it only with a
    classifier implementation whose detection quality has been evaluated.
    """

    def __init__(
        self, config: ClassifierConfig, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self.config = config
        self.client = client

    def bind_client(self, client: httpx.AsyncClient | None) -> None:
        """Use a shared pooled client; ``None`` restores a client per request."""
        self.client = client

    async def classify(
        self,
        captured: CapturedEvent,
        context: list[CapturedEvent],
        incomplete: bool,
    ) -> ClassifierVerdict:
        """Send bounded sanitized data; reject oversized and malformed responses."""
        items: list[dict[str, object]] = []
        remaining = self.config.context_chars
        for item in [captured, *reversed(context)]:
            event = item.event
            payload = json.dumps(event.payload.model_dump(mode="json"), ensure_ascii=True)
            selected = payload[: max(0, remaining)]
            if len(selected) != len(payload):
                incomplete = True
            items.append(
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "source_trust": "untrusted",
                    "content": selected,
                }
            )
            remaining -= len(selected)
            if remaining <= 0:
                incomplete = True
                break
        body = {
            "schema_version": "1.0",
            "task": "detect_prompt_injection",
            "content_is_untrusted": True,
            "context_incomplete": incomplete,
            "events": items,
        }
        verdict = ClassifierVerdict.model_validate_json(await self._request(body))
        return verdict.model_copy(
            update={
                "context_incomplete": incomplete or verdict.context_incomplete,
            }
        )

    async def _request(self, body: dict[str, object]) -> bytes:
        return await post_json(
            self.client,
            self.config.endpoint,
            body,
            headers={"Authorization": f"Bearer {secret(self.config.api_key_env)}"},
            timeout=self.config.timeout_seconds,
        )


class LakeraInjectionClassifier(HTTPInjectionClassifier):
    """Lakera Guard v2 adapter using per-detector results, including Detect mode.

    The provider's top-level ``flagged`` is forced false in Detect mode. Only a
    prompt_attack breakdown for the current message is used as injection evidence.
    Configure Prompt Defense on user::content and tool::content in the provider project.
    """

    async def classify(
        self,
        captured: CapturedEvent,
        context: list[CapturedEvent],
        incomplete: bool,
    ) -> ClassifierVerdict:
        """Screen the current message last, with bounded earlier messages as context."""
        messages: list[dict[str, object]] = []
        remaining = self.config.context_chars
        for item in [captured, *reversed(context)]:
            payload = item.event.payload
            if isinstance(payload, Message):
                content = payload.text
                role = "assistant" if item.event.event_type == "response.completed" else "user"
            elif isinstance(payload, ToolActivity) and item.event.event_type == "tool.completed":
                content = (
                    payload.result
                    if isinstance(payload.result, str)
                    else json.dumps(payload.result, ensure_ascii=True)
                )
                role = "tool"
            else:
                continue
            if len(content) > remaining:
                incomplete = True
            message: dict[str, object] = {"role": role, "content": content[:remaining]}
            if role == "tool":
                message["tool_call_id"] = item.event.tool_call_id
            messages.append(message)
            remaining -= len(str(message["content"]))
            if role == "tool" and isinstance(payload, ToolActivity):
                arguments = json.dumps(payload.arguments, ensure_ascii=True)
                if len(arguments) > remaining:
                    arguments = "{}"
                    incomplete = True
                remaining = max(0, remaining - len(arguments))
                # Reverse construction keeps the matching call immediately before its result.
                # These are captured call fields, not a new invocation of the agent/tool.
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.event.tool_call_id,
                                "type": "function",
                                "function": {"name": payload.name, "arguments": arguments},
                            }
                        ],
                    }
                )
            if remaining <= 0:
                incomplete = True
                break
        messages.reverse()
        if not messages:
            return ClassifierVerdict(outcome="insufficient_context", reason="no screenable content")
        body: dict[str, object] = {
            "messages": messages,
            "project_id": self.config.provider_project_id,
            "breakdown": True,
            "payload": False,
        }
        response = json.loads(await self._request(body))
        if not isinstance(response, dict) or not isinstance(response.get("breakdown"), list):
            raise ValueError("provider breakdown unavailable")
        matches = []
        for entry in response["breakdown"]:
            if not isinstance(entry, dict):
                raise ValueError("invalid provider breakdown")
            if (
                entry.get("detector_type") == "prompt_attack"
                and type(entry.get("message_id")) is int
                and entry.get("message_id") == len(messages) - 1
            ):
                if type(entry.get("detected")) is not bool:
                    raise ValueError("invalid provider detection outcome")
                matches.append(entry["detected"])
        if not matches:
            return ClassifierVerdict(
                outcome="insufficient_context",
                reason="provider did not screen current prompt",
                context_incomplete=True,
            )
        detected = any(matches)
        return ClassifierVerdict(
            outcome="detected" if detected else "insufficient_context" if incomplete else "clear",
            reason="provider prompt-attack detection"
            if detected
            else "no current prompt-attack detection",
            context_incomplete=incomplete,
        )
