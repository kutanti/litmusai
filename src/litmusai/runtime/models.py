"""Versioned contracts for observing live agent activity (not executing agents)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[\w.:-]+$")]
Category = Literal[
    "unauthorized_tool",
    "forbidden_destination",
    "sensitive_data",
    "prompt_injection",
    "excessive_tool_usage",
]
Severity = Literal["low", "medium", "high", "critical"]
Outcome = Literal["clear", "detected", "insufficient_context", "needs_review", "error", "skipped"]
Stage = Literal["attempt", "requested", "observed"]


def utcnow() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


def new_id() -> str:
    """Generate a transport-independent identifier."""
    return str(uuid4())


class Contract(BaseModel):
    """Reject unknown fields and accidental mutation of contract fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Message(Contract):
    """A captured complete message, context document, or agent response."""

    kind: Literal["message"] = "message"
    text: str = Field(max_length=32768)
    role: Literal["user", "assistant", "system", "tool", "context"] = "user"
    destination: str | None = Field(default=None, max_length=512)


class ToolActivity(Contract):
    """An actual tool boundary; completion does not by itself prove exfiltration."""

    kind: Literal["tool"] = "tool"
    name: Identifier
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    destination: str | None = Field(default=None, max_length=512)
    result: JsonValue = None
    error_type: Identifier | None = None


class SessionEnd(Contract):
    """An optional end marker; detection never waits for this event."""

    kind: Literal["session_end"] = "session_end"


Payload = Annotated[Message | ToolActivity | SessionEnd, Field(discriminator="kind")]
EventType = Literal[
    "message.received",
    "context.received",
    "tool.requested",
    "tool.completed",
    "tool.failed",
    "response.completed",
    "session.ended",
]


class RuntimeEvent(Contract):
    """Client event. Project must match the credential used for ingestion."""

    schema_version: Literal["1.0"] = "1.0"
    event_id: Identifier = Field(default_factory=new_id)
    project_id: Identifier
    agent_id: Identifier
    deployment_id: Identifier = "default"
    session_id: Identifier
    actor_id: Identifier | None = None
    producer_id: Identifier
    sequence: int = Field(ge=1)
    event_type: EventType
    observed_at: datetime = Field(default_factory=utcnow)
    parent_event_id: Identifier | None = None
    tool_call_id: Identifier | None = None
    source_trust: Literal["untrusted", "application"] = "untrusted"
    payload: Payload

    @field_validator("observed_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        """Require timezone information instead of guessing the client's timezone."""
        if value.tzinfo is None:
            raise ValueError("observed_at requires a timezone")
        return value

    def validate_boundary(self) -> None:
        """Check consistency between event type and payload at the capture boundary."""
        expected = (
            "tool"
            if self.event_type.startswith("tool.")
            else "session_end"
            if self.event_type == "session.ended"
            else "message"
        )
        if self.payload.kind != expected:
            raise ValueError("event type and payload kind disagree")
        if expected == "tool" and not self.tool_call_id:
            raise ValueError("tool events require tool_call_id")


class SensitiveSignal(Contract):
    """A locally detected data type, never the protected value itself."""

    data_type: str
    location: str


class CapturedEvent(Contract):
    """Sanitized durable envelope created by the authenticated collector."""

    event: RuntimeEvent
    received_at: datetime = Field(default_factory=utcnow)
    signals: list[SensitiveSignal] = Field(default_factory=list)
    capture_gap: bool = False


class ToolUsageEvidence(Contract):
    """An observed count, not an estimate of malicious intent or risk probability."""

    observed_count: int = Field(ge=1)
    limit: int = Field(ge=1)
    window_seconds: int = Field(ge=1)
    window_start: datetime
    window_end: datetime
    counting_basis: Literal["collector_received_at"] = "collector_received_at"
    threshold_crossed: bool
    source_events_truncated: bool = False


class EvaluationTrace(Contract):
    """Selection and measured provider telemetry; absent cost is not zero cost."""

    stage: Literal["screen", "review"]
    decision: str
    screen_version: str
    review_version: str
    gate_version: str
    provider_called: bool = False
    elapsed_ms: float = Field(default=0, ge=0)
    reported_cost_usd: float | None = Field(default=None, ge=0, strict=True)
    input_tokens: int | None = Field(default=None, ge=0, strict=True)
    output_tokens: int | None = Field(default=None, ge=0, strict=True)


class DetectionResult(Contract):
    """A detector outcome, kept separate from delivery and processing state."""

    event_id: str
    detector: str
    detector_version: str = "1"
    policy_id: str
    policy_version: str
    outcome: Outcome
    category: Category | None = None
    severity: Severity = "high"
    stage: Stage = "attempt"
    reason: str
    evidence: list[str] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    context_incomplete: bool = False
    usage: ToolUsageEvidence | None = None
    evaluation: EvaluationTrace | None = None


class ThreatAlert(Contract):
    """An immutable revision of a logical threat episode."""

    schema_version: Literal["1.0", "1.1", "1.2"] = "1.0"
    alert_id: str
    revision: int = Field(ge=1)
    project_id: str
    agent_id: str
    deployment_id: str
    session_id: str
    actor_id: str | None
    category: Category
    severity: Severity
    stage: Stage
    evidence: list[str]
    reason: str
    source_event_ids: list[str]
    policy_id: str
    policy_version: str
    detector: str
    detector_version: str
    observed_at: datetime
    detected_at: datetime = Field(default_factory=utcnow)
    context_incomplete: bool = False
    usage: ToolUsageEvidence | None = None
    evaluation: EvaluationTrace | None = None


class CloudEvent(Contract):
    """Structured CloudEvents 1.0 envelope shared by every publisher."""

    specversion: Literal["1.0"] = "1.0"
    id: str = Field(default_factory=new_id)
    source: str
    type: Literal["com.litmusai.threat.detected", "com.litmusai.threat.updated"]
    subject: str
    time: datetime = Field(default_factory=utcnow)
    datacontenttype: Literal["application/json"] = "application/json"
    data: ThreatAlert
