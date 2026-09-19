"""Operator-owned runtime configuration. No credentials are serialized here."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import Field, TypeAdapter, model_validator

from litmusai.runtime.models import (
    Category,
    Contract,
    EventType,
    Identifier,
    RuntimeEvent,
    Severity,
    ToolActivity,
)


def secret(name: str) -> str:
    """Resolve an environment reference without including values in errors."""
    value = os.environ.get(name)
    if not value:
        raise ValueError("a required runtime secret is missing")
    return value


def validate_url(value: str, allow_local_http: bool = False) -> str:
    """Require HTTPS; permit explicitly configured loopback HTTP for development."""
    url = urlsplit(value)
    local = url.hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        not url.hostname
        or url.username
        or url.password
        or url.fragment
        or url.query
        or (url.scheme != "https" and not (allow_local_http and local and url.scheme == "http"))
    ):
        raise ValueError("endpoint requires HTTPS without credentials, query, or fragment")
    return value.rstrip("/")


class ProtectedPattern(Contract):
    """A literal prefix and bounded alphabet, avoiding user-supplied regex execution."""

    name: Identifier
    prefix: str = Field(min_length=3, max_length=128)
    alphabet: str = Field(
        default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        min_length=1,
        max_length=128,
    )
    min_suffix: int = Field(default=16, ge=1, le=1024)
    max_suffix: int = Field(default=256, ge=1, le=4096)

    @model_validator(mode="after")
    def ordered(self) -> ProtectedPattern:
        """Reject empty pattern ranges."""
        if self.min_suffix > self.max_suffix:
            raise ValueError("min_suffix exceeds max_suffix")
        return self


class ThreatPolicy(Contract):
    """Immutable policy snapshot attached to each accepted event's jobs."""

    policy_id: Identifier = "default"
    version: Identifier = "1"
    allowed_tools: list[Identifier] | None = None
    allowed_destinations: list[str] | None = None
    destination_tools: list[Identifier] = Field(default_factory=list)
    severity: Severity = "high"
    protected_patterns: list[ProtectedPattern] = Field(default_factory=list, max_length=20)
    cooldown_seconds: float = Field(default=30, ge=0, le=3600)


class ToolUsagePolicy(Contract):
    """Alert when distinct captured tool requests exceed a conversation window limit."""

    policy_id: Identifier
    version: Identifier = "1"
    max_calls: int = Field(ge=1, le=10000, strict=True)
    window_seconds: int = Field(default=60, ge=1, le=3600, strict=True)
    tools: list[Identifier] = Field(default_factory=list, max_length=100)
    agents: list[Identifier] = Field(default_factory=list, max_length=100)
    deployments: list[Identifier] = Field(default_factory=list, max_length=100)
    severity: Severity = "high"
    cooldown_seconds: Literal[0] = 0

    def applies(self, event: RuntimeEvent) -> bool:
        """Only actual request boundaries in the configured scope count."""
        return (
            event.event_type == "tool.requested"
            and isinstance(event.payload, ToolActivity)
            and (not self.tools or event.payload.name in self.tools)
            and (not self.agents or event.agent_id in self.agents)
            and (not self.deployments or event.deployment_id in self.deployments)
        )


class ClassifierConfig(Contract):
    """Contract-compatible classifier endpoint; disabled unless explicitly configured."""

    provider: Literal["http", "lakera"] = "http"
    endpoint: str
    provider_project_id: str | None = Field(default=None, max_length=128)
    api_key_env: Identifier
    version: Identifier
    timeout_seconds: float = Field(default=2, gt=0, le=30)
    calls_per_minute: int = Field(default=60, ge=1, le=10000)
    context_events: int = Field(default=8, ge=0, le=50)
    context_chars: int = Field(default=16000, ge=512, le=64000)
    allow_local_http: bool = False

    @model_validator(mode="after")
    def endpoint_valid(self) -> ClassifierConfig:
        """Validate the operator's endpoint."""
        validate_url(self.endpoint, self.allow_local_http)
        if self.provider == "lakera" and not self.provider_project_id:
            raise ValueError("Lakera requires an explicitly configured provider project")
        return self


class ReviewConfig(Contract):
    """Escalate uncertain screening results and audit a deterministic sample of clears."""

    version: Identifier
    evaluator: ClassifierConfig
    on_outcomes: list[Literal["needs_review", "insufficient_context", "error"]] = Field(
        default=["needs_review", "insufficient_context"],
        max_length=3,
    )
    clear_sample_rate: float = Field(default=0.01, ge=0, le=1)


class ConversationPolicy(Contract):
    """A trusted rubric evaluated against bounded live conversation evidence."""

    policy_id: Identifier
    version: Identifier
    category: Literal[
        "sensitive_data_request",
        "business_policy",
        "suspicious_pattern",
        "abuse",
        "out_of_scope",
        "ungrounded_response",
    ]
    rubric: str = Field(min_length=1, max_length=4000)
    evaluator: ClassifierConfig
    event_types: list[EventType] = Field(min_length=1, max_length=6)
    agents: list[Identifier] = Field(default_factory=list, max_length=100)
    deployments: list[Identifier] = Field(default_factory=list, max_length=100)
    min_context_events: int = Field(default=0, ge=0, le=50)
    grounding_tools: list[Identifier] = Field(default_factory=list, max_length=100)
    severity: Severity = "high"
    cooldown_seconds: float = Field(default=30, ge=0, le=3600)
    threshold: float | None = Field(default=None, ge=0, le=1, strict=True)
    score_semantics: str | None = Field(default=None, min_length=1, max_length=500)
    score_version: Identifier | None = None

    @model_validator(mode="after")
    def evidence_contract(self) -> ConversationPolicy:
        """Reject unsupported evaluators or policies that cannot supply their evidence."""
        if self.evaluator.provider != "http":
            raise ValueError("conversation policies require the policy HTTP evaluator contract")
        if "session.ended" in self.event_types:
            raise ValueError("conversation policies evaluate activity, not session end markers")
        if self.min_context_events > self.evaluator.context_events:
            raise ValueError("required history exceeds evaluator context limit")
        if self.category == "suspicious_pattern" and self.min_context_events < 1:
            raise ValueError("pattern policies require prior conversation context")
        if self.category == "ungrounded_response" and (
            not self.grounding_tools or set(self.event_types) != {"response.completed"}
        ):
            raise ValueError("grounding policies require response events and authoritative tools")
        scored = (self.threshold, self.score_semantics, self.score_version)
        if any(v is not None for v in scored) and any(v is None for v in scored):
            raise ValueError("threshold requires score semantics and version together")
        return self

    def applies(self, event: RuntimeEvent) -> bool:
        """Cheap applicability filters run before any external evaluation is scheduled."""
        return (
            event.event_type in self.event_types
            and (not self.agents or event.agent_id in self.agents)
            and (not self.deployments or event.deployment_id in self.deployments)
        )


class ProjectConfig(Contract):
    """Project identity comes from authentication, never a model assertion."""

    project_id: Identifier
    api_key_env: Identifier
    policy: ThreatPolicy = Field(default_factory=ThreatPolicy)
    usage_policies: list[ToolUsagePolicy] = Field(default_factory=list, max_length=20)
    conversation_policies: list[ConversationPolicy] = Field(default_factory=list, max_length=20)
    classifier: ClassifierConfig | None = None
    review: ReviewConfig | None = None
    max_pending_events: int = Field(default=10000, ge=1, le=1000000)
    max_stored_events: int = Field(default=100000, ge=1, le=1000000)
    max_pending_deliveries: int = Field(default=20000, ge=1, le=1000000)

    @model_validator(mode="after")
    def policy_ids_unique(self) -> ProjectConfig:
        """Avoid ambiguous policy identities within a project."""
        ids = [
            self.policy.policy_id,
            *(p.policy_id for p in self.usage_policies),
            *(p.policy_id for p in self.conversation_policies),
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("policy IDs must be unique within a project")
        if self.review and self.classifier is None:
            raise ValueError("conditional review requires a first-stage classifier")
        return self


class DestinationBase(Contract):
    """Routing and bounded delivery budget shared by all transports."""

    destination_id: Identifier
    version: Identifier = "1"
    project_id: Identifier
    categories: list[Category] = Field(default_factory=list)
    severities: list[Severity] = Field(default_factory=list)
    agents: list[Identifier] = Field(default_factory=list)
    deployments: list[Identifier] = Field(default_factory=list)
    timeout_seconds: float = Field(default=5, gt=0, le=30)
    max_attempts: int = Field(default=6, ge=1, le=20)


class WebhookDestination(DestinationBase):
    """Signed HTTPS webhook destination."""

    kind: Literal["webhook"] = "webhook"
    url: str
    signing_secret_env: Identifier
    bearer_token_env: Identifier | None = None
    allow_local_http: bool = False

    @model_validator(mode="after")
    def endpoint_valid(self) -> WebhookDestination:
        """Restrict cleartext transport to explicit loopback development."""
        validate_url(self.url, self.allow_local_http)
        return self


class KafkaDestination(DestinationBase):
    """Kafka producer configuration with TLS enabled by default."""

    kind: Literal["kafka"] = "kafka"
    bootstrap_servers: str
    topic: str = Field(min_length=1, max_length=249, pattern=r"^[\w.-]+$")
    security_protocol: Literal["SSL", "SASL_SSL", "PLAINTEXT"] = "SSL"
    sasl_mechanism: Literal["PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"] = "PLAIN"
    username_env: Identifier | None = None
    password_env: Identifier | None = None
    ca_location: str | None = None
    certificate_location: str | None = None
    key_location: str | None = None
    key_password_env: Identifier | None = None
    allow_insecure_development: bool = False

    @model_validator(mode="after")
    def credentials_valid(self) -> KafkaDestination:
        """Require complete credentials and explicit opt-in for development plaintext."""
        if self.security_protocol == "PLAINTEXT" and not self.allow_insecure_development:
            raise ValueError("plaintext Kafka requires allow_insecure_development")
        if self.security_protocol == "SASL_SSL" and not (self.username_env and self.password_env):
            raise ValueError("SASL_SSL requires username and password references")
        return self


class EventGridDestination(DestinationBase):
    """Azure CloudEvents-compatible topic, using Entra or an access-key reference."""

    kind: Literal["event_grid"] = "event_grid"
    endpoint: str
    access_key_env: Identifier | None = None
    managed_identity_client_id: str | None = None

    @model_validator(mode="after")
    def endpoint_valid(self) -> EventGridDestination:
        """Require encrypted Azure publishing."""
        validate_url(self.endpoint)
        return self


Destination = Annotated[
    WebhookDestination | KafkaDestination | EventGridDestination, Field(discriminator="kind")
]
DESTINATION_ADAPTER: TypeAdapter[Destination] = TypeAdapter(Destination)


class RuntimeConfig(Contract):
    """Single-instance pilot settings loaded from trusted local YAML."""

    database: str = ".litmus/runtime.sqlite3"
    projects: list[ProjectConfig] = Field(min_length=1, max_length=100)
    destinations: list[Destination] = Field(default_factory=list, max_length=100)
    max_request_bytes: int = Field(default=262144, ge=1024, le=1048576)
    max_event_bytes: int = Field(default=65536, ge=512, le=262144)
    batch_size: int = Field(default=50, ge=1, le=100)
    retention_days: int = Field(default=7, ge=1, le=365)
    poll_seconds: float = Field(default=0.05, ge=0.01, le=5)

    @model_validator(mode="after")
    def references_valid(self) -> RuntimeConfig:
        """Reject ambiguous authentication or delivery identities."""
        ids = [p.project_id for p in self.projects]
        keys = [p.api_key_env for p in self.projects]
        destinations = [d.destination_id for d in self.destinations]
        if len(set(ids)) != len(ids) or len(set(keys)) != len(keys):
            raise ValueError("projects and credential references must be unique")
        if len(set(destinations)) != len(destinations):
            raise ValueError("destination IDs must be unique")
        if any(d.project_id not in ids for d in self.destinations):
            raise ValueError("destination references an unknown project")
        return self


def load_config(path: str | Path) -> RuntimeConfig:
    """Read and validate operator-owned configuration without resolving credentials."""
    return RuntimeConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
