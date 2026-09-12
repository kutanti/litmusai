"""Dataset identity and source references, without a provider SDK dependency."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from litmusai.metrics.schema import _validate_json_value

_T = TypeVar("_T")


def snapshot_json(value: _T, context: str) -> _T:
    """Copy JSON data without coercing keys, nonfinite numbers, or Python objects."""
    try:
        _validate_json_value(value)
        return deepcopy(value)
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"{context} must contain finite, acyclic JSON data: {exc}") from exc


class SourceReference(BaseModel):
    """Original provider identities; trace_id is never an evaluation run ID."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    provider: str
    dataset_id: str | None = None
    example_id: str | None = None
    trace_id: str | None = None

    @field_validator("provider", "dataset_id", "example_id", "trace_id")
    @classmethod
    def nonempty_identity(cls, value: str | None) -> str | None:
        """Preserve provider IDs verbatim and reject blank supplied identities."""
        if value is not None and not value.strip():
            raise ValueError("source identities must contain non-whitespace characters")
        return value


class DatasetInfo(BaseModel):
    """Dataset identity, external revision, and a fingerprint of local content.

    Missing external identities stay None. TestSuite computes the fingerprint
    from its input records and labels; it does not infer a provider revision.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str | None = None
    revision: str | None = None
    fingerprint: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: SourceReference | None = None

    @field_validator("id", "revision")
    @classmethod
    def nonempty_identity(cls, value: str | None) -> str | None:
        """Reject blank IDs/revisions without normalizing externally assigned IDs."""
        if value is not None and not value.strip():
            raise ValueError("dataset identities must contain non-whitespace characters")
        return value

    @field_validator("metadata")
    @classmethod
    def json_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Require metadata that survives JSON serialization unchanged."""
        return snapshot_json(value, "dataset metadata")
