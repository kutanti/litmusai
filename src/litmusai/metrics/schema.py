"""Versioned contracts for labeled tasks and individual predictions."""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Literal["1.0"] = "1.0"


def _validate_json_value(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("observation values must be finite JSON data")
    elif isinstance(value, list):
        for item in value:
            _validate_json_value(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("observation object keys must be strings")
            _validate_json_value(item)
    else:
        raise ValueError("observation values must be JSON scalars, lists, or string-keyed objects")


class MetricConfig(BaseModel):
    """Suite-wide matching rules; prediction_field is an optional JSON Pointer.

    Classification without a pointer reads the literal response as a label.
    Extraction always reads JSON. An empty pointer selects the JSON root.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    task_type: Literal["classification", "extraction"]
    labels: list[str] = Field(default_factory=list)
    prediction_field: str | None = None
    normalize_whitespace: bool = False
    casefold: bool = False

    @model_validator(mode="after")
    def validate_rules(self) -> MetricConfig:
        """Reject ambiguous labels and unsupported matching options."""
        if self.task_type == "classification":
            if not self.labels or any(not label for label in self.labels):
                raise ValueError("classification requires nonempty string labels")
            if len(set(self.labels)) != len(self.labels):
                raise ValueError("classification labels must be unique")
            if self.normalize_whitespace or self.casefold:
                raise ValueError("normalization options apply only to extraction")
        elif self.labels:
            raise ValueError("labels apply only to classification")
        pointer = self.prediction_field
        if pointer is not None and (
            (pointer and not pointer.startswith("/")) or re.search(r"~(?![01])", pointer)
        ):
            raise ValueError("prediction_field must be a JSON Pointer, e.g. /output/label")
        return self


class Observation(BaseModel):
    """One full prediction and its evidence, independent of display truncation.

    The (evaluation_id, repetition, case_id) tuple identifies an observation.
    Expected/predicted values and evidence must contain JSON-compatible values.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    evaluation_id: str = Field(min_length=1)
    repetition: int = Field(default=1, ge=1)
    case_id: str = Field(min_length=1)
    task_type: Literal["classification", "extraction"]
    expected: Any
    predicted: Any = None
    status: Literal["ok", "invalid_prediction", "execution_error"] = "ok"
    error: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_json(self) -> Observation:
        """Reject values that change type or lose keys during a JSON round trip."""
        try:
            _validate_json_value([self.expected, self.predicted, self.evidence])
        except RecursionError as exc:
            raise ValueError("observation values must be acyclic JSON data") from exc
        return self
