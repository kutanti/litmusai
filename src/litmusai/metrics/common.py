"""Prediction parsing and count arithmetic shared by task metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from litmusai._json import parse_json_output
from litmusai.metrics.schema import MetricConfig, Observation


def ratio(numerator: int | float, denominator: int | float) -> dict[str, Any]:
    """Return a finite ratio and flag a zero denominator explicitly."""
    return {
        "value": numerator / denominator if denominator else 0.0,
        "defined": denominator != 0,
        "numerator": numerator,
        "denominator": denominator,
    }


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """Calculate precision, recall and F1 directly from integer counts."""
    return {
        "tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def coverage(observations: Sequence[Observation]) -> dict[str, Any]:
    """Include every attempted prediction in coverage and error counts."""
    statuses = Counter(o.status for o in observations)
    return {
        "total": len(observations),
        "valid_predictions": statuses["ok"],
        "invalid_predictions": statuses["invalid_prediction"],
        "execution_errors": statuses["execution_error"],
        "prediction_coverage": ratio(statuses["ok"], len(observations)),
    }


def validate_observations(observations: Sequence[Observation], config: MetricConfig) -> None:
    """Reject mixed tasks and repeated identities instead of double-counting."""
    seen: set[tuple[str, int, str]] = set()
    for observation in observations:
        if observation.task_type != config.task_type:
            raise ValueError("observation task_type does not match metric configuration")
        identity = (observation.evaluation_id, observation.repetition, observation.case_id)
        if identity in seen:
            raise ValueError(f"duplicate observation identity: {identity}")
        seen.add(identity)


def _read_pointer(value: Any, pointer: str) -> Any:
    if not pointer:
        return value
    for part in pointer[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and (
            key == "0" or (key.isascii() and key.isdigit() and not key.startswith("0"))
        ) and int(key) < len(value):
            value = value[int(key)]
        else:
            raise ValueError(f"prediction_field {pointer!r} is missing")
    return value


def parse_observation(
    expected: Any, output: str, *, config: MetricConfig, case_id: str,
    evaluation_id: str, repetition: int = 1, success: bool = True, error: str | None = None,
) -> Observation:
    """Read a full response; retain invalid JSON and execution errors as evidence."""
    observation = Observation(
        case_id=case_id, evaluation_id=evaluation_id, repetition=repetition,
        task_type=config.task_type, expected=deepcopy(expected), predicted=output,
    )
    if not success:
        observation.status = "execution_error"
        observation.error = error or "agent execution failed"
        return observation
    if config.task_type == "extraction" or config.prediction_field is not None:
        try:
            value = parse_json_output(output)
            observation.predicted = _read_pointer(value, config.prediction_field or "")
        except (ValueError, RecursionError) as exc:
            observation.status = "invalid_prediction"
            observation.error = str(exc)
    return observation
