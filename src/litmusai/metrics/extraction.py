"""One-to-one matching for extracted fields and entity occurrences."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Sequence
from typing import Any

from litmusai.metrics.common import coverage, parse_observation, prf, ratio, validate_observations
from litmusai.metrics.schema import MetricConfig, Observation


def validate_extraction_truth(expected: Any, config: MetricConfig) -> None:
    """Require an entity list or field mapping made of finite JSON values."""
    if config.task_type != "extraction":
        raise ValueError("extraction metrics require task_type='extraction'")
    if not isinstance(expected, (list, dict)):
        raise ValueError("extraction ground truth must be an entity list or field mapping")
    _validate_value(expected)


def _validate_value(value: Any) -> None:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("extraction field names must be strings")
        for child in value.values():
            _validate_value(child)
    elif isinstance(value, list):
        for child in value:
            _validate_value(child)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("extraction values must be JSON data")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("extraction values must be finite JSON data") from exc


def _normalize(value: Any, config: MetricConfig) -> Any:
    if isinstance(value, str):
        if config.normalize_whitespace:
            value = " ".join(value.split())
        if config.casefold:
            value = value.casefold()
    elif isinstance(value, list):
        value = [_normalize(child, config) for child in value]
    elif isinstance(value, dict):
        value = {key: _normalize(child, config) for key, child in value.items()}
    return value


def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    return [{"field": key, "value": item} for key, child in value.items()
            for item in (child if isinstance(child, list) else [child])]


def _key(item: Any, fields: bool, config: MetricConfig) -> str:
    normalized = ({"field": item["field"], "value": _normalize(item["value"], config)}
                  if fields else _normalize(item, config))
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _match(observation: Observation, config: MetricConfig) -> dict[str, Any]:
    expected = _items(observation.expected)
    predicted = _items(observation.predicted) if observation.status == "ok" else []
    fields = isinstance(observation.expected, dict)
    available: dict[str, deque[int]] = defaultdict(deque)
    for i, item in enumerate(predicted):
        available[_key(item, fields, config)].append(i)
    matched: list[dict[str, Any]] = []
    missing: list[Any] = []
    used: set[int] = set()
    for item in expected:
        candidates = available[_key(item, fields, config)]
        if candidates:
            index = candidates.popleft()
            used.add(index)
            matched.append({"expected": item, "predicted": predicted[index]})
        else:
            missing.append(item)
    extra = [item for i, item in enumerate(predicted) if i not in used]
    return {"matched": matched, "missing": missing, "extra": extra,
            "tp": len(matched), "fp": len(extra), "fn": len(missing)}


def extraction_observation(
    expected: Any, output: str, *, config: MetricConfig, case_id: str,
    evaluation_id: str, repetition: int = 1, success: bool = True, error: str | None = None,
) -> Observation:
    """Match one full JSON response and retain original matched/missing/extra values."""
    validate_extraction_truth(expected, config)
    observation = parse_observation(
        expected, output, config=config, case_id=case_id, evaluation_id=evaluation_id,
        repetition=repetition, success=success, error=error,
    )
    if observation.status == "ok" and type(observation.predicted) is not type(expected):
        observation.status = "invalid_prediction"
        observation.error = f"expected a JSON {type(expected).__name__} prediction"
    observation.evidence = _match(observation, config)
    return observation


def extraction_metrics(
    observations: Sequence[Observation], config: MetricConfig,
) -> dict[str, Any] | None:
    """Pool occurrence counts before computing extraction precision, recall, and F1."""
    if config.task_type != "extraction":
        raise ValueError("extraction metrics require task_type='extraction'")
    validate_observations(observations, config)
    if not observations:
        return None
    tp = fp = fn = exact = 0
    for observation in observations:
        validate_extraction_truth(observation.expected, config)
        if observation.status == "ok":
            if type(observation.predicted) is not type(observation.expected):
                raise ValueError("an ok extraction prediction must have the ground-truth shape")
            _validate_value(observation.predicted)
        evidence = _match(observation, config)
        tp += evidence["tp"]
        fp += evidence["fp"]
        fn += evidence["fn"]
        exact += int(observation.status == "ok" and evidence["fp"] == evidence["fn"] == 0)
    return {"task_type": "extraction", **coverage(observations), **prf(tp, fp, fn),
            "exact_match_accuracy": ratio(exact, len(observations))}
