"""Single-label classification with explicit labels and invalid-prediction counts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from litmusai.metrics.common import coverage, parse_observation, prf, ratio, validate_observations
from litmusai.metrics.schema import MetricConfig, Observation


def validate_classification_truth(expected: Any, config: MetricConfig) -> None:
    """Require ground truth to be one of the declared string labels."""
    if config.task_type != "classification":
        raise ValueError("classification metrics require task_type='classification'")
    if not isinstance(expected, str) or expected not in config.labels:
        raise ValueError(f"expected label {expected!r} must be one of {config.labels!r}")


def classification_observation(
    expected: Any, output: str, *, config: MetricConfig, case_id: str,
    evaluation_id: str, repetition: int = 1, success: bool = True, error: str | None = None,
) -> Observation:
    """Score one response without allowing invalid predictions to disappear."""
    validate_classification_truth(expected, config)
    observation = parse_observation(
        expected, output, config=config, case_id=case_id, evaluation_id=evaluation_id,
        repetition=repetition, success=success, error=error,
    )
    predicted = observation.predicted
    if observation.status == "ok" and (
        not isinstance(predicted, str) or predicted not in config.labels
    ):
        observation.status = "invalid_prediction"
        observation.error = f"predicted label {predicted!r} is not in the declared labels"
    valid = observation.status == "ok"
    correct = valid and predicted == expected
    observation.evidence = {
        "expected_label": expected,
        "predicted_label": predicted if valid else None,
        "correct": correct,
        "tp": int(correct), "fp": int(valid and not correct), "fn": int(not correct),
    }
    return observation


def classification_metrics(
    observations: Sequence[Observation], config: MetricConfig,
) -> dict[str, Any] | None:
    """Recompute counts across all observations, then calculate each average.

    Invalid predictions contribute a false negative for their expected class.
    The final confusion-matrix column (label null) holds invalid predictions.
    """
    if config.task_type != "classification":
        raise ValueError("classification metrics require task_type='classification'")
    validate_observations(observations, config)
    if not observations:
        return None
    labels = config.labels
    positions = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * (len(labels) + 1) for _ in labels]
    for observation in observations:
        validate_classification_truth(observation.expected, config)
        predicted = observation.predicted
        if observation.status == "ok":
            if not isinstance(predicted, str) or predicted not in positions:
                raise ValueError("an ok observation must contain a declared predicted label")
            column = positions[predicted]
        else:
            column = len(labels)
        matrix[positions[observation.expected]][column] += 1

    per_class: dict[str, Any] = {}
    for label, i in positions.items():
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        fn = sum(matrix[i]) - tp
        per_class[label] = prf(tp, fp, fn)
    counts = {key: sum(c[key] for c in per_class.values()) for key in ("tp", "fp", "fn")}

    def average(weighted: bool) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for metric in ("precision", "recall", "f1"):
            weights = {label: c["support"] if weighted else 1 for label, c in per_class.items()}
            value = ratio(
                sum(c[metric]["value"] * weights[label] for label, c in per_class.items()),
                sum(weights.values()),
            )
            undefined = [label for label, c in per_class.items()
                         if weights[label] and not c[metric]["defined"]]
            value["defined"] = value["defined"] and not undefined
            value["undefined_labels"] = undefined
            result[metric] = value
        return result

    return {
        "task_type": "classification", **coverage(observations),
        "accuracy": ratio(counts["tp"], len(observations)),
        "per_class": per_class,
        "micro": prf(**counts), "macro": average(False), "weighted": average(True),
        "confusion_matrix": {
            "expected_labels": labels, "predicted_labels": [*labels, None], "counts": matrix,
        },
    }
