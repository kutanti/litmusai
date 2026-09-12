"""Deterministic metrics for labeled evaluation datasets."""

from collections.abc import Sequence
from typing import Any

from litmusai.metrics.classification import classification_metrics, classification_observation
from litmusai.metrics.extraction import extraction_metrics, extraction_observation
from litmusai.metrics.schema import MetricConfig, Observation

__all__ = [
    "MetricConfig", "Observation", "classification_metrics", "classification_observation",
    "extraction_metrics", "extraction_observation",
    "aggregate_metrics",
]


def aggregate_metrics(
    observations: Sequence[Observation], config: MetricConfig,
) -> dict[str, Any] | None:
    """Compute the configured task metrics from all case/repetition records."""
    if config.task_type == "classification":
        return classification_metrics(observations, config)
    return extraction_metrics(observations, config)
