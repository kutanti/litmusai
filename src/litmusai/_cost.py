"""Cost arithmetic and formatting that preserve unavailable estimates."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def read_cost(value: Any) -> float | None:
    """Read a nonnegative finite USD estimate; missing or invalid data is unknown."""
    if value is None or isinstance(value, bool):
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def sum_costs(values: Iterable[float | None]) -> float | None:
    """Sum complete costs, returning None when any estimate is unavailable."""
    total = 0.0
    for value in values:
        cost = read_cost(value)
        if cost is None:
            return None
        total += cost
    return read_cost(total)


def format_cost(value: float | None) -> str:
    """Display a USD estimate or an explicit unknown label."""
    cost = read_cost(value)
    return "Unknown" if cost is None else f"${cost:.4f}"


def round_cost(value: float | None) -> float | None:
    """Round an available estimate for result serialization."""
    cost = read_cost(value)
    return round(cost, 6) if cost is not None else None


def estimate_cost(usage: dict[str, Any], *models: str) -> float | None:
    """Estimate USD only with registered pricing and complete split token usage."""
    from litmusai.benchmarks import get_pricing

    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if (type(input_tokens) is not int or input_tokens < 0
            or type(output_tokens) is not int or output_tokens < 0):
        return None
    for model in models:
        pricing = get_pricing(model) if model else None
        if pricing is not None:
            return read_cost(
                input_tokens * pricing.input_cost_per_token
                + output_tokens * pricing.output_cost_per_token
            )
    return None
