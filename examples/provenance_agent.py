"""Local agent for the structured-input dataset example."""

from typing import Any


def route(task: str, *, inputs: dict[str, Any]) -> str:
    """Route by the text field while retaining the complete input in saved results."""
    return "billing" if "refund" in inputs["text"].lower() else "technical"
