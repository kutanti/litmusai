"""Tracing for agent evaluation runs.

Captures per-step timing, tool calls, and metadata for debugging
and observability.

Usage::

    from litmusai.tracing import Tracer, Span

    tracer = Tracer()
    with tracer.span("evaluate") as s:
        result = await agent.run(task)
        s.set_attribute("tokens", result.input_tokens)

    print(tracer.to_json())
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    """A single traced operation."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_error(self, error: str) -> None:
        self.status = "error"
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
        }
        if self.attributes:
            result["attributes"] = self.attributes
        if self.error:
            result["error"] = self.error
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result


class Tracer:
    """Collects spans for an evaluation run.

    Example::

        tracer = Tracer()
        with tracer.span("agent.run") as s:
            result = await agent.run(task)
            s.set_attribute("model", "gpt-4o")
            s.set_attribute("tokens", 150)

        with tracer.span("scoring") as s:
            score = scorer.score(case, result)
            s.set_attribute("passed", score.passed)

        print(tracer.summary())
    """

    def __init__(self, name: str = "evaluation") -> None:
        self.name = name
        self.spans: list[Span] = []
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str) -> Generator[Span, None, None]:
        """Create a traced span.

        Args:
            name: Name of the operation being traced.

        Yields:
            A :class:`Span` to attach attributes to.
        """
        s = Span(name=name, start_time=time.monotonic())

        if self._stack:
            self._stack[-1].children.append(s)
        else:
            self.spans.append(s)

        self._stack.append(s)
        try:
            yield s
        except Exception as e:
            s.set_error(str(e))
            raise
        finally:
            s.end_time = time.monotonic()
            self._stack.pop()

    @property
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans)

    def summary(self) -> str:
        """Human-readable trace summary."""
        lines: list[str] = [
            f"📊 Trace: {self.name} "
            f"({self.total_duration_ms:.0f}ms total)",
        ]
        for s in self.spans:
            lines.append(_format_span(s, indent=1))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "spans": [s.to_dict() for s in self.spans],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> Path:
        """Save trace to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    def reset(self) -> None:
        """Clear all spans."""
        self.spans.clear()
        self._stack.clear()


def _format_span(span: Span, indent: int = 0) -> str:
    """Format a span as a tree line."""
    prefix = "  " * indent
    status = "✅" if span.status == "ok" else "❌"
    line = (
        f"{prefix}{status} {span.name} "
        f"({span.duration_ms:.1f}ms)"
    )
    attrs = ", ".join(
        f"{k}={v}" for k, v in span.attributes.items()
    )
    if attrs:
        line += f" [{attrs}]"
    lines = [line]
    for child in span.children:
        lines.append(_format_span(child, indent + 1))
    return "\n".join(lines)
