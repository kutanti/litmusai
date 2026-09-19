"""Parse one complete JSON answer without selecting fragments from prose."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", re.DOTALL | re.IGNORECASE)


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number: {value}")


def parse_json_output(text: str) -> Any:
    """Parse finite JSON, optionally enclosed in one complete Markdown fence.

    Surrounding prose and multiple answers are invalid. Raise ``ValueError``
    on invalid input; JSON ``null`` is a valid value returned as ``None``.
    """
    source = text.strip()
    fence = _FENCE.fullmatch(source)
    if fence is not None:
        source = fence.group(1)
    try:
        value = json.loads(source, parse_constant=_reject_constant)
        json.dumps(value, allow_nan=False)  # Also reject overflow such as 1e999.
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc
    return value
