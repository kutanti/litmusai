"""Assertion engine — multi-strategy scoring for agent evaluation.

Assertions are composable checks that evaluate agent responses.
Instead of simple substring matching, combine multiple strategies
for robust, credible scoring.

Built-in assertion types:
    - **Exact**: Exact string match
    - **Contains**: Substring check (any/all modes)
    - **NotContains**: Must NOT include patterns
    - **Regex**: Regular expression matching
    - **Numeric**: Number extraction + tolerance
    - **JsonSchema**: Validate JSON against a schema
    - **JsonPath**: Check specific JSON values
    - **Semantic**: Embedding cosine similarity (requires API)
    - **LLMGrade**: LLM judges response against criteria
    - **Custom**: User-defined function

Composite assertions:
    - **All**: Every assertion must pass
    - **Any**: At least one must pass
    - **AtLeast**: N of M must pass
    - **Weighted**: Weighted score (no hard pass/fail)

Example:
    >>> from litmusai.assertions import (
    ...     Numeric, NotContains, LLMGrade, All
    ... )
    >>> assertion = All(
    ...     Numeric(36, tolerance=0.01),
    ...     NotContains(["I can't", "I don't know"]),
    ... )
    >>> result = assertion.check("The answer is 36.")
    >>> print(result.passed, result.score)
    True 1.0
"""

from __future__ import annotations

import json as _json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ─── Result ──────────────────────────────────────────────────────


@dataclass
class AssertionResult:
    """Result of evaluating a single assertion.

    Attributes:
        passed: Whether the assertion passed.
        score: Confidence score (0.0–1.0).
        reason: Human-readable explanation.
        assertion_type: Name of the assertion class.
        details: Optional structured details.
    """

    passed: bool
    score: float
    reason: str
    assertion_type: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        icon = "✅" if self.passed else "❌"
        return f"{icon} {self.assertion_type}: {self.reason} ({self.score:.2f})"


# ─── Base ────────────────────────────────────────────────────────


class Assertion(ABC):
    """Base class for all assertions.

    Subclasses implement :meth:`check` to evaluate a response string.
    Each assertion has a ``weight`` used by :class:`Weighted` composites.
    """

    weight: float = 1.0

    @abstractmethod
    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        """Evaluate a response.

        Args:
            response: The agent's text output.
            context: Optional context (task, expected, metadata).

        Returns:
            An :class:`AssertionResult` with pass/fail, score, reason.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ─── Async-capable base ─────────────────────────────────────────


class AsyncAssertion(Assertion):
    """Base for assertions that require async calls (LLM, embeddings).

    Subclasses implement :meth:`acheck`. The sync :meth:`check` raises
    ``RuntimeError`` — use :meth:`acheck` or run within an event loop.
    """

    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        """Sync check — raises RuntimeError. Use acheck() instead."""
        msg = (
            f"{type(self).__name__} requires async. "
            f"Use `await assertion.acheck(response)` instead."
        )
        raise RuntimeError(msg)

    @abstractmethod
    async def acheck(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        """Async evaluation of the response."""
        ...


# ─── String Assertions ──────────────────────────────────────────


class Exact(Assertion):
    """Exact string match.

    Args:
        expected: The expected string.
        case_sensitive: Whether comparison is case-sensitive.
        strip: Whether to strip whitespace before comparing.

    Example:
        >>> Exact("Paris").check("Paris")
        ✅ Exact: Matched 'Paris' (1.00)
        >>> Exact("Paris").check("paris is great")
        ❌ Exact: Expected 'Paris', got 'paris is great' (0.00)
    """

    def __init__(
        self,
        expected: str,
        *,
        case_sensitive: bool = False,
        strip: bool = True,
    ):
        self.expected = expected
        self.case_sensitive = case_sensitive
        self.strip = strip

    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        resp = response.strip() if self.strip else response
        exp = self.expected.strip() if self.strip else self.expected

        if not self.case_sensitive:
            match = resp.lower() == exp.lower()
        else:
            match = resp == exp

        if match:
            return AssertionResult(
                passed=True, score=1.0,
                reason=f"Matched '{self.expected}'",
                assertion_type="Exact",
            )
        # Truncate for readability
        got = resp[:60] + "..." if len(resp) > 60 else resp
        return AssertionResult(
            passed=False, score=0.0,
            reason=f"Expected '{self.expected}', got '{got}'",
            assertion_type="Exact",
        )

    def __repr__(self) -> str:
        return f"Exact('{self.expected}')"


class Contains(Assertion):
    """Substring check with any/all modes.

    Args:
        patterns: List of substrings to look for.
        mode: ``"any"`` = at least one pattern found;
              ``"all"`` = every pattern must be found.
        case_sensitive: Whether matching is case-sensitive.

    Example:
        >>> Contains(["36", "thirty-six"], mode="any").check("It's 36.")
        ✅ Contains: Found 1/2 patterns (any mode) (1.00)
    """

    def __init__(
        self,
        patterns: list[str],
        *,
        mode: str = "any",
        case_sensitive: bool = False,
    ):
        if mode not in ("any", "all"):
            msg = f"mode must be 'any' or 'all', got '{mode}'"
            raise ValueError(msg)
        self.patterns = patterns
        self.mode = mode
        self.case_sensitive = case_sensitive

    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        resp = response if self.case_sensitive else response.lower()
        found = []
        missing = []

        for p in self.patterns:
            target = p if self.case_sensitive else p.lower()
            if target in resp:
                found.append(p)
            else:
                missing.append(p)

        total = len(self.patterns)
        n_found = len(found)

        if self.mode == "any":
            passed = n_found > 0
            score = min(1.0, n_found / max(total, 1))
        else:  # all
            passed = n_found == total
            score = n_found / max(total, 1)

        if passed:
            reason = f"Found {n_found}/{total} patterns ({self.mode} mode)"
        else:
            reason = f"Missing: {missing}"

        return AssertionResult(
            passed=passed, score=score, reason=reason,
            assertion_type="Contains",
            details={"found": found, "missing": missing},
        )

    def __repr__(self) -> str:
        return f"Contains({self.patterns}, mode='{self.mode}')"


class NotContains(Assertion):
    """Must NOT include any of the given patterns.

    Useful for safety checks — ensuring the agent doesn't reveal
    harmful information.

    Args:
        patterns: Patterns that must NOT appear.
        case_sensitive: Whether matching is case-sensitive.

    Example:
        >>> NotContains(["hack", "exploit"]).check("I can't help with that.")
        ✅ NotContains: None of 2 forbidden patterns found (1.00)
    """

    def __init__(
        self,
        patterns: list[str],
        *,
        case_sensitive: bool = False,
    ):
        self.patterns = patterns
        self.case_sensitive = case_sensitive

    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        resp = response if self.case_sensitive else response.lower()
        violations = []

        for p in self.patterns:
            target = p if self.case_sensitive else p.lower()
            if target in resp:
                violations.append(p)

        passed = len(violations) == 0
        score = 1.0 - (len(violations) / max(len(self.patterns), 1))

        if passed:
            reason = f"None of {len(self.patterns)} forbidden patterns found"
        else:
            reason = f"Found forbidden: {violations}"

        return AssertionResult(
            passed=passed, score=max(0.0, score), reason=reason,
            assertion_type="NotContains",
            details={"violations": violations},
        )

    def __repr__(self) -> str:
        return f"NotContains({self.patterns})"


class RegexMatch(Assertion):
    """Regular expression matching.

    Args:
        pattern: Regex pattern string.
        flags: Regex flags (default: ``re.IGNORECASE``).
        full_match: If True, the entire response must match.

    Example:
        >>> RegexMatch(r"\\b36\\.?0*\\b").check("The answer is 36.")
        ✅ RegexMatch: Pattern matched (1.00)
    """

    def __init__(
        self,
        pattern: str,
        *,
        flags: int = re.IGNORECASE,
        full_match: bool = False,
    ):
        self.pattern = pattern
        self.flags = flags
        self.full_match = full_match
        self._compiled = re.compile(pattern, flags)

    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        if self.full_match:
            m = self._compiled.fullmatch(response.strip())
        else:
            m = self._compiled.search(response)

        if m:
            return AssertionResult(
                passed=True, score=1.0,
                reason=f"Pattern matched: '{m.group()}'",
                assertion_type="RegexMatch",
                details={"match": m.group(), "span": list(m.span())},
            )

        return AssertionResult(
            passed=False, score=0.0,
            reason=f"Pattern /{self.pattern}/ not found",
            assertion_type="RegexMatch",
        )

    def __repr__(self) -> str:
        return f"RegexMatch(r'{self.pattern}')"


# ─── Numeric Assertion ───────────────────────────────────────────


# Pattern to extract numbers (integers, decimals, negatives)
_NUM_RE = re.compile(
    r"-?\d+(?:,\d{3})*(?:\.\d+)?",
)

# Words to numbers for simple cases
_WORD_NUMS: dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1_000_000, "billion": 1_000_000_000,
}


def _extract_numbers(text: str) -> list[float]:
    """Extract numeric values from text."""
    numbers: list[float] = []

    # Try regex first (handles "36", "3.14", "1,000", "-5")
    for m in _NUM_RE.finditer(text):
        raw = m.group().replace(",", "")
        try:
            numbers.append(float(raw))
        except ValueError:
            pass

    # Try common word-numbers ("thirty-six", "forty two")
    lower = text.lower()
    for word, val in _WORD_NUMS.items():
        if word in lower:
            numbers.append(val)

    # Handle compound: "thirty-six" → 36
    compound_re = re.compile(
        r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
        r"[- ]?(one|two|three|four|five|six|seven|eight|nine)\b",
        re.IGNORECASE,
    )
    for m in compound_re.finditer(lower):
        tens = _WORD_NUMS.get(m.group(1).lower(), 0)
        ones = _WORD_NUMS.get(m.group(2).lower(), 0)
        numbers.append(tens + ones)

    return numbers


class Numeric(Assertion):
    """Numeric answer check with tolerance.

    Extracts numbers from the response text and checks if any
    match the expected value within the given tolerance.

    Args:
        expected: The expected numeric answer.
        tolerance: Acceptable absolute difference (default 0.01).
        relative_tolerance: If set, tolerance as a fraction of expected.

    Example:
        >>> Numeric(36, tolerance=0.1).check("The answer is 36.")
        ✅ Numeric: Found 36.0, expected 36 (±0.1) (1.00)
        >>> Numeric(36).check("About thirty-six")
        ✅ Numeric: Found 36.0, expected 36 (±0.01) (1.00)
    """

    def __init__(
        self,
        expected: float,
        *,
        tolerance: float = 0.01,
        relative_tolerance: float | None = None,
    ):
        self.expected = float(expected)
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance

    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        numbers = _extract_numbers(response)

        if not numbers:
            return AssertionResult(
                passed=False, score=0.0,
                reason="No numbers found in response",
                assertion_type="Numeric",
            )

        tol = self.tolerance
        if self.relative_tolerance is not None:
            tol = abs(self.expected * self.relative_tolerance)

        # Find the closest match
        closest = min(numbers, key=lambda n: abs(n - self.expected))
        diff = abs(closest - self.expected)

        if diff <= tol:
            # Score: 1.0 at exact match, decreasing toward tolerance
            if tol > 0:
                score = 1.0 - (diff / tol) * 0.2  # slight penalty
            else:
                score = 1.0 if diff == 0 else 0.0

            return AssertionResult(
                passed=True, score=min(1.0, score),
                reason=(
                    f"Found {closest}, expected {self.expected} "
                    f"(±{tol})"
                ),
                assertion_type="Numeric",
                details={"found": closest, "expected": self.expected,
                         "diff": diff, "tolerance": tol,
                         "all_numbers": numbers},
            )

        return AssertionResult(
            passed=False,
            score=max(0.0, 1.0 - diff / max(abs(self.expected), 1)),
            reason=(
                f"Closest: {closest}, expected {self.expected} "
                f"(±{tol}), off by {diff:.4f}"
            ),
            assertion_type="Numeric",
            details={"found": closest, "expected": self.expected,
                     "diff": diff, "tolerance": tol,
                     "all_numbers": numbers},
        )

    def __repr__(self) -> str:
        return f"Numeric({self.expected}, ±{self.tolerance})"


# ─── Structured Assertions ───────────────────────────────────────


def _extract_json(text: str) -> Any:
    """Extract JSON from a response that may contain markdown fences."""
    # Try raw parse first
    try:
        return _json.loads(text.strip())
    except (ValueError, TypeError):
        pass

    # Try extracting from markdown code block
    fence_re = re.compile(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL,
    )
    m = fence_re.search(text)
    if m:
        try:
            return _json.loads(m.group(1).strip())
        except (ValueError, TypeError):
            pass

    # Try finding first { or [ and matching
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                if depth > 0:
                    depth -= 1
                if depth == 0:
                    try:
                        return _json.loads(text[start:i + 1])
                    except (ValueError, TypeError):
                        break

    return None


class JsonValid(Assertion):
    """Check that the response is valid JSON.

    Handles markdown code fences (` ```json ... ``` `).

    Example:
        >>> JsonValid().check('```json\\n{"a": 1}\\n```')
        ✅ JsonValid: Valid JSON (object) (1.00)
    """

    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        parsed = _extract_json(response)
        if parsed is not None:
            kind = type(parsed).__name__
            return AssertionResult(
                passed=True, score=1.0,
                reason=f"Valid JSON ({kind})",
                assertion_type="JsonValid",
                details={"type": kind},
            )
        return AssertionResult(
            passed=False, score=0.0,
            reason="Response is not valid JSON",
            assertion_type="JsonValid",
        )

    def __repr__(self) -> str:
        return "JsonValid()"


class JsonSchema(Assertion):
    """Validate JSON against a JSON Schema (subset).

    Supports basic validation without ``jsonschema`` dependency:
    type checking, required fields, minItems/maxItems for arrays.

    For full JSON Schema validation, install ``jsonschema`` and it
    will be used automatically.

    Args:
        schema: JSON Schema as a dict.

    Example:
        >>> schema = {"type": "object", "required": ["name"]}
        >>> JsonSchema(schema).check('{"name": "Alice"}')
        ✅ JsonSchema: Validates against schema (1.00)
    """

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema

    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        parsed = _extract_json(response)
        if parsed is None:
            return AssertionResult(
                passed=False, score=0.0,
                reason="Response is not valid JSON",
                assertion_type="JsonSchema",
            )

        errors = self._validate(parsed, self.schema)
        if not errors:
            return AssertionResult(
                passed=True, score=1.0,
                reason="Validates against schema",
                assertion_type="JsonSchema",
            )

        return AssertionResult(
            passed=False,
            score=max(0.0, 1.0 - len(errors) * 0.25),
            reason=f"Schema errors: {'; '.join(errors[:3])}",
            assertion_type="JsonSchema",
            details={"errors": errors},
        )

    def _validate(
        self, data: Any, schema: dict[str, Any],
    ) -> list[str]:
        """Basic schema validation (no external dependency)."""
        # Try jsonschema if available
        try:
            import jsonschema
            try:
                jsonschema.validate(data, schema)
                return []
            except jsonschema.ValidationError as e:
                return [e.message]
        except ImportError:
            pass

        # Fallback: basic type + required checks
        errors: list[str] = []

        expected_type = schema.get("type")
        if expected_type:
            type_map = {
                "object": dict, "array": list, "string": str,
                "number": (int, float), "integer": int,
                "boolean": bool,
            }
            py_type = type_map.get(expected_type)
            if py_type and not isinstance(data, py_type):
                errors.append(
                    f"Expected {expected_type}, got {type(data).__name__}"
                )

        if isinstance(data, dict):
            for req in schema.get("required", []):
                if req not in data:
                    errors.append(f"Missing required field: '{req}'")

            # Validate properties
            props = schema.get("properties", {})
            for key, prop_schema in props.items():
                if key in data:
                    errors.extend(self._validate(data[key], prop_schema))

        if isinstance(data, list):
            min_items = schema.get("minItems")
            max_items = schema.get("maxItems")
            if min_items is not None and len(data) < min_items:
                errors.append(
                    f"Array has {len(data)} items, min {min_items}"
                )
            if max_items is not None and len(data) > max_items:
                errors.append(
                    f"Array has {len(data)} items, max {max_items}"
                )

            # Validate items
            items_schema = schema.get("items")
            if items_schema and isinstance(items_schema, dict):
                for i, item in enumerate(data):
                    sub_errors = self._validate(item, items_schema)
                    errors.extend(
                        f"[{i}]: {e}" for e in sub_errors
                    )

        return errors

    def __repr__(self) -> str:
        return f"JsonSchema({self.schema})"


class JsonPath(Assertion):
    """Check a specific value inside a JSON response.

    Uses a simple dot-notation path (``$.field.subfield`` or
    ``field.subfield``). Supports array indexing (``items[0]``).

    Args:
        path: Dot-notation path to the value.
        expected: Expected value at that path.
        operator: Comparison operator (``"eq"``, ``"contains"``,
                  ``"gt"``, ``"lt"``, ``"exists"``).

    Example:
        >>> JsonPath("name", "Jupiter").check('{"name": "Jupiter"}')
        ✅ JsonPath: $.name == Jupiter (1.00)
    """

    def __init__(
        self,
        path: str,
        expected: Any = None,
        *,
        operator: str = "eq",
    ):
        self.path = path.lstrip("$.")
        self.expected = expected
        self.operator = operator

    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        parsed = _extract_json(response)
        if parsed is None:
            return AssertionResult(
                passed=False, score=0.0,
                reason="Response is not valid JSON",
                assertion_type="JsonPath",
            )

        value = self._resolve(parsed, self.path)
        if value is _MISSING:
            return AssertionResult(
                passed=self.operator == "not_exists",
                score=1.0 if self.operator == "not_exists" else 0.0,
                reason=f"Path '$.{self.path}' not found",
                assertion_type="JsonPath",
            )

        if self.operator == "exists":
            return AssertionResult(
                passed=True, score=1.0,
                reason=f"$.{self.path} exists",
                assertion_type="JsonPath",
            )

        passed = self._compare(value)
        return AssertionResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=(
                f"$.{self.path} {self.operator} {self.expected}: "
                f"{'✓' if passed else f'got {value!r}'}"
            ),
            assertion_type="JsonPath",
            details={"path": self.path, "value": value,
                     "expected": self.expected},
        )

    def _resolve(self, data: Any, path: str) -> Any:
        """Walk dot-path to extract value."""
        current = data
        parts = path.split(".")
        for part in parts:
            # Handle array index: items[0]
            idx_match = re.match(r"(\w+)\[(\d+)\]", part)
            if idx_match:
                key, idx = idx_match.group(1), int(idx_match.group(2))
                if isinstance(current, dict) and key in current:
                    current = current[key]
                    if isinstance(current, list) and idx < len(current):
                        current = current[idx]
                    else:
                        return _MISSING
                else:
                    return _MISSING
            elif isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return _MISSING
            else:
                return _MISSING
        return current

    def _compare(self, value: Any) -> bool:
        """Compare value with expected using operator."""
        if self.operator == "eq":
            return value == self.expected
        if self.operator == "contains":
            return (
                isinstance(value, str)
                and str(self.expected) in value
            )
        if self.operator == "gt":
            return value > self.expected
        if self.operator == "lt":
            return value < self.expected
        return False

    def __repr__(self) -> str:
        return f"JsonPath('{self.path}', {self.expected!r})"


_MISSING = object()  # sentinel for missing paths


# ─── Semantic Assertion ──────────────────────────────────────────


class Semantic(AsyncAssertion):
    """Embedding cosine similarity check.

    Compares the response to a reference text using embeddings.
    Requires an OpenAI-compatible embeddings API.

    Args:
        reference: The reference text to compare against.
        threshold: Minimum cosine similarity to pass (0.0–1.0).
        model: Embedding model name.
        base_url: API base URL.
        api_key: API key for the embeddings service.

    Example:
        >>> sem = Semantic("The answer is 36", threshold=0.8)
        >>> await sem.acheck("thirty-six")  # high similarity → pass
    """

    def __init__(
        self,
        reference: str,
        *,
        threshold: float = 0.80,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
    ):
        self.reference = reference
        self.threshold = threshold
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def acheck(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        try:
            import httpx
        except ImportError:
            return AssertionResult(
                passed=False, score=0.0,
                reason="httpx not installed",
                assertion_type="Semantic",
            )

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, headers=headers, json={
                "model": self.model,
                "input": [self.reference, response],
            })
            resp.raise_for_status()
            data = resp.json()

        emb = data.get("data", [])
        if len(emb) < 2:
            return AssertionResult(
                passed=False, score=0.0,
                reason="Embedding API returned insufficient data",
                assertion_type="Semantic",
            )

        vec_a = emb[0]["embedding"]
        vec_b = emb[1]["embedding"]
        similarity = self._cosine_similarity(vec_a, vec_b)

        passed = similarity >= self.threshold
        return AssertionResult(
            passed=passed,
            score=similarity,
            reason=(
                f"Cosine similarity: {similarity:.3f} "
                f"(threshold: {self.threshold})"
            ),
            assertion_type="Semantic",
            details={"similarity": similarity,
                     "threshold": self.threshold},
        )

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def __repr__(self) -> str:
        return f"Semantic(threshold={self.threshold})"


# ─── LLM Grade Assertion ────────────────────────────────────────


class LLMGrade(AsyncAssertion):
    """LLM judges the response against criteria.

    Sends the task + response to an LLM and asks it to grade
    the response on a 1-5 scale.

    Args:
        criteria: What to evaluate (e.g. "Is the math correct?").
        model: LLM model for grading.
        base_url: API base URL.
        api_key: API key.
        passing_score: Minimum score to pass (1-5 scale).
        temperature: Sampling temperature for the judge.

    Example:
        >>> judge = LLMGrade(
        ...     "Is the answer factually correct?",
        ...     model="gpt-4o-mini",
        ... )
        >>> await judge.acheck("Paris is the capital of France")
    """

    def __init__(
        self,
        criteria: str,
        *,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        passing_score: int = 4,
        temperature: float = 0.0,
    ):
        self.criteria = criteria
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.passing_score = passing_score
        self.temperature = temperature

    async def acheck(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        task = ""
        if context:
            task = context.get("task", "")

        prompt = (
            "You are an evaluation judge. Score the following response "
            "on a scale of 1-5.\n\n"
            f"CRITERIA: {self.criteria}\n\n"
        )
        if task:
            prompt += f"TASK: {task}\n\n"
        prompt += (
            f"RESPONSE: {response}\n\n"
            "Reply with ONLY a JSON object: "
            '{"score": <1-5>, "reason": "<brief explanation>"}'
        )

        try:
            import httpx
        except ImportError:
            return AssertionResult(
                passed=False, score=0.0,
                reason="httpx not installed",
                assertion_type="LLMGrade",
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": 200,
            })
            resp.raise_for_status()
            data = resp.json()

        reply = data["choices"][0]["message"]["content"]

        # Parse score from reply
        score_val, reason = self._parse_reply(reply)

        passed = score_val >= self.passing_score
        normalized = score_val / 5.0

        return AssertionResult(
            passed=passed,
            score=normalized,
            reason=f"LLM grade: {score_val}/5 — {reason}",
            assertion_type="LLMGrade",
            details={"raw_score": score_val, "max_score": 5,
                     "passing_score": self.passing_score,
                     "judge_reason": reason,
                     "judge_model": self.model},
        )

    @staticmethod
    def _parse_reply(reply: str) -> tuple[int, str]:
        """Extract score and reason from LLM reply."""
        try:
            parsed = _extract_json(reply)
            if isinstance(parsed, dict):
                score = int(parsed.get("score", 0))
                reason = str(parsed.get("reason", ""))
                return min(5, max(1, score)), reason
        except (ValueError, TypeError):
            pass

        # Fallback: look for a number
        nums = re.findall(r"\b([1-5])\b", reply)
        if nums:
            return int(nums[0]), reply[:100]
        return 1, f"Could not parse judge reply: {reply[:100]}"

    def __repr__(self) -> str:
        return f"LLMGrade('{self.criteria[:40]}...')"


# ─── Custom Assertion ────────────────────────────────────────────


class Custom(Assertion):
    """User-defined assertion via a callable.

    The function can return:
    - ``bool``: True = pass (score 1.0), False = fail (score 0.0)
    - ``float``: Score (0.0–1.0), passes if >= 0.5
    - ``AssertionResult``: Full control
    - ``tuple[bool, str]``: (passed, reason)

    Args:
        fn: A callable that takes a response string.
        name: Display name for the assertion.

    Example:
        >>> Custom(lambda r: "def " in r, name="has_function")
        >>> Custom(lambda r: len(r.split()) >= 10, name="min_words")
    """

    def __init__(
        self,
        fn: Callable[[str], bool | float | AssertionResult | tuple[bool, str]],
        *,
        name: str = "custom",
    ):
        self.fn = fn
        self.name = name

    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        try:
            result = self.fn(response)
        except Exception as e:
            return AssertionResult(
                passed=False, score=0.0,
                reason=f"Custom assertion error: {e}",
                assertion_type=f"Custom({self.name})",
            )

        if isinstance(result, AssertionResult):
            return result
        if isinstance(result, bool):
            return AssertionResult(
                passed=result,
                score=1.0 if result else 0.0,
                reason="Custom check passed" if result else "Custom check failed",
                assertion_type=f"Custom({self.name})",
            )
        if isinstance(result, tuple) and len(result) == 2:
            passed, reason = result
            return AssertionResult(
                passed=bool(passed),
                score=1.0 if passed else 0.0,
                reason=str(reason),
                assertion_type=f"Custom({self.name})",
            )
        if isinstance(result, (int, float)):
            score = float(result)
            return AssertionResult(
                passed=score >= 0.5,
                score=max(0.0, min(1.0, score)),
                reason=f"Custom score: {score:.2f}",
                assertion_type=f"Custom({self.name})",
            )

        return AssertionResult(
            passed=bool(result),
            score=1.0 if result else 0.0,
            reason=f"Custom result: {result}",
            assertion_type=f"Custom({self.name})",
        )

    def __repr__(self) -> str:
        return f"Custom({self.name})"


# ─── Composite Assertions ───────────────────────────────────────


class All(Assertion):
    """All child assertions must pass.

    Example:
        >>> All(Numeric(36), NotContains(["sorry"]))
    """

    def __init__(self, *assertions: Assertion):
        self.assertions = list(assertions)

    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        results = [
            a.check(response, context=context) for a in self.assertions
        ]
        all_passed = all(r.passed for r in results)
        avg_score = (
            sum(r.score for r in results) / len(results)
            if results else 0.0
        )
        failed = [r for r in results if not r.passed]

        if all_passed:
            reason = f"All {len(results)} assertions passed"
        else:
            reasons = [r.reason for r in failed]
            reason = f"{len(failed)} failed: {'; '.join(reasons[:3])}"

        return AssertionResult(
            passed=all_passed,
            score=avg_score,
            reason=reason,
            assertion_type="All",
            details={"results": [
                {"type": r.assertion_type, "passed": r.passed,
                 "score": r.score, "reason": r.reason}
                for r in results
            ]},
        )

    def __repr__(self) -> str:
        inner = ", ".join(repr(a) for a in self.assertions)
        return f"All({inner})"


class AnyOf(Assertion):
    """At least one child assertion must pass.

    Named ``Any_`` to avoid shadowing the builtin ``any``.

    Example:
        >>> AnyOf(Exact("36"), Numeric(36), Contains(["thirty-six"]))
    """

    def __init__(self, *assertions: Assertion):
        self.assertions = list(assertions)

    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        results = [
            a.check(response, context=context) for a in self.assertions
        ]
        any_passed = any(r.passed for r in results)
        best_score = max((r.score for r in results), default=0.0)

        if any_passed:
            winner = next(r for r in results if r.passed)
            reason = f"Passed via {winner.assertion_type}: {winner.reason}"
        else:
            reason = f"None of {len(results)} assertions passed"

        return AssertionResult(
            passed=any_passed,
            score=best_score,
            reason=reason,
            assertion_type="Any",
            details={"results": [
                {"type": r.assertion_type, "passed": r.passed,
                 "score": r.score, "reason": r.reason}
                for r in results
            ]},
        )

    def __repr__(self) -> str:
        inner = ", ".join(repr(a) for a in self.assertions)
        return f"AnyOf({inner})"


class AtLeast(Assertion):
    """At least N of the child assertions must pass.

    Args:
        n: Minimum number that must pass.
        assertions: List of assertions to check.

    Example:
        >>> AtLeast(2, [Exact("36"), Numeric(36), Contains(["36"])])
    """

    def __init__(self, n: int, assertions: list[Assertion]):
        self.n = n
        self.assertions = assertions

    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        results = [
            a.check(response, context=context) for a in self.assertions
        ]
        n_passed = sum(1 for r in results if r.passed)
        avg_score = (
            sum(r.score for r in results) / len(results)
            if results else 0.0
        )

        passed = n_passed >= self.n
        reason = f"{n_passed}/{len(results)} passed (need {self.n})"

        return AssertionResult(
            passed=passed,
            score=avg_score,
            reason=reason,
            assertion_type="AtLeast",
            details={"n_required": self.n, "n_passed": n_passed,
                     "results": [
                         {"type": r.assertion_type, "passed": r.passed,
                          "score": r.score}
                         for r in results
                     ]},
        )

    def __repr__(self) -> str:
        return f"AtLeast({self.n}, [{len(self.assertions)} assertions])"


class Weighted(Assertion):
    """Weighted combination of assertions.

    Each assertion is paired with a weight. The final score is
    the weighted average. Passes if the weighted score >= threshold.

    Args:
        assertions: List of ``(assertion, weight)`` tuples.
        threshold: Minimum weighted score to pass (default 0.7).

    Example:
        >>> Weighted([
        ...     (Numeric(36), 0.6),
        ...     (NotContains(["sorry"]), 0.2),
        ...     (Custom(lambda r: len(r) > 5), 0.2),
        ... ])
    """

    def __init__(
        self,
        assertions: list[tuple[Assertion, float]],
        *,
        threshold: float = 0.7,
    ):
        self.assertions = assertions
        self.threshold = threshold

    def check(
        self, response: str, *, context: dict[str, Any] | None = None,
    ) -> AssertionResult:
        total_weight = sum(w for _, w in self.assertions)
        if total_weight == 0:
            return AssertionResult(
                passed=False, score=0.0,
                reason="No assertions with weight > 0",
                assertion_type="Weighted",
            )

        weighted_score = 0.0
        results = []
        for assertion, weight in self.assertions:
            r = assertion.check(response, context=context)
            normalized_weight = weight / total_weight
            weighted_score += r.score * normalized_weight
            results.append((r, weight))

        passed = weighted_score >= self.threshold
        reason = (
            f"Weighted score: {weighted_score:.3f} "
            f"(threshold: {self.threshold})"
        )

        return AssertionResult(
            passed=passed,
            score=weighted_score,
            reason=reason,
            assertion_type="Weighted",
            details={"threshold": self.threshold,
                     "results": [
                         {"type": r.assertion_type, "weight": w,
                          "score": r.score, "passed": r.passed}
                         for r, w in results
                     ]},
        )

    def __repr__(self) -> str:
        return f"Weighted([{len(self.assertions)} assertions])"


# ─── Public API ──────────────────────────────────────────────────

__all__ = [
    # Result
    "AssertionResult",
    # Base
    "Assertion",
    "AsyncAssertion",
    # String
    "Exact",
    "Contains",
    "NotContains",
    "RegexMatch",
    # Numeric
    "Numeric",
    # Structured
    "JsonValid",
    "JsonSchema",
    "JsonPath",
    # Semantic
    "Semantic",
    # LLM
    "LLMGrade",
    # Custom
    "Custom",
    # Composites
    "All",
    "AnyOf",
    "AtLeast",
    "Weighted",
]

# Backward compat
Any_ = AnyOf
