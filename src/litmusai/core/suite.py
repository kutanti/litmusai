"""Test suite management — define, load, and run test suites."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from litmusai.assertions import Assertion


@dataclass
class TestCase:
    """A single test case for agent evaluation.

    Supports both legacy scoring (``expected_contains``) and the
    new assertion engine (``assertions`` list).  When ``assertions``
    is non-empty it takes precedence.
    """

    id: str
    name: str
    task: str
    expected: str | None = None
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)
    assertions: list[Assertion] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    timeout_seconds: float = 60.0
    metadata: dict[str, Any] = field(default_factory=dict)
    scorer: str = "default"


class TestSuite:
    """Collection of test cases for evaluating AI agents."""

    def __init__(self, name: str, cases: list[TestCase] | None = None, description: str = ""):
        self.name = name
        self.description = description
        self.cases = cases or []

    def add_case(self, case: TestCase) -> None:
        """Add a test case to the suite."""
        self.cases.append(case)

    def add(
        self,
        task: str,
        expected: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> TestCase:
        """Quick-add a test case."""
        case = TestCase(
            id=f"test_{len(self.cases) + 1:03d}",
            name=name or f"Test {len(self.cases) + 1}",
            task=task,
            expected=expected,
            **kwargs,
        )
        self.cases.append(case)
        return case

    @classmethod
    def load(cls, name: str) -> TestSuite:
        """Load a built-in test suite by name."""
        suite_dir = Path(__file__).parent.parent / "suites" / name
        if not suite_dir.exists():
            raise ValueError(
                f"Suite '{name}' not found. Available: {cls.available()}"
            )

        config_file = suite_dir / "suite.yaml"
        if config_file.exists():
            return cls.from_yaml(config_file)

        return cls(name=name)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TestSuite:
        """Load a test suite from a YAML file.

        Supports YAML-defined assertions::

            cases:
              - id: q1
                task: "What is 6*7?"
                assertions:
                  - type: numeric
                    value: 42
                  - type: contains
                    value: "42"
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        suite = cls(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
        )

        for case_data in data.get("cases", []):
            # Parse YAML assertions into Assertion objects
            raw_assertions = case_data.pop("assertions", None)
            case = TestCase(**case_data)
            if raw_assertions is not None:
                if not isinstance(raw_assertions, list):
                    msg = (
                        f"'assertions' must be a list in case "
                        f"'{case_data.get('id', '?')}', "
                        f"got {type(raw_assertions).__name__}"
                    )
                    raise ValueError(msg)
                case.assertions = _parse_yaml_assertions(
                    raw_assertions,
                )
            suite.add_case(case)

        return suite

    def to_yaml(self, path: str | Path) -> None:
        """Save the test suite to a YAML file.

        Note: ``assertions`` are not serialized (they are Python
        objects).  Use YAML for legacy-style suites.
        """
        path = Path(path)
        # Skip non-serializable fields (assertions are Python objects)
        skip_fields = {"assertions", "scorer"}
        data = {
            "name": self.name,
            "description": self.description,
            "cases": [
                {
                    k: v for k, v in case.__dict__.items()
                    if v and k not in skip_fields
                }
                for case in self.cases
            ],
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def available(cls) -> list[str]:
        """List available built-in test suites."""
        suites_dir = Path(__file__).parent.parent / "suites"
        if not suites_dir.exists():
            return []
        return [
            d.name for d in suites_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ]

    def __len__(self) -> int:
        return len(self.cases)

    def __repr__(self) -> str:
        return f"TestSuite(name='{self.name}', cases={len(self.cases)})"


# ─── YAML Assertion Parser ──────────────────────────────────────


_ASSERTION_TYPES: dict[str, type] = {}


def _ensure_registry() -> None:
    """Lazily populate the assertion type registry."""
    if _ASSERTION_TYPES:
        return

    from litmusai.assertions import (
        All,
        AnyOf,
        Contains,
        Exact,
        JsonPath,
        JsonSchema,
        JsonValid,
        NotContains,
        Numeric,
        RegexMatch,
    )

    _ASSERTION_TYPES.update({
        "contains": Contains,
        "not_contains": NotContains,
        "notcontains": NotContains,
        "exact": Exact,
        "numeric": Numeric,
        "regex": RegexMatch,
        "regex_match": RegexMatch,
        "json_valid": JsonValid,
        "jsonvalid": JsonValid,
        "json_schema": JsonSchema,
        "jsonschema": JsonSchema,
        "json_path": JsonPath,
        "jsonpath": JsonPath,
        "all": All,
        "any_of": AnyOf,
        "anyof": AnyOf,
    })


def _parse_single_assertion(spec: dict[str, Any]) -> Any:
    """Parse a single YAML assertion spec into an Assertion object.

    Supported formats::

        # Simple value-based
        - type: contains
          value: "hello"

        # With options
        - type: contains
          value: "hello"
          case_sensitive: true

        # Numeric with tolerance
        - type: numeric
          value: 42
          tolerance: 0.1

        # JSON path
        - type: json_path
          path: "$.name"
          expected: "Alice"

        # JSON schema
        - type: json_schema
          schema:
            type: object
            required: ["name"]

        # Regex
        - type: regex
          pattern: "\\d{3}-\\d{4}"

        # Not contains (list of patterns)
        - type: not_contains
          patterns: ["hack", "exploit"]

        # Composite
        - type: any_of
          assertions:
            - type: contains
              value: "yes"
            - type: contains
              value: "correct"
    """
    _ensure_registry()

    raw_type = spec.get("type", "")
    if not isinstance(raw_type, str) or not raw_type.strip():
        msg = f"Assertion spec missing 'type' or type is not a string: {spec}"
        raise ValueError(msg)
    atype = raw_type.lower().strip()

    cls = _ASSERTION_TYPES.get(atype)
    if cls is None:
        valid = ", ".join(sorted(_ASSERTION_TYPES.keys()))
        msg = f"Unknown assertion type '{atype}'. Valid: {valid}"
        raise ValueError(msg)

    # Build kwargs from spec (excluding 'type')
    kwargs = {k: v for k, v in spec.items() if k != "type"}

    # Type-specific handling
    from litmusai.assertions import (
        All,
        AnyOf,
        Contains,
        Exact,
        JsonPath,
        JsonSchema,
        JsonValid,
        NotContains,
        Numeric,
        RegexMatch,
    )

    if cls is Contains:
        patterns = kwargs.get("patterns") or kwargs.get("value", "")
        if isinstance(patterns, str):
            patterns = [patterns]
        return Contains(
            patterns,
            mode=kwargs.get("mode", "all"),
            case_sensitive=kwargs.get("case_sensitive", False),
        )

    if cls is NotContains:
        patterns = kwargs.get("patterns") or kwargs.get("value")
        if isinstance(patterns, str):
            patterns = [patterns]
        return NotContains(
            patterns or [],
            case_sensitive=kwargs.get("case_sensitive", False),
        )

    if cls is Exact:
        return Exact(
            kwargs.get("value", ""),
            case_sensitive=kwargs.get("case_sensitive", False),
            strip=kwargs.get("strip", True),
        )

    if cls is Numeric:
        return Numeric(
            kwargs.get("value", 0),
            tolerance=kwargs.get("tolerance", 0.01),
        )

    if cls is RegexMatch:
        return RegexMatch(kwargs.get("pattern", ""))

    if cls is JsonValid:
        return JsonValid()

    if cls is JsonSchema:
        return JsonSchema(kwargs.get("schema", {}))

    if cls is JsonPath:
        return JsonPath(
            kwargs.get("path", ""),
            expected=kwargs.get("expected"),
            operator=kwargs.get("operator", "eq"),
        )

    if cls in (All, AnyOf):
        sub_specs = kwargs.get("assertions", [])
        sub_assertions = _parse_yaml_assertions(sub_specs)
        return cls(*sub_assertions)

    # Fallback — try passing kwargs directly
    return cls(**kwargs)


def _parse_yaml_assertions(
    specs: list[dict[str, Any]],
) -> list[Any]:
    """Parse a list of YAML assertion specs."""
    return [_parse_single_assertion(s) for s in specs]
