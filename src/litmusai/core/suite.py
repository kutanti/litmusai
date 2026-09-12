"""Test suite management — define, load, and run test suites."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from litmusai.datasets import DatasetInfo, SourceReference, snapshot_json
from litmusai.ground_truth import GroundTruth
from litmusai.metrics.schema import SCHEMA_VERSION, MetricConfig

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
    name: str = ""
    task: str = ""
    expected: str | None = None
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)
    assertions: list[Assertion] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    timeout_seconds: float = 60.0
    metadata: dict[str, Any] = field(default_factory=dict)
    scorer: str = "default"
    ground_truth: GroundTruth | None = None
    inputs: dict[str, Any] | None = None
    source: SourceReference | None = None

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.id

    @property
    def expected_value(self) -> Any:
        """Read labeled truth, including the legacy metadata representation."""
        if self.ground_truth is not None:
            return self.ground_truth.answer
        return self.metadata.get("ground_truth", {}).get("answer")

    def to_dict(self, *, include_assertions: bool = True) -> dict[str, Any]:
        """Export a case without dropping empty values, labels, or source IDs.

        Set include_assertions=False for data-only fingerprints. Callable
        assertions cannot be exported as a portable dataset definition.
        """
        data = {k: v for k, v in self.__dict__.items()
                if k not in {"assertions", "ground_truth", "source"}}
        data["metadata"] = dict(self.metadata)
        truth = self.ground_truth
        if truth is None and "ground_truth" in self.metadata:
            truth = GroundTruth.from_dict(self.metadata["ground_truth"])
        if truth is not None:
            data["ground_truth"] = truth.to_dict()
            data["metadata"]["ground_truth"] = truth.to_dict()
        if self.source is not None:
            data["source"] = self.source.model_dump(mode="json")
        if include_assertions:
            try:
                data["assertions"] = [_assertion_to_spec(item) for item in self.assertions]
            except ValueError as exc:
                raise ValueError(f"case {self.id!r}: {exc}") from exc
        return snapshot_json(data, f"case {self.id!r}")


class TestSuite:
    """Collection of test cases for evaluating AI agents."""

    def __init__(
        self, name: str, cases: list[TestCase] | None = None, description: str = "",
        *, metrics: MetricConfig | None = None, dataset: DatasetInfo | None = None,
    ):
        self.name = name
        self.description = description
        self.cases = cases or []
        self.metrics = metrics
        self.dataset = dataset or DatasetInfo()

    @property
    def dataset_info(self) -> DatasetInfo:
        """Describe the current input data; case/key ordering does not affect its hash.

        The fingerprint covers cases, labels and metric configuration. Runtime
        assertions, scorers, timeouts, and dataset-level identity are excluded.
        """
        self.validate_case_ids()
        records = []
        for case in sorted(self.cases, key=lambda item: item.id):
            record = case.to_dict(include_assertions=False)
            record.pop("scorer")
            record.pop("timeout_seconds")
            records.append(record)
        content = {"cases": records,
                   "metrics": self.metrics.model_dump() if self.metrics else None}
        canonical = json.dumps(content, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
        fingerprint = "sha256:" + hashlib.sha256(canonical).hexdigest()
        return DatasetInfo.model_validate({**self.dataset.model_dump(), "fingerprint": fingerprint})

    def snapshot(self) -> TestSuite:
        """Capture inputs and provenance before an evaluation starts.

        Assertion implementations stay callable Python objects. Input data is
        copied so an agent cannot modify subsequent repetitions or saved truth.
        """
        self.validate_metrics()
        cases = [replace(case, inputs=snapshot_json(case.inputs, f"case {case.id!r} inputs"),
                         metadata=snapshot_json(case.metadata, f"case {case.id!r} metadata"),
                         source=case.source.model_copy(deep=True) if case.source else None,
                         ground_truth=deepcopy(case.ground_truth),
                         expected_contains=list(case.expected_contains),
                         expected_not_contains=list(case.expected_not_contains),
                         tags=list(case.tags), assertions=list(case.assertions))
                 for case in self.cases]
        return TestSuite(self.name, cases, self.description,
                         metrics=self.metrics.model_copy(deep=True) if self.metrics else None,
                         dataset=self.dataset_info)

    def add_case(self, case: TestCase) -> None:
        """Add a test case to the suite."""
        self.cases.append(case)

    def validate_case_ids(self) -> None:
        """Require nonempty, unique string IDs before loading or running a suite."""
        seen: set[str] = set()
        for case in self.cases:
            if not isinstance(case.id, str) or not case.id.strip() or case.id in seen:
                raise ValueError(f"case IDs must be nonempty and unique: {case.id!r}")
            seen.add(case.id)

    def validate_metrics(self) -> None:
        """Validate labeled identities and all ground truth before agent execution."""
        self.validate_case_ids()
        if not isinstance(self.dataset, DatasetInfo):
            raise ValueError("dataset must be a DatasetInfo")
        for case in self.cases:
            if not isinstance(case.metadata, dict):
                raise ValueError(f"case {case.id!r}: metadata must be a mapping")
            if case.inputs is not None and not isinstance(case.inputs, dict):
                raise ValueError(f"case {case.id!r}: inputs must be a mapping or None")
            if case.source is not None and not isinstance(case.source, SourceReference):
                raise ValueError(f"case {case.id!r}: source must be a SourceReference")
            snapshot_json(case.inputs, f"case {case.id!r} inputs")
            snapshot_json(case.metadata, f"case {case.id!r} metadata")
        if self.metrics is None:
            return
        from litmusai.metrics.classification import validate_classification_truth
        from litmusai.metrics.extraction import validate_extraction_truth

        MetricConfig.model_validate(self.metrics.model_dump())
        validate = (validate_classification_truth if self.metrics.task_type == "classification"
                    else validate_extraction_truth)
        for case in self.cases:
            try:
                validate(case.expected_value, self.metrics)
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"case {case.id!r}: {exc}") from exc

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
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data, default_name=path.stem)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, default_name: str = "dataset") -> TestSuite:
        """Load the versioned dataset contract, including legacy suite dictionaries."""
        if not isinstance(data, dict):
            raise ValueError("dataset must be a mapping")
        data = snapshot_json(data, "dataset")
        unknown = set(data) - {
            "schema_version", "name", "description", "cases", "metrics", "dataset", "task_type",
        }
        if unknown:
            raise ValueError(f"unsupported dataset fields: {', '.join(sorted(unknown))}")
        if data.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported suite schema_version: {data['schema_version']!r}")
        if not isinstance(data.get("cases", []), list):
            raise ValueError("dataset cases must be a list")

        suite = cls(
            name=data.get("name", default_name),
            description=data.get("description", ""),
            metrics=MetricConfig.model_validate(data["metrics"]) if "metrics" in data else None,
            dataset=DatasetInfo.model_validate(data.get("dataset", {})),
        )
        task_type = suite.metrics.task_type if suite.metrics else "assertion"
        if data.get("task_type", task_type) != task_type:
            raise ValueError(
                f"dataset task_type must be {task_type!r} for its metric configuration"
            )

        for case_data in data.get("cases", []):
            if not isinstance(case_data, dict):
                raise ValueError("each dataset case must be a mapping")
            # Parse YAML assertions into Assertion objects
            raw_assertions = case_data.pop("assertions", None)
            has_ground_truth = "ground_truth" in case_data
            raw_ground_truth = case_data.pop("ground_truth", None)
            try:
                if "source" in case_data and case_data["source"] is not None:
                    case_data["source"] = SourceReference.model_validate(case_data["source"])
                case = TestCase(**case_data)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"case {case_data.get('id', '<missing>')!r}: {exc}") from exc
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
            # Auto-generate assertions from ground_truth if no
            # explicit assertions were provided
            if has_ground_truth:
                if not isinstance(raw_ground_truth, dict):
                    msg = (
                        f"'ground_truth' must be a mapping in case "
                        f"'{case.id}', got {type(raw_ground_truth).__name__}"
                    )
                    raise ValueError(msg)
                try:
                    gt = GroundTruth.from_dict(raw_ground_truth)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Case '{case.id}': {exc}") from exc
                case.ground_truth = gt
                if raw_assertions is None and not case.assertions and suite.metrics is None:
                    case.assertions = gt.to_assertions()
                case.metadata["ground_truth"] = gt.to_dict()
            suite.add_case(case)

        suite.validate_metrics()
        if suite.dataset.fingerprint is not None:
            if suite.dataset_info.fingerprint != suite.dataset.fingerprint:
                raise ValueError(
                    "dataset fingerprint does not match its cases and metric configuration"
                )
        return suite

    def to_dict(self) -> dict[str, Any]:
        """Export a versioned dataset with its current content fingerprint."""
        self.validate_metrics()
        data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION, "name": self.name, "description": self.description,
            "task_type": self.metrics.task_type if self.metrics else "assertion",
            "dataset": self.dataset_info.model_dump(mode="json"),
            "cases": [case.to_dict() for case in self.cases],
        }
        if self.metrics is not None:
            data["metrics"] = self.metrics.model_dump()
        return data

    @classmethod
    def from_json(cls, path: str | Path) -> TestSuite:
        """Read an exported dataset from a UTF-8 JSON file."""
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), default_name=path.stem)

    def to_json(self, path: str | Path) -> None:
        """Save a portable dataset as UTF-8 JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")

    def to_yaml(self, path: str | Path) -> None:
        """Save a dataset and supported declarative assertions as UTF-8 YAML."""
        path = Path(path)
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False,
                           allow_unicode=True)

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


def _assertion_to_spec(assertion: Any) -> dict[str, Any]:
    """Serialize supported assertions without inspecting callable/client state."""
    from litmusai.assertions import (
        All,
        AnyOf,
        AtLeast,
        Contains,
        Exact,
        JsonPath,
        JsonSchema,
        JsonValid,
        NotContains,
        Numeric,
        RegexMatch,
        Weighted,
    )

    cls = type(assertion)
    if cls is Exact:
        return {"type": "exact", "value": assertion.expected,
                "case_sensitive": assertion.case_sensitive, "strip": assertion.strip}
    if cls in (Contains, NotContains):
        spec = {"type": "contains" if cls is Contains else "not_contains",
                "patterns": assertion.patterns, "case_sensitive": assertion.case_sensitive}
        if cls is Contains:
            spec["mode"] = assertion.mode
        return spec
    if cls is Numeric:
        return {"type": "numeric", "value": assertion.expected, "tolerance": assertion.tolerance,
                "relative_tolerance": assertion.relative_tolerance}
    if cls is RegexMatch:
        return {"type": "regex", "pattern": assertion.pattern, "flags": int(assertion.flags),
                "full_match": assertion.full_match}
    if cls is JsonValid:
        return {"type": "json_valid"}
    if cls is JsonSchema:
        return {"type": "json_schema", "schema": assertion.schema}
    if cls is JsonPath:
        return {"type": "json_path", "path": assertion.path, "expected": assertion.expected,
                "operator": assertion.operator}
    if cls in (All, AnyOf, AtLeast):
        spec = {"type": "all" if cls is All else "any_of" if cls is AnyOf else "at_least",
                "assertions": [_assertion_to_spec(item) for item in assertion.assertions]}
        if cls is AtLeast:
            spec["n"] = assertion.n
        return spec
    if cls is Weighted:
        return {"type": "weighted", "threshold": assertion.threshold,
                "assertions": [{"assertion": _assertion_to_spec(item), "weight": weight}
                               for item, weight in assertion.assertions]}
    raise ValueError(f"assertion {cls.__name__} cannot be exported as a portable specification; "
                     "keep callable evaluators in Python")


def _ensure_registry() -> None:
    """Lazily populate the assertion type registry."""
    if _ASSERTION_TYPES:
        return

    from litmusai.assertions import (
        All,
        AnyOf,
        AtLeast,
        Contains,
        Exact,
        JsonPath,
        JsonSchema,
        JsonValid,
        NotContains,
        Numeric,
        RegexMatch,
        Weighted,
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
        "at_least": AtLeast,
        "weighted": Weighted,
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

    if not isinstance(spec, dict):
        raise ValueError("each assertion specification must be a mapping")
    raw_type = spec.get("type", "")
    if not isinstance(raw_type, str) or not raw_type.strip():
        msg = f"Assertion spec missing 'type' or type is not a string: {spec}"
        raise ValueError(msg)
    atype = raw_type.lower().strip()

    cls = _ASSERTION_TYPES.get(atype)
    if cls is None:
        # Check custom plugin registry
        from litmusai.assertions import _custom_assertions

        cls = _custom_assertions.get(atype)

    if cls is None:
        valid = ", ".join(sorted(
            set(_ASSERTION_TYPES.keys()) | set(_custom_assertions.keys())
        ))
        msg = f"Unknown assertion type '{atype}'. Valid: {valid}"
        raise ValueError(msg)

    # Build kwargs from spec (excluding 'type')
    kwargs = {k: v for k, v in spec.items() if k != "type"}

    # Type-specific handling
    from litmusai.assertions import (
        All,
        AnyOf,
        AtLeast,
        Contains,
        Exact,
        JsonPath,
        JsonSchema,
        JsonValid,
        NotContains,
        Numeric,
        RegexMatch,
        Weighted,
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
            relative_tolerance=kwargs.get("relative_tolerance"),
        )

    if cls is RegexMatch:
        return RegexMatch(kwargs.get("pattern", ""), flags=kwargs.get("flags", re.IGNORECASE),
                          full_match=kwargs.get("full_match", False))

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

    if cls is AtLeast:
        return AtLeast(kwargs["n"], _parse_yaml_assertions(kwargs["assertions"]))

    if cls is Weighted:
        return Weighted([(_parse_single_assertion(item["assertion"]), item["weight"])
                         for item in kwargs["assertions"]], threshold=kwargs.get("threshold", 0.5))

    # Fallback — try passing kwargs directly
    return cls(**kwargs)


def _parse_yaml_assertions(
    specs: list[dict[str, Any]],
) -> list[Any]:
    """Parse a list of YAML assertion specs."""
    return [_parse_single_assertion(s) for s in specs]
