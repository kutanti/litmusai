"""Test suite management — define, load, and run test suites."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TestCase:
    """A single test case for agent evaluation."""
    id: str
    name: str
    task: str
    expected: str | None = None
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)
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
        """Load a test suite from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        suite = cls(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
        )

        for case_data in data.get("cases", []):
            suite.add_case(TestCase(**case_data))

        return suite

    def to_yaml(self, path: str | Path) -> None:
        """Save the test suite to a YAML file."""
        path = Path(path)
        data = {
            "name": self.name,
            "description": self.description,
            "cases": [
                {k: v for k, v in case.__dict__.items() if v}
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
