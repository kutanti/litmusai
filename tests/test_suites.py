"""Tests for built-in test suites."""

import pytest

from litmusai.core.suite import TestSuite


class TestBuiltInSuites:
    """Verify all built-in suites load correctly."""

    def test_available_lists_all(self):
        suites = TestSuite.available()
        assert len(suites) >= 8
        expected = [
            "coding", "research", "safety", "planning",
            "customer_support", "summarization",
            "instruction_following", "tool_use",
        ]
        for name in expected:
            assert name in suites, f"Suite '{name}' not found"

    @pytest.mark.parametrize("suite_name", [
        "coding", "research", "safety", "planning",
        "customer_support", "summarization",
        "instruction_following", "tool_use",
    ])
    def test_suite_loads(self, suite_name):
        suite = TestSuite.load(suite_name)
        assert suite.name
        assert len(suite.cases) > 0, (
            f"Suite '{suite_name}' has no cases"
        )

    @pytest.mark.parametrize("suite_name", [
        "coding", "research", "safety", "planning",
        "customer_support", "summarization",
        "instruction_following", "tool_use",
    ])
    def test_suite_cases_have_ids(self, suite_name):
        suite = TestSuite.load(suite_name)
        ids = set()
        for case in suite.cases:
            assert case.id, f"Case in {suite_name} has no id"
            assert case.name, f"Case in {suite_name} has no name"
            assert case.task, f"Case in {suite_name} has no task"
            assert case.id not in ids, (
                f"Duplicate id '{case.id}' in {suite_name}"
            )
            ids.add(case.id)

    def test_total_cases(self):
        """All suites combined should have 40+ test cases."""
        total = 0
        for name in TestSuite.available():
            suite = TestSuite.load(name)
            total += len(suite.cases)
        assert total >= 40, f"Expected 40+ cases, got {total}"

    def test_customer_support_suite(self):
        suite = TestSuite.load("customer_support")
        assert suite.name == "customer-support"
        assert len(suite.cases) == 8
        # Check first case
        assert suite.cases[0].id == "cs_001"

    def test_summarization_suite(self):
        suite = TestSuite.load("summarization")
        assert suite.name == "summarization"
        assert len(suite.cases) == 5

    def test_instruction_following_suite(self):
        suite = TestSuite.load("instruction_following")
        assert suite.name == "instruction-following"
        assert len(suite.cases) == 9

    def test_tool_use_suite(self):
        suite = TestSuite.load("tool_use")
        assert suite.name == "tool-use"
        assert len(suite.cases) == 6

    def test_nonexistent_suite(self):
        with pytest.raises(Exception):
            TestSuite.load("nonexistent_suite_xyz")
