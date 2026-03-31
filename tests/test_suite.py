"""Tests for the TestSuite class."""

import tempfile

from litmusai.core.suite import TestSuite


class TestTestSuite:
    def test_create_suite(self):
        suite = TestSuite(name="test-suite")
        assert suite.name == "test-suite"
        assert len(suite) == 0

    def test_add_case(self):
        suite = TestSuite(name="test")
        suite.add(task="Say hello", expected="hello", name="Greeting")
        assert len(suite) == 1
        assert suite.cases[0].task == "Say hello"

    def test_yaml_roundtrip(self):
        suite = TestSuite(name="test", description="A test suite")
        suite.add(task="What is 2+2?", expected="4", name="Math")
        suite.add(task="Say hello", expected="hello", name="Greeting")

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            suite.to_yaml(f.name)
            loaded = TestSuite.from_yaml(f.name)

        assert loaded.name == "test"
        assert len(loaded) == 2
        assert loaded.cases[0].task == "What is 2+2?"

    def test_repr(self):
        suite = TestSuite(name="test")
        suite.add(task="hello")
        assert "test" in repr(suite)
        assert "1" in repr(suite)
