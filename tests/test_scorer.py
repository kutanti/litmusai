"""Tests for the Scorer class."""

from litmusai.core.agent import AgentResponse
from litmusai.core.scorer import Scorer
from litmusai.core.suite import TestCase


class TestScorer:
    def setup_method(self):
        self.scorer = Scorer()

    def test_exact_match_pass(self):
        case = TestCase(id="1", name="test", task="Say hello", expected="hello")
        response = AgentResponse(output="Hello there!")
        result = self.scorer.score(case, response)
        assert result.passed

    def test_exact_match_fail(self):
        case = TestCase(id="1", name="test", task="Say hello", expected="goodbye")
        response = AgentResponse(output="Hello there!")
        result = self.scorer.score(case, response)
        assert not result.passed

    def test_contains_check(self):
        case = TestCase(
            id="1", name="test", task="test",
            expected_contains=["python", "function"]
        )
        response = AgentResponse(output="Here is a Python function that works")
        result = self.scorer.score(case, response)
        assert result.passed

    def test_error_response(self):
        case = TestCase(id="1", name="test", task="test")
        response = AgentResponse(output="", success=False, error="timeout")
        result = self.scorer.score(case, response)
        assert not result.passed
        assert result.score == 0.0
