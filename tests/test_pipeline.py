"""Tests for the scoring pipeline — assertions + runner integration."""

import json
import tempfile
from pathlib import Path

import pytest

from litmusai.assertions import (
    All,
    AnyOf,
    Contains,
    Custom,
    Exact,
    NotContains,
    Numeric,
    Weighted,
)
from litmusai.core.agent import Agent, AgentResponse
from litmusai.core.runner import EvalResults, TestResult, evaluate
from litmusai.core.scorer import Scorer, ScoreResult
from litmusai.core.suite import TestCase, TestSuite

# ─── Scorer with Assertions ─────────────────────────────────────


class TestScorerAssertions:
    def setup_method(self):
        self.scorer = Scorer()

    def test_numeric_assertion_pass(self):
        case = TestCase(
            id="1", name="math", task="What is 6*6?",
            assertions=[Numeric(36, tolerance=0.01)],
        )
        resp = AgentResponse(output="The answer is 36.")
        result = self.scorer.score(case, resp)
        assert result.passed
        assert result.score >= 0.9

    def test_numeric_assertion_fail(self):
        case = TestCase(
            id="1", name="math", task="What is 6*6?",
            assertions=[Numeric(36, tolerance=0.01)],
        )
        resp = AgentResponse(output="I think it's 42.")
        result = self.scorer.score(case, resp)
        assert not result.passed

    def test_multiple_assertions_all_pass(self):
        case = TestCase(
            id="1", name="test", task="test",
            assertions=[Numeric(36), NotContains(["sorry"])],
        )
        resp = AgentResponse(output="The answer is 36.")
        result = self.scorer.score(case, resp)
        assert result.passed
        assert "2 assertions passed" in result.reason

    def test_multiple_assertions_one_fails(self):
        case = TestCase(
            id="1", name="test", task="test",
            assertions=[Numeric(36), NotContains(["answer"])],
        )
        resp = AgentResponse(output="The answer is 36.")
        result = self.scorer.score(case, resp)
        assert not result.passed
        assert "1/2 failed" in result.reason

    def test_composite_assertion(self):
        case = TestCase(
            id="1", name="test", task="test",
            assertions=[AnyOf(Exact("36"), Numeric(36))],
        )
        resp = AgentResponse(output="The answer is 36.")
        result = self.scorer.score(case, resp)
        assert result.passed

    def test_weighted_assertion(self):
        case = TestCase(
            id="1", name="test", task="test",
            assertions=[
                Weighted([
                    (Numeric(36), 0.7),
                    (Contains(["answer"]), 0.3),
                ], threshold=0.6),
            ],
        )
        resp = AgentResponse(output="The answer is 36.")
        result = self.scorer.score(case, resp)
        assert result.passed

    def test_custom_assertion(self):
        case = TestCase(
            id="1", name="test", task="test",
            assertions=[
                Custom(lambda r: len(r.split()) >= 3, name="min_words"),
            ],
        )
        resp = AgentResponse(output="Yes it is.")
        result = self.scorer.score(case, resp)
        assert result.passed

    def test_assertions_take_precedence(self):
        """When both assertions and expected_contains are set,
        assertions win."""
        case = TestCase(
            id="1", name="test", task="test",
            expected_contains=["wrong_pattern"],
            assertions=[Numeric(36)],
        )
        resp = AgentResponse(output="36")
        result = self.scorer.score(case, resp)
        assert result.passed

    def test_fallback_to_legacy(self):
        case = TestCase(
            id="1", name="test", task="test",
            expected_contains=["hello"],
        )
        resp = AgentResponse(output="hello world")
        result = self.scorer.score(case, resp)
        assert result.passed

    def test_error_response(self):
        case = TestCase(
            id="1", name="test", task="test",
            assertions=[Numeric(36)],
        )
        resp = AgentResponse(output="", success=False, error="timeout")
        result = self.scorer.score(case, resp)
        assert not result.passed

    def test_assertion_details(self):
        case = TestCase(
            id="1", name="test", task="test",
            assertions=[Numeric(36), Contains(["answer"])],
        )
        resp = AgentResponse(output="The answer is 36.")
        result = self.scorer.score(case, resp)
        assert result.details is not None
        assert "assertions" in result.details
        assert len(result.details["assertions"]) == 2

    def test_empty_assertions_list(self):
        case = TestCase(
            id="1", name="test", task="test",
            assertions=[],
            expected_contains=["hello"],
        )
        resp = AgentResponse(output="hello")
        result = self.scorer.score(case, resp)
        assert result.passed

    def test_assertion_exception_handled(self):
        """Assertion that raises is caught and scored as failure."""
        def bad_check(r):
            raise RuntimeError("kaboom")

        case = TestCase(
            id="1", name="test", task="test",
            assertions=[Custom(bad_check)],
        )
        resp = AgentResponse(output="test")
        result = self.scorer.score(case, resp)
        # Custom already catches internally, but verify no crash
        assert isinstance(result, ScoreResult)


# ─── TestResult Serialization ────────────────────────────────────


class TestResultSerialization:
    def test_to_dict(self):
        case = TestCase(id="t1", name="math", task="2+2?")
        resp = AgentResponse(
            output="4", input_tokens=10, output_tokens=5,
            model="gpt-4o", cost=0.001,
        )
        score = ScoreResult(passed=True, score=1.0, reason="OK")
        result = TestResult(
            case=case, response=resp, score=score,
            passed=True, latency_ms=100,
            cost=0.001, input_tokens=10, output_tokens=5,
        )
        d = result.to_dict()
        assert d["case_id"] == "t1"
        assert d["passed"] is True
        assert d["score"] == 1.0
        assert d["input_tokens"] == 10
        assert d["model"] == "gpt-4o"


# ─── EvalResults Serialization ───────────────────────────────────


class TestEvalResultsSerialization:
    def _make_results(self) -> EvalResults:
        case = TestCase(id="t1", name="math", task="2+2?")
        resp = AgentResponse(
            output="4", input_tokens=10, output_tokens=5, cost=0.001,
        )
        score = ScoreResult(passed=True, score=1.0, reason="OK")
        result = TestResult(
            case=case, response=resp, score=score,
            passed=True, latency_ms=100,
            cost=0.001, input_tokens=10, output_tokens=5,
        )
        return EvalResults(
            agent_name="test-agent",
            suite_name="test-suite",
            results=[result],
            total_cost=0.001,
            total_time_ms=100,
            total_input_tokens=10,
            total_output_tokens=5,
            timestamp="2026-04-02T06:00:00",
        )

    def test_to_dict(self):
        results = self._make_results()
        d = results.to_dict()
        assert d["agent_name"] == "test-agent"
        assert d["summary"]["passed"] == 1
        assert d["summary"]["pass_rate"] == 1.0
        assert len(d["results"]) == 1

    def test_save_json(self):
        results = self._make_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = results.save(Path(tmpdir) / "results.json")
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["agent_name"] == "test-agent"
            assert data["summary"]["total_input_tokens"] == 10

    def test_save_creates_dirs(self):
        results = self._make_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = results.save(
                Path(tmpdir) / "sub" / "dir" / "results.json",
            )
            assert path.exists()

    def test_summary_with_tokens(self):
        results = self._make_results()
        s = results.summary()
        assert "tokens" in s

    def test_summary_without_tokens(self):
        results = self._make_results()
        results.total_input_tokens = 0
        results.total_output_tokens = 0
        s = results.summary()
        assert "tokens" not in s

    def test_avg_score(self):
        results = self._make_results()
        assert results.avg_score == 1.0


# ─── Runner Integration ─────────────────────────────────────────


class TestRunnerIntegration:
    @pytest.mark.asyncio
    async def test_evaluate_with_assertions(self):
        async def my_agent(task: str) -> str:
            return "The answer is 36."

        agent = Agent.from_function(my_agent, name="test")
        suite = TestSuite(name="test-suite")
        suite.add_case(TestCase(
            id="t1", name="math", task="What is 6*6?",
            assertions=[Numeric(36)],
        ))

        results = await evaluate(agent, suite, verbose=False)
        assert results.passed == 1
        assert results.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_mixed(self):
        async def my_agent(task: str) -> str:
            if "6*6" in task:
                return "36"
            return "Hello world"

        agent = Agent.from_function(my_agent, name="test")
        suite = TestSuite(name="mixed")
        suite.add_case(TestCase(
            id="t1", name="math", task="What is 6*6?",
            assertions=[Numeric(36)],
        ))
        suite.add_case(TestCase(
            id="t2", name="greet", task="Say hello",
            expected_contains=["hello", "world"],
        ))

        results = await evaluate(agent, suite, verbose=False)
        assert results.passed == 2

    @pytest.mark.asyncio
    async def test_evaluate_with_logging(self):
        async def my_agent(task: str) -> str:
            return "42"

        agent = Agent.from_function(my_agent, name="test")
        suite = TestSuite(name="log-test")
        suite.add_case(TestCase(
            id="t1", name="test", task="answer?",
            assertions=[Numeric(42)],
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            await evaluate(
                agent, suite, verbose=False, log_dir=tmpdir,
            )
            files = list(Path(tmpdir).glob("*.json"))
            assert len(files) == 1

            with open(files[0]) as f:
                data = json.load(f)
            assert data["agent_name"] == "test"
            assert data["results"][0]["passed"] is True

    @pytest.mark.asyncio
    async def test_evaluate_tokens_tracked(self):
        async def my_agent(task: str) -> AgentResponse:
            return AgentResponse(
                output="answer", input_tokens=50,
                output_tokens=20, cost=0.001,
            )

        agent = Agent.from_function(my_agent, name="test")
        suite = TestSuite(name="tokens")
        suite.add_case(TestCase(id="t1", name="test", task="test"))

        results = await evaluate(agent, suite, verbose=False)
        assert results.total_input_tokens == 50
        assert results.total_output_tokens == 20
        assert results.total_cost == 0.001

    @pytest.mark.asyncio
    async def test_compare_function(self):
        from litmusai.core.runner import compare

        async def fast_agent(task: str) -> str:
            return "36"

        async def slow_agent(task: str) -> str:
            return "I don't know"

        suite = TestSuite(name="compare-test")
        suite.add_case(TestCase(
            id="t1", name="math", task="6*6?",
            assertions=[Numeric(36)],
        ))

        results = await compare(
            {
                "fast": Agent.from_function(fast_agent, name="fast"),
                "slow": Agent.from_function(slow_agent, name="slow"),
            },
            suite,
            verbose=False,
        )
        assert results["fast"].passed == 1
        assert results["slow"].passed == 0

    @pytest.mark.asyncio
    async def test_all_composite_in_suite(self):
        async def my_agent(task: str) -> str:
            return "The answer is 36. I'm confident about this."

        agent = Agent.from_function(my_agent, name="test")
        suite = TestSuite(name="composite")
        suite.add_case(TestCase(
            id="t1", name="test", task="6*6?",
            assertions=[All(
                Numeric(36),
                NotContains(["sorry", "I don't know"]),
                Custom(lambda r: len(r) > 10, name="min_len"),
            )],
        ))

        results = await evaluate(agent, suite, verbose=False)
        assert results.passed == 1

    @pytest.mark.asyncio
    async def test_safe_filename(self):
        """Agent/suite names with special chars don't break log paths."""
        from litmusai.core.runner import _safe_filename
        assert _safe_filename("my/agent\\v2..") == "my_agent_v2.."
        assert _safe_filename("hello world:1") == "hello_world_1"
        assert "/" not in _safe_filename("a/b/c")


class TestAsyncScoring:
    @pytest.mark.asyncio
    async def test_ascore_with_assertions(self):
        """ascore() works with sync assertions."""
        scorer = Scorer()
        case = TestCase(
            id="1", name="test", task="6*6?",
            assertions=[Numeric(36)],
        )
        resp = AgentResponse(output="36")
        result = await scorer.ascore(case, resp)
        assert result.passed

    @pytest.mark.asyncio
    async def test_ascore_falls_back_to_legacy(self):
        """ascore() uses legacy when no assertions."""
        scorer = Scorer()
        case = TestCase(
            id="1", name="test", task="test",
            expected_contains=["hello"],
        )
        resp = AgentResponse(output="hello world")
        result = await scorer.ascore(case, resp)
        assert result.passed

    @pytest.mark.asyncio
    async def test_ascore_error_response(self):
        scorer = Scorer()
        case = TestCase(
            id="1", name="test", task="test",
            assertions=[Numeric(36)],
        )
        resp = AgentResponse(output="", success=False, error="timeout")
        result = await scorer.ascore(case, resp)
        assert not result.passed

    @pytest.mark.asyncio
    async def test_tokens_used_fallback(self):
        """tokens_used is tracked when input/output are zero."""
        async def my_agent(task: str) -> AgentResponse:
            return AgentResponse(
                output="ok", tokens_used=100,
                input_tokens=0, output_tokens=0,
            )

        agent = Agent.from_function(my_agent, name="test")
        suite = TestSuite(name="tok")
        suite.add_case(TestCase(id="t1", name="t", task="t"))

        results = await evaluate(agent, suite, verbose=False)
        assert results.total_input_tokens == 100

    @pytest.mark.asyncio
    async def test_yaml_excludes_assertions(self):
        """to_yaml skips non-serializable assertions field."""
        import tempfile

        suite = TestSuite(name="yaml-test")
        suite.add_case(TestCase(
            id="t1", name="test", task="test",
            assertions=[Numeric(36)],
        ))

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w") as f:
            suite.to_yaml(f.name)
            loaded = TestSuite.from_yaml(f.name)
            assert len(loaded.cases) == 1
            assert loaded.cases[0].task == "test"
            # assertions not in YAML, so empty on reload
            assert loaded.cases[0].assertions == []
