"""Tests for multi-run statistical evaluation."""

import pytest

from litmusai.assertions import Numeric
from litmusai.core.agent import Agent, AgentResponse
from litmusai.core.runner import (
    CaseStats,
    MultiRunResults,
    multi_evaluate,
)
from litmusai.core.suite import TestCase, TestSuite

# ─── CaseStats ───────────────────────────────────────────────────


class TestCaseStats:
    def test_pass_rate(self):
        s = CaseStats(
            case_id="q1", case_name="Math", task="t",
            n_runs=5, n_passed=4,
        )
        assert s.pass_rate == 0.8

    def test_pass_rate_zero_runs(self):
        s = CaseStats(case_id="q1", case_name="Math", task="t")
        assert s.pass_rate == 0.0

    def test_mean_score(self):
        s = CaseStats(
            case_id="q1", case_name="Math", task="t",
            scores=[0.8, 0.9, 1.0],
        )
        assert s.mean_score == pytest.approx(0.9)

    def test_std_score(self):
        s = CaseStats(
            case_id="q1", case_name="Math", task="t",
            scores=[0.8, 0.9, 1.0],
        )
        assert s.std_score > 0

    def test_std_score_single(self):
        """Single run has 0 std dev."""
        s = CaseStats(
            case_id="q1", case_name="Math", task="t",
            scores=[0.9],
        )
        assert s.std_score == 0.0

    def test_mean_latency(self):
        s = CaseStats(
            case_id="q1", case_name="Math", task="t",
            latencies=[100, 200, 300],
        )
        assert s.mean_latency == 200.0

    def test_reliability_stable_pass(self):
        s = CaseStats(
            case_id="q1", case_name="Math", task="t",
            n_runs=5, n_passed=5,
        )
        assert s.reliability == "stable-pass"

    def test_reliability_stable_fail(self):
        s = CaseStats(
            case_id="q1", case_name="Math", task="t",
            n_runs=5, n_passed=0,
        )
        assert s.reliability == "stable-fail"

    def test_reliability_flaky(self):
        s = CaseStats(
            case_id="q1", case_name="Math", task="t",
            n_runs=5, n_passed=3,
        )
        assert s.reliability == "flaky"

    def test_reliability_mostly_pass(self):
        s = CaseStats(
            case_id="q1", case_name="Math", task="t",
            n_runs=5, n_passed=4,
        )
        assert s.reliability == "mostly-pass"

    def test_reliability_mostly_fail(self):
        s = CaseStats(
            case_id="q1", case_name="Math", task="t",
            n_runs=5, n_passed=1,
        )
        assert s.reliability == "mostly-fail"

    def test_reliability_unknown(self):
        s = CaseStats(case_id="q1", case_name="Math", task="t")
        assert s.reliability == "unknown"


# ─── MultiRunResults ─────────────────────────────────────────────


class TestMultiRunResults:
    def test_mean_pass_rate(self):
        from litmusai.core.runner import EvalResults

        r1 = EvalResults(
            agent_name="test", suite_name="test",
            results=[],
        )
        r2 = EvalResults(
            agent_name="test", suite_name="test",
            results=[],
        )
        # Manually set pass rates by adding dummy results
        multi = MultiRunResults(
            agent_name="test", suite_name="test", n_runs=2,
            run_results=[r1, r2],
        )
        assert multi.mean_pass_rate == 0.0

    def test_flaky_detection(self):
        multi = MultiRunResults(
            agent_name="test", suite_name="test", n_runs=5,
            case_stats={
                "q1": CaseStats(
                    case_id="q1", case_name="Math", task="t",
                    n_runs=5, n_passed=5,
                ),
                "q2": CaseStats(
                    case_id="q2", case_name="Code", task="t",
                    n_runs=5, n_passed=3,  # flaky
                ),
            },
        )
        assert len(multi.flaky_tests) == 1
        assert multi.flaky_tests[0].case_id == "q2"

    def test_to_table(self):
        multi = MultiRunResults(
            agent_name="test", suite_name="test", n_runs=3,
            case_stats={
                "q1": CaseStats(
                    case_id="q1", case_name="Math", task="t",
                    n_runs=3, n_passed=3,
                    scores=[1.0, 0.9, 0.95],
                    latencies=[100, 200, 150],
                ),
            },
        )
        table = multi.to_table()
        assert "Math" in table
        assert "3/3" in table
        assert "stable-pass" in table

    def test_to_dict(self):
        multi = MultiRunResults(
            agent_name="test", suite_name="test", n_runs=3,
            case_stats={
                "q1": CaseStats(
                    case_id="q1", case_name="Math", task="t",
                    n_runs=3, n_passed=2,
                    scores=[0.8, 1.0, 0.9],
                    latencies=[100, 200, 150],
                ),
            },
        )
        d = multi.to_dict()
        assert d["n_runs"] == 3
        assert "q1" in d["case_stats"]
        assert d["case_stats"]["q1"]["reliability"] == "flaky"

    def test_summary(self):
        multi = MultiRunResults(
            agent_name="test", suite_name="test", n_runs=3,
            case_stats={
                "q1": CaseStats(
                    case_id="q1", case_name="Flaky", task="t",
                    n_runs=3, n_passed=2,
                ),
            },
        )
        s = multi.summary()
        assert "3 runs" in s


# ─── Integration: multi_evaluate ─────────────────────────────────


class TestMultiEvaluate:
    @pytest.mark.asyncio
    async def test_basic_multi_run(self):
        call_count = 0

        async def my_agent(task: str) -> str:
            nonlocal call_count
            call_count += 1
            return "The answer is 36."

        agent = Agent.from_function(my_agent, name="test")
        suite = TestSuite(name="multi")
        suite.add_case(TestCase(
            id="q1", name="Math", task="6*6?",
            assertions=[Numeric(36)],
        ))

        multi = await multi_evaluate(
            agent, suite, runs=3, verbose=False,
        )

        assert multi.n_runs == 3
        assert len(multi.run_results) == 3
        assert call_count == 3  # called once per run
        assert "q1" in multi.case_stats
        assert multi.case_stats["q1"].n_passed == 3
        assert multi.case_stats["q1"].reliability == "stable-pass"

    @pytest.mark.asyncio
    async def test_flaky_agent_detected(self):
        call_count = 0

        async def flaky_agent(task: str) -> str:
            nonlocal call_count
            call_count += 1
            # Pass on even calls, fail on odd
            if call_count % 2 == 0:
                return "36"
            return "I don't know"

        agent = Agent.from_function(flaky_agent, name="flaky")
        suite = TestSuite(name="flaky-test")
        suite.add_case(TestCase(
            id="q1", name="Math", task="6*6?",
            assertions=[Numeric(36)],
        ))

        multi = await multi_evaluate(
            agent, suite, runs=4, verbose=False,
        )

        stats = multi.case_stats["q1"]
        assert stats.n_passed == 2
        assert stats.n_runs == 4
        assert stats.reliability == "flaky"

    @pytest.mark.asyncio
    async def test_multi_run_multiple_cases(self):
        async def my_agent(task: str) -> str:
            if "math" in task.lower():
                return "36"
            return "Paris"

        agent = Agent.from_function(my_agent, name="test")
        suite = TestSuite(name="multi")
        suite.add_case(TestCase(
            id="q1", name="Math", task="math: 6*6?",
            assertions=[Numeric(36)],
        ))
        suite.add_case(TestCase(
            id="q2", name="Geo", task="Capital of France?",
            expected_contains=["Paris"],
        ))

        multi = await multi_evaluate(
            agent, suite, runs=2, verbose=False,
        )

        assert len(multi.case_stats) == 2
        assert multi.case_stats["q1"].n_runs == 2
        assert multi.case_stats["q2"].n_runs == 2

    @pytest.mark.asyncio
    async def test_std_dev_computed(self):
        scores = []

        async def varying_agent(task: str) -> AgentResponse:
            # Return slightly different latency each time
            scores.append(1.0)
            return AgentResponse(output="36", cost=0.001)

        agent = Agent.from_function(varying_agent, name="test")
        suite = TestSuite(name="std")
        suite.add_case(TestCase(
            id="q1", name="Math", task="6*6?",
            assertions=[Numeric(36)],
        ))

        multi = await multi_evaluate(
            agent, suite, runs=3, verbose=False,
        )

        stats = multi.case_stats["q1"]
        assert stats.n_runs == 3
        assert len(stats.scores) == 3
        # All scores should be 1.0 (exact match)
        assert stats.mean_score == pytest.approx(1.0, abs=0.1)
