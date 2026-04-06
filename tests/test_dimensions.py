"""Tests for multi-dimensional scoring (#34)."""

from __future__ import annotations

import pytest

from litmusai.scoring import (
    DEFAULT_WEIGHTS,
    DIMENSIONS,
    DimensionBudget,
    ScoreVector,
    aggregate_vectors,
    build_score_vector,
)


class TestScoreVector:
    def test_default_values(self):
        v = ScoreVector()
        for dim in DIMENSIONS:
            assert getattr(v, dim) == 0.0
        assert v.overall == 0.0

    def test_compute_overall_defaults(self):
        v = ScoreVector(
            correctness=1.0,
            completeness=0.8,
            format=0.9,
            relevance=0.7,
            safety=1.0,
            latency=0.5,
            cost=0.6,
        )
        result = v.compute_overall()
        assert 0.0 <= result <= 1.0
        assert v.overall == result

        # Manual calculation
        expected = (
            1.0 * 0.40 + 0.8 * 0.20 + 0.9 * 0.10
            + 0.7 * 0.10 + 1.0 * 0.10 + 0.5 * 0.05
            + 0.6 * 0.05
        )
        assert abs(result - expected) < 0.001

    def test_compute_overall_custom_weights(self):
        v = ScoreVector(correctness=1.0, safety=0.0)
        v.compute_overall({"correctness": 1.0, "safety": 1.0})
        # Only correctness=1.0 and safety=0.0, rest are 0
        # Weights normalized: each gets 0.5 after normalization
        # (but all OTHER dimensions also have weights from defaults)
        assert 0.0 < v.overall < 1.0

    def test_compute_overall_all_ones(self):
        v = ScoreVector(
            correctness=1.0, completeness=1.0, format=1.0,
            relevance=1.0, safety=1.0, latency=1.0, cost=1.0,
        )
        v.compute_overall()
        assert abs(v.overall - 1.0) < 0.001

    def test_compute_overall_all_zeros(self):
        v = ScoreVector()
        v.compute_overall()
        assert v.overall == 0.0

    def test_to_dict(self):
        v = ScoreVector(correctness=0.9, safety=1.0)
        d = v.to_dict()
        assert d["correctness"] == 0.9
        assert d["safety"] == 1.0
        assert "details" not in d  # empty dict removed

    def test_to_dict_with_details(self):
        v = ScoreVector(
            correctness=0.9,
            details={"correctness": "2/2 passed"},
        )
        d = v.to_dict()
        assert d["details"]["correctness"] == "2/2 passed"

    def test_from_dict(self):
        d = {"correctness": 0.9, "safety": 1.0, "overall": 0.5}
        v = ScoreVector.from_dict(d)
        assert v.correctness == 0.9
        assert v.safety == 1.0
        assert v.overall == 0.5

    def test_from_dict_ignores_unknown(self):
        d = {"correctness": 0.9, "unknown_field": 42}
        v = ScoreVector.from_dict(d)
        assert v.correctness == 0.9

    def test_dimension_table(self):
        v = ScoreVector(
            correctness=0.9, safety=1.0,
            details={"correctness": "Good"},
        )
        table = v.dimension_table()
        assert len(table) == 7
        assert table[0] == ("correctness", 0.9, "Good")
        assert table[4] == ("safety", 1.0, "")


class TestDimensionBudget:
    def test_score_latency_under_budget(self):
        b = DimensionBudget(latency_ms=2000)
        assert b.score_latency(1000) == 1.0

    def test_score_latency_at_budget(self):
        b = DimensionBudget(latency_ms=2000)
        assert b.score_latency(2000) == 1.0

    def test_score_latency_over_max(self):
        b = DimensionBudget(latency_ms=2000, latency_max_ms=10000)
        assert b.score_latency(10000) == 0.0
        assert b.score_latency(15000) == 0.0

    def test_score_latency_interpolated(self):
        b = DimensionBudget(latency_ms=2000, latency_max_ms=10000)
        score = b.score_latency(6000)
        assert abs(score - 0.5) < 0.01

    def test_score_cost_under_budget(self):
        b = DimensionBudget(cost_usd=0.01)
        assert b.score_cost(0.005) == 1.0

    def test_score_cost_over_max(self):
        b = DimensionBudget(cost_usd=0.01, cost_max_usd=0.10)
        assert b.score_cost(0.10) == 0.0

    def test_score_cost_interpolated(self):
        b = DimensionBudget(cost_usd=0.01, cost_max_usd=0.10)
        score = b.score_cost(0.055)
        assert abs(score - 0.5) < 0.01


class TestBuildScoreVector:
    def test_from_score_result(self):
        from litmusai.core.agent import AgentResponse
        from litmusai.core.scorer import ScoreResult

        sr = ScoreResult(passed=True, score=0.9, reason="Good")
        resp = AgentResponse(
            output="42", model="test",
            latency_ms=1500, cost=0.005,
        )

        v = build_score_vector(score_result=sr, response=resp)
        assert v.correctness == 0.9
        assert v.latency == 1.0  # under 2000ms budget
        assert v.cost == 1.0  # under $0.01 budget
        assert v.overall > 0

    def test_custom_budget(self):
        from litmusai.core.agent import AgentResponse
        from litmusai.core.scorer import ScoreResult

        sr = ScoreResult(passed=True, score=1.0, reason="OK")
        resp = AgentResponse(
            output="x" * 100, model="test",
            latency_ms=5000, cost=0.05,
        )

        budget = DimensionBudget(latency_ms=1000, cost_usd=0.001)
        v = build_score_vector(
            score_result=sr, response=resp, budget=budget,
        )
        assert v.latency < 1.0  # over budget
        assert v.cost < 1.0  # over budget

    def test_with_assertion_details(self):
        from litmusai.core.agent import AgentResponse
        from litmusai.core.scorer import ScoreResult

        sr = ScoreResult(
            passed=True, score=1.0, reason="OK",
            details={
                "assertions": [
                    {"type": "contains", "passed": True, "score": 1.0, "reason": "OK"},
                    {"type": "jsonvalid", "passed": True, "score": 1.0, "reason": "Valid JSON"},
                    {"type": "semantic", "passed": True, "score": 0.9, "reason": "Similar"},
                ],
            },
        )
        resp = AgentResponse(output="{'key': 'value'}", model="test")

        v = build_score_vector(score_result=sr, response=resp)
        assert v.format == 1.0  # jsonvalid passed
        assert v.relevance == 0.9  # semantic score
        assert v.correctness == 1.0  # contains passed

    def test_zero_cost_defaults_to_one(self):
        from litmusai.core.agent import AgentResponse
        from litmusai.core.scorer import ScoreResult

        sr = ScoreResult(passed=True, score=1.0, reason="OK")
        resp = AgentResponse(output="hello", model="test", cost=0.0)

        v = build_score_vector(score_result=sr, response=resp)
        assert v.cost == 1.0  # No cost data → 1.0

    def test_empty_output(self):
        from litmusai.core.agent import AgentResponse
        from litmusai.core.scorer import ScoreResult

        sr = ScoreResult(passed=False, score=0.0, reason="Empty")
        resp = AgentResponse(output="", model="test")

        v = build_score_vector(score_result=sr, response=resp)
        assert v.completeness == 0.0
        assert v.overall < 0.5


class TestAggregateVectors:
    def test_aggregate_two(self):
        v1 = ScoreVector(correctness=1.0, safety=0.8)
        v2 = ScoreVector(correctness=0.6, safety=1.0)
        avg = aggregate_vectors([v1, v2])
        assert abs(avg.correctness - 0.8) < 0.001
        assert abs(avg.safety - 0.9) < 0.001
        assert avg.overall > 0  # computed

    def test_aggregate_empty(self):
        avg = aggregate_vectors([])
        assert avg.correctness == 0.0
        assert avg.overall == 0.0

    def test_aggregate_single(self):
        v = ScoreVector(correctness=0.9, completeness=0.7)
        v.compute_overall()
        avg = aggregate_vectors([v])
        assert avg.correctness == 0.9
        assert avg.completeness == 0.7


class TestEvalResultsDimensions:
    @pytest.mark.asyncio
    async def test_results_have_dimensions(self):
        from litmusai import Agent, TestCase, TestSuite, evaluate
        from litmusai.core.agent import AgentResponse

        async def fn(task, **kw):
            return AgentResponse(
                output="42",
                model="test",
                latency_ms=500,
                cost=0.001,
            )

        agent = Agent(fn=fn, name="test", model="test")
        suite = TestSuite(name="test")
        suite.add_case(TestCase(
            id="q1", name="Q", task="What?",
            expected_contains=["42"],
        ))

        results = await evaluate(agent, suite, verbose=False)
        assert results.results[0].dimensions is not None
        assert results.results[0].dimensions.correctness > 0

    @pytest.mark.asyncio
    async def test_avg_dimensions(self):
        from litmusai import Agent, TestCase, TestSuite, evaluate
        from litmusai.core.agent import AgentResponse

        async def fn(task, **kw):
            return AgentResponse(output="Paris", model="test")

        agent = Agent(fn=fn, name="test", model="test")
        suite = TestSuite(name="test")
        suite.add_case(TestCase(
            id="q1", name="Q1", task="Capital?",
            expected_contains=["Paris"],
        ))
        suite.add_case(TestCase(
            id="q2", name="Q2", task="City?",
            expected_contains=["Paris"],
        ))

        results = await evaluate(agent, suite, verbose=False)
        avg = results.avg_dimensions
        assert avg is not None
        assert avg.correctness > 0

    @pytest.mark.asyncio
    async def test_dimensions_in_to_dict(self):
        from litmusai import Agent, TestCase, TestSuite, evaluate
        from litmusai.core.agent import AgentResponse

        async def fn(task, **kw):
            return AgentResponse(output="42", model="test")

        agent = Agent(fn=fn, name="test", model="test")
        suite = TestSuite(name="test")
        suite.add_case(TestCase(
            id="q1", name="Q", task="What?",
            expected_contains=["42"],
        ))

        results = await evaluate(agent, suite, verbose=False)
        d = results.to_dict()
        assert "dimensions" in d
        assert "correctness" in d["dimensions"]
        assert "overall" in d["dimensions"]

    @pytest.mark.asyncio
    async def test_dimensions_in_result_to_dict(self):
        from litmusai import Agent, TestCase, TestSuite, evaluate
        from litmusai.core.agent import AgentResponse

        async def fn(task, **kw):
            return AgentResponse(output="42", model="test")

        agent = Agent(fn=fn, name="test", model="test")
        suite = TestSuite(name="test")
        suite.add_case(TestCase(
            id="q1", name="Q", task="What?",
            expected_contains=["42"],
        ))

        results = await evaluate(agent, suite, verbose=False)
        rd = results.results[0].to_dict()
        assert "dimensions" in rd
        assert rd["dimensions"]["correctness"] > 0


class TestDimensionsCLI:
    def test_dimensions_flag(self):
        from click.testing import CliRunner

        from litmusai.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert "--dimensions" in result.output

    def test_format_table_with_dimensions(self, capsys):
        from litmusai.ci import format_table

        data = {
            "suite": "test",
            "summary": {
                "total": 1, "passed": 1, "failed": 0,
                "pass_rate": 1.0, "avg_score": 1.0,
                "avg_latency_ms": 500, "total_cost": 0.001,
            },
            "results": [{
                "test": "Q1", "passed": True, "score": 1.0,
                "latency_ms": 500, "cost": 0.001,
                "dimensions": {
                    "correctness": 1.0, "completeness": 0.8,
                    "format": 0.9, "relevance": 0.7, "safety": 1.0,
                },
            }],
            "dimensions": {
                "correctness": 1.0, "completeness": 0.8,
                "format": 0.9, "relevance": 0.7, "safety": 1.0,
                "latency": 0.9, "cost": 0.95, "overall": 0.91,
            },
        }
        format_table(data, show_dimensions=True)
        captured = capsys.readouterr()
        # Verify dimension columns rendered
        assert "Corr" in captured.out
        assert "1.00" in captured.out
        assert "overall=0.91" in captured.out

    def test_format_table_without_dimensions(self):
        from litmusai.ci import format_table

        data = {
            "suite": "test",
            "summary": {
                "total": 1, "passed": 1, "failed": 0,
                "pass_rate": 1.0, "avg_score": 1.0,
                "avg_latency_ms": 500, "total_cost": 0.001,
            },
            "results": [{
                "test": "Q1", "passed": True, "score": 1.0,
                "latency_ms": 500, "cost": 0.001,
            }],
        }
        # Should not raise (no dimensions)
        format_table(data, show_dimensions=False)


class TestHtmlReportDimensions:
    def test_html_report_with_dimensions(self, tmp_path):
        from litmusai.reports import render_html

        data = {
            "agent_name": "test",
            "suite_name": "test",
            "timestamp": "2026-04-05T00:00:00",
            "summary": {
                "total": 1, "passed": 1, "failed": 0,
                "pass_rate": 1.0, "avg_score": 1.0,
                "avg_latency_ms": 500, "total_cost": 0.001,
                "total_input_tokens": 100,
                "total_output_tokens": 50,
            },
            "results": [{
                "case_id": "q1", "test": "Q1", "task": "What?",
                "response": "42", "passed": True, "score": 1.0,
                "latency_ms": 500, "cost": 0.001,
            }],
            "dimensions": {
                "correctness": 1.0, "completeness": 0.8,
                "format": 0.9, "relevance": 0.7, "safety": 1.0,
                "latency": 0.9, "cost": 0.95, "overall": 0.91,
            },
        }

        path = render_html(data, tmp_path / "report.html")
        content = path.read_text()
        assert "Quality Dimensions" in content
        assert "Correctness" in content
        assert "radar" not in content.lower() or "polygon" in content.lower()
        # SVG radar chart
        assert "<svg" in content
        assert "polygon" in content

    def test_html_report_without_dimensions(self, tmp_path):
        from litmusai.reports import render_html

        data = {
            "agent_name": "test",
            "suite_name": "test",
            "timestamp": "2026-04-05T00:00:00",
            "summary": {
                "total": 1, "passed": 1, "failed": 0,
                "pass_rate": 1.0, "avg_score": 1.0,
                "avg_latency_ms": 500, "total_cost": 0.001,
                "total_input_tokens": 100,
                "total_output_tokens": 50,
            },
            "results": [{
                "case_id": "q1", "test": "Q1", "task": "What?",
                "response": "42", "passed": True, "score": 1.0,
                "latency_ms": 500, "cost": 0.001,
            }],
        }

        path = render_html(data, tmp_path / "report.html")
        content = path.read_text()
        assert "Quality Dimensions" not in content


class TestDefaultWeights:
    def test_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_all_dimensions_covered(self):
        assert set(DIMENSIONS) == set(DEFAULT_WEIGHTS.keys())
        assert len(DIMENSIONS) == 7


class TestEdgeCases:
    def test_budget_zero_range_no_crash(self):
        """DimensionBudget with max <= target should not crash."""
        b = DimensionBudget(
            latency_ms=2000, latency_max_ms=2000,
            cost_usd=0.01, cost_max_usd=0.01,
        )
        assert b.score_latency(3000) == 0.0
        assert b.score_cost(0.05) == 0.0

    def test_budget_inverted_range_no_crash(self):
        b = DimensionBudget(
            latency_ms=5000, latency_max_ms=1000,
        )
        # max < target means any value over target returns 0.0
        # but 3000 < 5000 (target), so it's still 1.0
        assert b.score_latency(3000) == 1.0
        # Value above target hits the guard
        assert b.score_latency(6000) == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_with_custom_budget(self):
        from litmusai import Agent, TestCase, TestSuite, evaluate
        from litmusai.core.agent import AgentResponse

        async def fn(task, **kw):
            return AgentResponse(
                output="42", model="test",
                cost=0.05,  # over budget
            )

        agent = Agent(fn=fn, name="test", model="test")
        suite = TestSuite(name="test")
        suite.add_case(TestCase(
            id="q1", name="Q", task="What?",
            expected_contains=["42"],
        ))

        budget = DimensionBudget(cost_usd=0.001, cost_max_usd=0.10)
        results = await evaluate(
            agent, suite, verbose=False, dimension_budget=budget,
        )
        dims = results.results[0].dimensions
        assert dims is not None
        assert dims.cost < 1.0  # cost over budget


class TestResultsToDictDimensions:
    @pytest.mark.asyncio
    async def test_results_to_dict_includes_dimensions(self):
        from litmusai import Agent, TestCase, TestSuite, evaluate
        from litmusai.ci import results_to_dict
        from litmusai.core.agent import AgentResponse

        async def fn(task, **kw):
            return AgentResponse(output="42", model="test")

        agent = Agent(fn=fn, name="test", model="test")
        suite = TestSuite(name="test")
        suite.add_case(TestCase(
            id="q1", name="Q", task="What?",
            expected_contains=["42"],
        ))

        results = await evaluate(agent, suite, verbose=False)
        d = results_to_dict(results)
        assert "dimensions" in d
        assert "dimensions" in d["results"][0]
        assert "correctness" in d["results"][0]["dimensions"]
