"""Tests for Pipeline class."""

from __future__ import annotations

from pathlib import Path

import pytest

from litmusai import Pipeline, PipelineResult, run_pipeline
from litmusai.core.agent import Agent, AgentResponse
from litmusai.core.runner import EvalResults, MultiRunResults
from litmusai.core.suite import TestCase, TestSuite

# ─── Helpers ─────────────────────────────────────────────────────


def _make_agent(response: str = "42") -> Agent:
    """Create a mock agent."""
    async def fn(task: str, **kw) -> AgentResponse:
        return AgentResponse(
            output=response,
            model="test-model",
            input_tokens=10,
            output_tokens=5,
            tokens_used=15,
            latency_ms=100.0,
        )
    return Agent(fn=fn, name="test-agent", model="test-model")


def _make_suite() -> TestSuite:
    """Create a simple test suite."""
    suite = TestSuite(name="test-suite")
    suite.add_case(TestCase(
        id="q1", name="Math",
        task="What is 6 * 7?",
        expected_contains=["42"],
    ))
    suite.add_case(TestCase(
        id="q2", name="Greeting",
        task="Say hello",
        expected_contains=["42"],  # will match our mock
    ))
    return suite


# ─── Pipeline Construction ───────────────────────────────────────


class TestPipelineInit:
    def test_basic_construction(self):
        agent = _make_agent()
        suite = _make_suite()
        p = Pipeline(agent, suite)
        assert p.agent is agent
        assert p._suite is suite
        assert p.runs == 1
        assert p.safety is False
        assert p.report is None

    def test_with_all_options(self):
        agent = _make_agent()
        suite = _make_suite()
        p = Pipeline(
            agent, suite,
            safety=True,
            safety_depth="thorough",
            runs=5,
            report="html",
            report_path="out.html",
            threshold=0.9,
            concurrency=3,
            verbose=False,
        )
        assert p.safety is True
        assert p.safety_depth == "thorough"
        assert p.runs == 5
        assert p.report == "html"
        assert p.threshold == 0.9
        assert p.concurrency == 3

    def test_concurrency_minimum_1(self):
        agent = _make_agent()
        suite = _make_suite()
        p = Pipeline(agent, suite, concurrency=0)
        assert p.concurrency == 1
        p = Pipeline(agent, suite, concurrency=-3)
        assert p.concurrency == 1

    def test_string_suite_resolution(self):
        agent = _make_agent()
        p = Pipeline(agent, "coding")
        assert p._suite.name == "coding"
        assert len(p._suite) > 0

    def test_runs_minimum_1(self):
        agent = _make_agent()
        suite = _make_suite()
        p = Pipeline(agent, suite, runs=0)
        assert p.runs == 1
        p = Pipeline(agent, suite, runs=-5)
        assert p.runs == 1

    @pytest.mark.asyncio
    async def test_threshold_controls_passed(self):
        agent = _make_agent()
        suite = _make_suite()

        # Pass rate is 1.0, threshold 0.9 → should pass
        p = Pipeline(agent, suite, threshold=0.9, verbose=False)
        result = await p.run()
        assert result.passed is True
        assert result.threshold == 0.9

        # Threshold above pass rate → should fail
        # (our mock always passes, so use 1.1 which is impossible)
        result2 = await Pipeline(
            agent, suite, threshold=1.1, verbose=False,
        ).run()
        assert result2.passed is False


# ─── Pipeline Run ────────────────────────────────────────────────


class TestPipelineRun:
    @pytest.mark.asyncio
    async def test_basic_run(self):
        agent = _make_agent()
        suite = _make_suite()
        p = Pipeline(agent, suite, verbose=False)
        result = await p.run()

        assert isinstance(result, PipelineResult)
        assert isinstance(result.eval, EvalResults)
        assert result.eval.pass_rate == 1.0
        assert result.safety is None
        assert result.multi_run is None
        assert result.report_path is None
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_multi_run(self):
        agent = _make_agent()
        suite = _make_suite()
        p = Pipeline(agent, suite, runs=3, verbose=False)
        result = await p.run()

        assert result.multi_run is not None
        assert isinstance(result.multi_run, MultiRunResults)
        assert result.multi_run.n_runs == 3
        assert result.eval is not None  # first run

    @pytest.mark.asyncio
    async def test_with_safety(self):
        agent = _make_agent("I cannot help with that.")
        suite = _make_suite()
        p = Pipeline(agent, suite, safety=True, verbose=False)
        result = await p.run()

        assert result.safety is not None
        assert hasattr(result.safety, "safety_score")
        assert hasattr(result.safety, "is_safe")

    @pytest.mark.asyncio
    async def test_html_report(self, tmp_path):
        agent = _make_agent()
        suite = _make_suite()
        report_path = str(tmp_path / "report.html")
        p = Pipeline(
            agent, suite,
            report="html",
            report_path=report_path,
            verbose=False,
        )
        result = await p.run()

        assert result.report_path == report_path
        assert Path(report_path).exists()
        content = Path(report_path).read_text()
        assert "<html" in content

    @pytest.mark.asyncio
    async def test_junit_report(self, tmp_path):
        agent = _make_agent()
        suite = _make_suite()
        report_path = str(tmp_path / "results.xml")
        p = Pipeline(
            agent, suite,
            report="junit",
            report_path=report_path,
            verbose=False,
        )
        result = await p.run()

        assert result.report_path == report_path
        assert Path(report_path).exists()
        content = Path(report_path).read_text()
        assert "testsuite" in content

    @pytest.mark.asyncio
    async def test_csv_report(self, tmp_path):
        agent = _make_agent()
        suite = _make_suite()
        report_path = str(tmp_path / "results.csv")
        p = Pipeline(
            agent, suite,
            report="csv",
            report_path=report_path,
            verbose=False,
        )
        result = await p.run()

        assert result.report_path == report_path
        assert Path(report_path).exists()

    @pytest.mark.asyncio
    async def test_invalid_report_format(self):
        agent = _make_agent()
        suite = _make_suite()
        p = Pipeline(agent, suite, report="pdf", verbose=False)

        with pytest.raises(ValueError, match="Unknown report format"):
            await p.run()

    @pytest.mark.asyncio
    async def test_with_log_dir(self, tmp_path):
        agent = _make_agent()
        suite = _make_suite()
        log_dir = str(tmp_path / "logs")
        p = Pipeline(agent, suite, log_dir=log_dir, verbose=False)
        result = await p.run()

        assert result.eval is not None
        # Log dir should have been created by the runner
        assert Path(log_dir).exists()

    @pytest.mark.asyncio
    async def test_baseline_comparison(self, tmp_path):
        agent = _make_agent()
        suite = _make_suite()

        # First run — save results
        log_dir = str(tmp_path / "logs")
        p1 = Pipeline(agent, suite, log_dir=log_dir, verbose=False)
        await p1.run()

        # Find the saved result file
        log_files = list(Path(log_dir).glob("*.json"))
        assert len(log_files) >= 1

        # Second run with baseline
        p2 = Pipeline(
            agent, suite,
            baseline=str(log_files[0]),
            verbose=False,
        )
        r2 = await p2.run()

        assert r2.baseline_diff is not None


# ─── PipelineResult ──────────────────────────────────────────────


class TestPipelineResult:
    @pytest.mark.asyncio
    async def test_summary(self):
        agent = _make_agent()
        suite = _make_suite()
        p = Pipeline(agent, suite, verbose=False)
        result = await p.run()

        summary = result.summary()
        assert "✅" in summary
        assert "⏱️" in summary

    @pytest.mark.asyncio
    async def test_passed_property(self):
        agent = _make_agent()
        suite = _make_suite()
        p = Pipeline(agent, suite, verbose=False)
        result = await p.run()

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_summary_with_safety(self):
        agent = _make_agent("I cannot help with that.")
        suite = _make_suite()
        p = Pipeline(agent, suite, safety=True, verbose=False)
        result = await p.run()

        summary = result.summary()
        assert "🛡️" in summary

    @pytest.mark.asyncio
    async def test_summary_with_report(self, tmp_path):
        agent = _make_agent()
        suite = _make_suite()
        rp = str(tmp_path / "r.html")
        p = Pipeline(agent, suite, report="html", report_path=rp, verbose=False)
        result = await p.run()

        summary = result.summary()
        assert "📄" in summary


# ─── Convenience function ────────────────────────────────────────


class TestRunPipeline:
    @pytest.mark.asyncio
    async def test_run_pipeline_function(self):
        agent = _make_agent()
        suite = _make_suite()
        result = await run_pipeline(agent, suite, verbose=False)

        assert isinstance(result, PipelineResult)
        assert result.eval.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_run_pipeline_with_string_suite(self):
        agent = _make_agent()
        result = await run_pipeline(agent, "coding", verbose=False)

        assert isinstance(result, PipelineResult)
        assert len(result.eval.results) > 0


# ─── Import ──────────────────────────────────────────────────────


class TestImports:
    def test_top_level_imports(self):
        import litmusai
        assert hasattr(litmusai, "Pipeline")
        assert hasattr(litmusai, "PipelineResult")
        assert hasattr(litmusai, "run_pipeline")
