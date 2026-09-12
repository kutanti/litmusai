"""Tests for Pipeline class."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from litmusai import GroundTruth, MetricConfig, Pipeline, PipelineResult, run_pipeline
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
        assert len(result.eval.results) == 3 * len(suite)

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
        content = Path(report_path).read_text(encoding="utf-8")
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
        content = Path(report_path).read_text(encoding="utf-8")
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

    @pytest.mark.parametrize("runs", [1, 3])
    @pytest.mark.parametrize(("baseline_runs", "saved_as"), [
        (1, "log"), (1, "legacy"), (2, "log"), (2, "envelope"), (2, "pooled"),
    ])
    async def test_baseline_uses_first_repetition_and_preserves_pooled_outputs(
        self, tmp_path, runs, baseline_runs, saved_as,
    ):
        suite = TestSuite("classification", [TestCase(
            id="answer", task="answer", expected_contains=["42"],
            ground_truth=GroundTruth(answer="42"),
        )], metrics=MetricConfig(task_type="classification", labels=["42", "wrong"]))
        baseline_outputs = iter(["42", "wrong"])
        baseline = await Pipeline(
            Agent.from_function(lambda _: next(baseline_outputs)), suite,
            runs=baseline_runs, log_dir=tmp_path / "baseline", verbose=False,
        ).run()
        baseline_path = next((tmp_path / "baseline").glob("*.json"))
        if saved_as == "pooled":
            baseline.eval.save(baseline_path)
        elif saved_as == "envelope":
            data = json.loads(baseline_path.read_text(encoding="utf-8"))
            # Reordering serialized runs must not change the selected repetition.
            data["run_results"].reverse()
            baseline_path.write_text(json.dumps({"success": True, "results": data}),
                                     encoding="utf-8")
        elif saved_as == "legacy":
            data = json.loads(baseline_path.read_text(encoding="utf-8"))
            data.pop("repetition")
            for row in data["results"]:
                row.pop("repetition")
            baseline_path.write_text(json.dumps(data), encoding="utf-8")
        saved_baseline = baseline_path.read_bytes()

        current_outputs = iter(["wrong", "42", "42"])
        report_path = tmp_path / "results.csv"
        result = await Pipeline(
            Agent.from_function(lambda _: next(current_outputs)), suite, runs=runs,
            baseline=baseline_path, report="csv", report_path=str(report_path),
            log_dir=tmp_path / "current", verbose=False,
        ).run()

        diff = result.baseline_diff
        assert len(diff.cases) == 1
        assert diff.cases[0].case_id == "answer"
        assert diff.cases[0].baseline_passed is True
        assert diff.cases[0].current_passed is False
        assert len(diff.regressions) == 1
        # Later runs improve: the case diff still compares run 1, while metrics,
        # thresholds, logs and reports include every repetition.
        assert len(result.eval.results) == runs
        assert result.eval.pass_rate == pytest.approx((runs - 1) / runs)
        assert result.eval.metrics["accuracy"]["value"] == pytest.approx((runs - 1) / runs)
        assert result.passed is (runs > 1)
        saved = json.loads(next((tmp_path / "current").glob("*.json")).read_text(
            encoding="utf-8"))
        assert len(saved["results"]) == runs
        with report_path.open(encoding="utf-8", newline="") as report:
            assert len(list(csv.DictReader(report))) == runs
        assert baseline_path.read_bytes() == saved_baseline

    @pytest.mark.parametrize("baseline", [
        {"run_results": []},
        {"run_results": [{"repetition": 2, "results": []}]},
        {"run_results": [{"repetition": 1}, {"repetition": 1}]},
        {"repetition": None, "results": [{"case_id": "q1", "repetition": 2}]},
        {"repetition": None, "results": [{"case_id": "q1"}]},
    ])
    async def test_baseline_rejects_missing_or_ambiguous_first_repetition(self, tmp_path, baseline):
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        with pytest.raises(ValueError, match="repetition"):
            await Pipeline(_make_agent(), _make_suite(), baseline=baseline_path,
                           runs=2, verbose=False).run()

    async def test_baseline_still_rejects_duplicate_cases_within_first_repetition(self, tmp_path):
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps({"run_results": [{
            "repetition": 1, "results": [{"case_id": "q1"}, {"case_id": "q1"}],
        }]}), encoding="utf-8")
        with pytest.raises(ValueError, match="case-level diff requires one repetition"):
            await Pipeline(_make_agent(), _make_suite(), baseline=baseline_path,
                           runs=2, verbose=False).run()


# ─── PipelineResult ──────────────────────────────────────────────


class TestPipelineResult:
    @pytest.mark.asyncio
    async def test_summary(self):
        agent = _make_agent()
        suite = _make_suite()
        p = Pipeline(agent, suite, verbose=False)
        result = await p.run()

        summary = result.summary()
        assert "passed" in summary
        assert "ms total" in summary

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
        assert "Safety:" in summary

    @pytest.mark.asyncio
    async def test_summary_with_report(self, tmp_path):
        agent = _make_agent()
        suite = _make_suite()
        rp = str(tmp_path / "r.html")
        p = Pipeline(agent, suite, report="html", report_path=rp, verbose=False)
        result = await p.run()

        summary = result.summary()
        assert "Report:" in summary


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
