"""Tests for the CI/CD integration module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from litmusai.ci import (
    compare_with_baseline,
    format_report,
    format_table,
    load_agent,
    load_baseline,
    results_to_dict,
    save_baseline,
)

# ─── Test Agent Loading ───────────────────────────────────────────


class TestLoadAgent:
    def test_no_colon_raises(self):
        with pytest.raises(ValueError, match="module:attribute"):
            load_agent("just_a_module")

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_agent("/tmp/nonexistent_agent.py:agent")

    def test_missing_module_raises(self):
        with pytest.raises(ImportError):
            load_agent("totally_fake_module_xyz:agent")

    def test_missing_attribute_raises(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            load_agent("json:nonexistent_thing_xyz")

    def test_non_callable_raises(self):
        # json.JSONDecodeError is a class but not an Agent or simple callable
        # Actually let's use a known non-callable attribute
        with pytest.raises((TypeError, AttributeError)):
            load_agent("json:__version__")

    def test_load_callable(self):
        # json.dumps is a callable — should wrap as Agent
        agent = load_agent("json:dumps")
        assert agent.name == "dumps"

    def test_load_from_file(self, tmp_path: Path):
        # Create a temp agent file
        agent_file = tmp_path / "my_agent.py"
        agent_file.write_text(
            "def my_func(task):\n"
            "    return f'answer: {task}'\n"
        )
        agent = load_agent(f"{agent_file}:my_func")
        assert agent.name == "my_func"

    def test_load_from_file_no_sys_path_mutation(self, tmp_path: Path):
        """Loading from file should not pollute sys.path."""
        import sys

        agent_file = tmp_path / "isolated_agent.py"
        agent_file.write_text("def run(t): return t\n")
        original_path = sys.path.copy()
        load_agent(f"{agent_file}:run")
        assert sys.path == original_path


# ─── Test Baseline ────────────────────────────────────────────────


class TestBaseline:
    def test_load_missing_file(self):
        assert load_baseline("/tmp/nonexistent_baseline.json") is None

    def test_load_invalid_json(self, tmp_path: Path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")
        assert load_baseline(bad_file) is None

    def test_save_and_load(self, tmp_path: Path):
        data = {"summary": {"pass_rate": 0.9, "total_cost": 0.05}}
        path = tmp_path / "baseline.json"
        save_baseline(data, path)
        loaded = load_baseline(path)
        assert loaded is not None
        assert loaded["summary"]["pass_rate"] == 0.9

    def test_save_creates_directories(self, tmp_path: Path):
        path = tmp_path / "deep" / "nested" / "baseline.json"
        save_baseline({"test": True}, path)
        assert path.exists()


# ─── Test Baseline Comparison ─────────────────────────────────────


class TestCompareWithBaseline:
    def _make_data(
        self, pass_rate: float, cost: float, latency: float
    ) -> dict:
        return {
            "summary": {
                "pass_rate": pass_rate,
                "total_cost": cost,
                "avg_latency_ms": latency,
            }
        }

    def test_no_regression(self):
        current = self._make_data(0.9, 0.05, 300)
        baseline = self._make_data(0.85, 0.05, 300)
        result = compare_with_baseline(current, baseline)
        assert not result["has_regression"]
        assert result["pass_rate"]["delta"] == pytest.approx(0.05)

    def test_pass_rate_regression(self):
        current = self._make_data(0.5, 0.05, 300)
        baseline = self._make_data(0.9, 0.05, 300)
        result = compare_with_baseline(current, baseline)
        assert result["has_regression"]
        assert any("Pass rate" in r for r in result["regressions"])

    def test_cost_regression(self):
        current = self._make_data(0.9, 1.0, 300)
        baseline = self._make_data(0.9, 0.05, 300)
        result = compare_with_baseline(current, baseline)
        assert result["has_regression"]
        assert any("Cost" in r for r in result["regressions"])

    def test_latency_regression(self):
        current = self._make_data(0.9, 0.05, 1000)
        baseline = self._make_data(0.9, 0.05, 300)
        result = compare_with_baseline(current, baseline)
        assert result["has_regression"]
        assert any("Latency" in r for r in result["regressions"])

    def test_improvement_no_regression(self):
        current = self._make_data(0.95, 0.03, 200)
        baseline = self._make_data(0.80, 0.10, 500)
        result = compare_with_baseline(current, baseline)
        assert not result["has_regression"]

    def test_small_drop_no_regression(self):
        # Less than 5% drop shouldn't trigger
        current = self._make_data(0.87, 0.05, 300)
        baseline = self._make_data(0.90, 0.05, 300)
        result = compare_with_baseline(current, baseline)
        assert not result["has_regression"]

    def test_zero_baseline_no_crash(self):
        current = self._make_data(0.9, 0.05, 300)
        baseline = self._make_data(0.0, 0.0, 0)
        result = compare_with_baseline(current, baseline)
        assert not result["has_regression"]


# ─── Test Report Formatting ───────────────────────────────────────


class TestFormatReport:
    def _make_data(self) -> dict:
        return {
            "agent": "test-agent",
            "suite": "test-suite",
            "timestamp": "2026-04-01T00:00:00",
            "summary": {
                "total": 5,
                "passed": 4,
                "failed": 1,
                "pass_rate": 0.8,
                "total_cost": 0.0512,
                "avg_latency_ms": 350,
            },
            "results": [
                {
                    "test": f"test_{i}",
                    "task": f"task {i}",
                    "passed": i < 4,
                    "score": 0.9 if i < 4 else 0.2,
                    "reason": "ok" if i < 4 else "failed",
                    "latency_ms": 300 + i * 25,
                    "cost": 0.01,
                    "output": f"output {i}",
                }
                for i in range(5)
            ],
        }

    def test_markdown_format(self):
        md = format_report(self._make_data(), fmt="markdown")
        assert "## 🧪 LitmusAI Report" in md
        assert "test-agent" in md
        assert "PASSED" in md
        assert "80%" in md

    def test_json_format(self):
        output = format_report(self._make_data(), fmt="json")
        data = json.loads(output)
        assert "results" in data

    def test_with_baseline(self):
        current = self._make_data()
        baseline = self._make_data()
        baseline["summary"]["pass_rate"] = 0.6
        md = format_report(current, baseline, fmt="markdown")
        assert "vs Baseline" in md
        assert "Current" in md

    def test_with_regression(self):
        current = self._make_data()
        current["summary"]["pass_rate"] = 0.3
        baseline = self._make_data()
        md = format_report(current, baseline, fmt="markdown")
        assert "Regressions" in md

    def test_json_with_baseline(self):
        current = self._make_data()
        baseline = self._make_data()
        output = format_report(current, baseline, fmt="json")
        data = json.loads(output)
        assert "comparison" in data

    def test_detailed_results_section(self):
        md = format_report(self._make_data(), fmt="markdown")
        assert "Detailed Results" in md
        assert "test_0" in md

    def test_custom_threshold_pass(self):
        data = self._make_data()
        data["summary"]["pass_rate"] = 0.6
        md = format_report(data, fmt="markdown", threshold=0.5)
        assert "PASSED" in md

    def test_custom_threshold_fail(self):
        data = self._make_data()
        data["summary"]["pass_rate"] = 0.6
        md = format_report(data, fmt="markdown", threshold=0.9)
        assert "FAILED" in md


class TestFormatTable:
    def test_format_table_no_crash(self, capsys: pytest.CaptureFixture[str]):
        data = {
            "suite": "test",
            "summary": {
                "total": 1, "passed": 1, "failed": 0,
                "pass_rate": 1.0, "total_cost": 0.01,
                "avg_latency_ms": 100,
            },
            "results": [
                {
                    "test": "t1", "passed": True, "score": 1.0,
                    "latency_ms": 100, "cost": 0.01,
                }
            ],
        }
        # Should not raise
        format_table(data)


# ─── Test Results Serialization ────────────────────────────────────


class TestResultsToDict:
    def test_converts_eval_results(self):
        from litmusai.core.agent import AgentResponse
        from litmusai.core.runner import EvalResults, TestResult
        from litmusai.core.scorer import ScoreResult
        from litmusai.core.suite import TestCase

        results = EvalResults(
            agent_name="test",
            suite_name="suite",
            timestamp="2026-01-01",
            results=[
                TestResult(
                    case=TestCase(id="t1", name="test1", task="do something"),
                    response=AgentResponse(output="done", latency_ms=100),
                    score=ScoreResult(passed=True, score=1.0, reason="ok"),
                    passed=True,
                    latency_ms=100,
                    cost=0.01,
                ),
            ],
            total_cost=0.01,
            total_time_ms=100,
        )

        data = results_to_dict(results)
        assert data["agent"] == "test"
        assert data["summary"]["total"] == 1
        assert data["summary"]["passed"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["passed"] is True
