"""Tests for result logging, loading, and diffing."""

import json
import tempfile
from pathlib import Path

import pytest

from litmusai.results import (
    CaseDiff,
    diff_results,
    list_results,
    load_results,
)

# ─── Fixtures ────────────────────────────────────────────────────


def _make_result(
    case_id: str = "q1",
    case_name: str = "Math",
    passed: bool = True,
    score: float = 1.0,
    latency_ms: float = 1000,
    cost: float = 0.001,
    input_tokens: int = 20,
    output_tokens: int = 10,
) -> dict:
    return {
        "case_id": case_id,
        "case_name": case_name,
        "task": f"Task for {case_name}",
        "response": "test response",
        "passed": passed,
        "score": score,
        "score_reason": "OK" if passed else "Failed",
        "latency_ms": latency_ms,
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": "gpt-4o",
    }


def _make_run(
    agent: str = "test-agent",
    suite: str = "test-suite",
    timestamp: str = "2026-04-02T10:00:00",
    results: list | None = None,
) -> dict:
    if results is None:
        results = [_make_result()]
    passed = sum(1 for r in results if r["passed"])
    return {
        "agent_name": agent,
        "suite_name": suite,
        "timestamp": timestamp,
        "config": {},
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": passed / len(results) if results else 0,
            "avg_score": (
                sum(r["score"] for r in results) / len(results)
                if results else 0
            ),
            "avg_latency_ms": (
                sum(r["latency_ms"] for r in results) / len(results)
                if results else 0
            ),
            "total_cost": sum(r["cost"] for r in results),
            "total_input_tokens": sum(
                r["input_tokens"] for r in results
            ),
            "total_output_tokens": sum(
                r["output_tokens"] for r in results
            ),
        },
        "results": results,
    }


# ─── CaseDiff ────────────────────────────────────────────────────


class TestCaseDiff:
    def test_regression(self):
        d = CaseDiff(
            case_id="q1", case_name="Math", task="test",
            baseline_passed=True, current_passed=False,
        )
        assert d.is_regression
        assert not d.is_improvement
        assert d.status_icon == "🔴"

    def test_improvement(self):
        d = CaseDiff(
            case_id="q1", case_name="Math", task="test",
            baseline_passed=False, current_passed=True,
        )
        assert d.is_improvement
        assert not d.is_regression
        assert d.status_icon == "🟢"

    def test_new_test(self):
        d = CaseDiff(
            case_id="q1", case_name="Math", task="test",
            baseline_passed=None, current_passed=True,
        )
        assert d.is_new
        assert d.status_icon == "🆕"

    def test_removed_test(self):
        d = CaseDiff(
            case_id="q1", case_name="Math", task="test",
            baseline_passed=True, current_passed=None,
        )
        assert d.is_removed
        assert d.status_icon == "⚪"

    def test_stable_pass(self):
        d = CaseDiff(
            case_id="q1", case_name="Math", task="test",
            baseline_passed=True, current_passed=True,
        )
        assert not d.is_regression
        assert not d.is_improvement
        assert d.status_icon == "✅"

    def test_stable_fail(self):
        d = CaseDiff(
            case_id="q1", case_name="Math", task="test",
            baseline_passed=False, current_passed=False,
        )
        assert d.status_icon == "❌"

    def test_score_change(self):
        d = CaseDiff(
            case_id="q1", case_name="Math", task="test",
            baseline_score=0.8, current_score=0.95,
        )
        assert d.score_change == pytest.approx(0.15)

    def test_score_change_none(self):
        d = CaseDiff(
            case_id="q1", case_name="Math", task="test",
            baseline_score=0.8, current_score=None,
        )
        assert d.score_change is None

    def test_latency_change(self):
        d = CaseDiff(
            case_id="q1", case_name="Math", task="test",
            baseline_latency_ms=1000, current_latency_ms=1500,
        )
        assert d.latency_change_pct == pytest.approx(50.0)

    def test_latency_change_zero_baseline(self):
        d = CaseDiff(
            case_id="q1", case_name="Math", task="test",
            baseline_latency_ms=0, current_latency_ms=1000,
        )
        assert d.latency_change_pct is None

    def test_latency_change_zero_current(self):
        """Zero current latency is valid — should compute change."""
        d = CaseDiff(
            case_id="q1", case_name="Math", task="test",
            baseline_latency_ms=1000, current_latency_ms=0,
        )
        assert d.latency_change_pct == pytest.approx(-100.0)

    def test_cost_change(self):
        d = CaseDiff(
            case_id="q1", case_name="Math", task="test",
            baseline_cost=0.001, current_cost=0.002,
        )
        assert d.cost_change_pct == pytest.approx(100.0)


# ─── Load / Save ─────────────────────────────────────────────────


class TestLoadSave:
    def test_load_results(self):
        data = _make_run()
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False,
        ) as f:
            json.dump(data, f)
            path = f.name
        # File is closed before load — safe on all platforms
        loaded = load_results(path)
        assert loaded["agent_name"] == "test-agent"
        assert len(loaded["results"]) == 1

    def test_list_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 result files
            for i in range(3):
                data = _make_run(
                    timestamp=f"2026-04-0{i+1}T10:00:00",
                    agent=f"agent-{i}",
                )
                path = Path(tmpdir) / f"run-{i}.json"
                with open(path, "w") as f:
                    json.dump(data, f)

            entries = list_results(tmpdir)
            assert len(entries) == 3
            assert entries[0]["agent_name"] == "agent-2"  # newest first

    def test_list_results_sorts_by_timestamp(self):
        """Verify sort is by timestamp, not filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # File "aaa.json" has newest timestamp
            data_new = _make_run(
                timestamp="2026-04-03T10:00:00", agent="newest",
            )
            with open(Path(tmpdir) / "aaa.json", "w") as f:
                json.dump(data_new, f)
            # File "zzz.json" has oldest timestamp
            data_old = _make_run(
                timestamp="2026-04-01T10:00:00", agent="oldest",
            )
            with open(Path(tmpdir) / "zzz.json", "w") as f:
                json.dump(data_old, f)

            entries = list_results(tmpdir)
            assert entries[0]["agent_name"] == "newest"
            assert entries[1]["agent_name"] == "oldest"

    def test_list_results_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = list_results(tmpdir)
            assert entries == []

    def test_list_results_nonexistent_dir(self):
        entries = list_results("/nonexistent/dir")
        assert entries == []

    def test_list_results_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                data = _make_run(agent=f"agent-{i}")
                path = Path(tmpdir) / f"run-{i:02d}.json"
                with open(path, "w") as f:
                    json.dump(data, f)

            entries = list_results(tmpdir, limit=3)
            assert len(entries) == 3

    def test_list_results_skips_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid
            data = _make_run()
            with open(Path(tmpdir) / "good.json", "w") as f:
                json.dump(data, f)
            # Invalid
            with open(Path(tmpdir) / "bad.json", "w") as f:
                f.write("not json{{{")

            entries = list_results(tmpdir)
            assert len(entries) == 1


# ─── Diff ────────────────────────────────────────────────────────


class TestDiffResults:
    def test_no_changes(self):
        baseline = _make_run(results=[
            _make_result("q1", "Math", True, 1.0),
        ])
        current = _make_run(results=[
            _make_result("q1", "Math", True, 1.0),
        ])
        diff = diff_results(baseline, current)
        assert len(diff.regressions) == 0
        assert len(diff.improvements) == 0
        assert diff.pass_rate_change == 0.0
        assert not diff.has_regressions

    def test_regression_detected(self):
        baseline = _make_run(results=[
            _make_result("q1", "Math", True, 1.0),
            _make_result("q2", "Code", True, 0.9),
        ])
        current = _make_run(results=[
            _make_result("q1", "Math", True, 1.0),
            _make_result("q2", "Code", False, 0.3),
        ])
        diff = diff_results(baseline, current)
        assert len(diff.regressions) == 1
        assert diff.regressions[0].case_id == "q2"
        assert diff.has_regressions

    def test_improvement_detected(self):
        baseline = _make_run(results=[
            _make_result("q1", "Math", False, 0.3),
        ])
        current = _make_run(results=[
            _make_result("q1", "Math", True, 1.0),
        ])
        diff = diff_results(baseline, current)
        assert len(diff.improvements) == 1
        assert diff.pass_rate_change > 0

    def test_new_test(self):
        baseline = _make_run(results=[
            _make_result("q1", "Math", True),
        ])
        current = _make_run(results=[
            _make_result("q1", "Math", True),
            _make_result("q2", "Code", True),
        ])
        diff = diff_results(baseline, current)
        assert len(diff.new_tests) == 1
        assert diff.new_tests[0].case_id == "q2"

    def test_removed_test(self):
        baseline = _make_run(results=[
            _make_result("q1", "Math", True),
            _make_result("q2", "Code", True),
        ])
        current = _make_run(results=[
            _make_result("q1", "Math", True),
        ])
        diff = diff_results(baseline, current)
        assert len(diff.removed_tests) == 1

    def test_pass_rate_change(self):
        baseline = _make_run(results=[
            _make_result("q1", passed=True),
            _make_result("q2", passed=True),
        ])
        current = _make_run(results=[
            _make_result("q1", passed=True),
            _make_result("q2", passed=False),
        ])
        diff = diff_results(baseline, current)
        assert diff.baseline_pass_rate == 1.0
        assert diff.current_pass_rate == 0.5
        assert diff.pass_rate_change == pytest.approx(-0.5)

    def test_to_markdown(self):
        baseline = _make_run(results=[
            _make_result("q1", "Math", True, 1.0, 1000, 0.001),
            _make_result("q2", "Code", True, 0.9, 2000, 0.002),
        ])
        current = _make_run(results=[
            _make_result("q1", "Math", True, 1.0, 1200, 0.001),
            _make_result("q2", "Code", False, 0.3, 3000, 0.003),
        ])
        diff = diff_results(baseline, current)
        md = diff.to_markdown()
        assert "## 📊 Evaluation Diff" in md
        assert "Regressions" in md
        assert "Code" in md
        assert "📉" in md  # pass rate dropped

    def test_to_markdown_no_regressions(self):
        baseline = _make_run(results=[
            _make_result("q1", "Math", True),
        ])
        current = _make_run(results=[
            _make_result("q1", "Math", True),
        ])
        diff = diff_results(baseline, current)
        md = diff.to_markdown()
        assert "Regressions" not in md

    def test_to_table(self):
        baseline = _make_run(results=[
            _make_result("q1", "Math", True, 1.0, 1000),
        ])
        current = _make_run(results=[
            _make_result("q1", "Math", False, 0.3, 2000),
        ])
        diff = diff_results(baseline, current)
        table = diff.to_table()
        assert "Math" in table
        assert "PASS" in table
        assert "FAIL" in table

    def test_complex_diff(self):
        """Mix of regressions, improvements, new, removed."""
        baseline = _make_run(results=[
            _make_result("q1", "Math", True),
            _make_result("q2", "Code", False),
            _make_result("q3", "Removed", True),
        ])
        current = _make_run(results=[
            _make_result("q1", "Math", False),  # regression
            _make_result("q2", "Code", True),   # improvement
            _make_result("q4", "New", True),     # new
            # q3 removed
        ])
        diff = diff_results(baseline, current)
        assert len(diff.regressions) == 1
        assert len(diff.improvements) == 1
        assert len(diff.new_tests) == 1
        assert len(diff.removed_tests) == 1

    def test_empty_baseline(self):
        baseline = _make_run(results=[])
        current = _make_run(
            timestamp="2026-04-02T11:00:00",
            results=[_make_result("q1", "Math", True)],
        )
        diff = diff_results(baseline, current)
        assert len(diff.new_tests) == 1
        assert diff.baseline_pass_rate == 0.0

    def test_empty_current(self):
        baseline = _make_run(results=[
            _make_result("q1", "Math", True),
        ])
        current = _make_run(
            timestamp="2026-04-02T11:00:00",
            results=[],
        )
        diff = diff_results(baseline, current)
        assert len(diff.removed_tests) == 1
