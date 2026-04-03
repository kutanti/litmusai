"""Tests for CLI commands — history, diff, scan."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from litmusai.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


# ─── litmus init ─────────────────────────────────────────────────


class TestInit:
    def test_init_creates_files(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert Path(".litmus/config.yaml").exists()
            assert Path("suites/example.yaml").exists()
            assert "initialized" in result.output

    def test_init_config_has_log_dir(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])
            config = Path(".litmus/config.yaml").read_text()
            assert "log_dir" in config


# ─── litmus suites ───────────────────────────────────────────────


class TestSuites:
    def test_suites_lists_available(self, runner):
        result = runner.invoke(cli, ["suites"])
        assert result.exit_code == 0


# ─── litmus history ──────────────────────────────────────────────


def _make_result_file(
    directory: Path,
    filename: str,
    agent: str = "test-agent",
    suite: str = "test-suite",
    timestamp: str = "2026-04-03T10:00:00",
    pass_rate: float = 0.8,
    passed: int = 4,
    total: int = 5,
) -> Path:
    data = {
        "agent_name": agent,
        "suite_name": suite,
        "timestamp": timestamp,
        "summary": {
            "pass_rate": pass_rate,
            "passed": passed,
            "total": total,
            "total_cost": 0.01,
        },
        "results": [],
    }
    path = directory / filename
    path.write_text(json.dumps(data))
    return path


class TestHistory:
    def test_history_no_results(self, runner, tmp_path):
        result = runner.invoke(
            cli, ["history", "--log-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "No results found" in result.output

    def test_history_shows_runs(self, runner, tmp_path):
        _make_result_file(tmp_path, "run1.json", agent="agent-1")
        _make_result_file(
            tmp_path, "run2.json",
            agent="agent-2",
            timestamp="2026-04-03T11:00:00",
        )

        result = runner.invoke(
            cli, ["history", "--log-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "agent-1" in result.output
        assert "agent-2" in result.output
        assert "Evaluation History" in result.output

    def test_history_limit(self, runner, tmp_path):
        for i in range(5):
            _make_result_file(
                tmp_path, f"run{i}.json",
                agent=f"agent-{i}",
                timestamp=f"2026-04-0{i+1}T10:00:00",
            )

        result = runner.invoke(
            cli,
            ["history", "--log-dir", str(tmp_path), "--limit", "2"],
        )
        assert result.exit_code == 0
        # Should show 2 runs header
        assert "2 runs" in result.output


# ─── litmus diff ─────────────────────────────────────────────────


def _make_run_file(
    directory: Path,
    filename: str,
    results: list,
    agent: str = "test-agent",
) -> Path:
    passed = sum(1 for r in results if r["passed"])
    data = {
        "agent_name": agent,
        "suite_name": "test",
        "timestamp": "2026-04-03T10:00:00",
        "summary": {
            "total": len(results),
            "passed": passed,
            "pass_rate": (
                passed / len(results) if results else 0
            ),
        },
        "results": results,
    }
    path = directory / filename
    path.write_text(json.dumps(data))
    return path


class TestDiff:
    def test_diff_no_changes(self, runner, tmp_path):
        results = [
            {"case_id": "q1", "case_name": "Math",
             "passed": True, "score": 1.0,
             "latency_ms": 100, "cost": 0.001,
             "input_tokens": 10, "output_tokens": 5},
        ]
        baseline = _make_run_file(
            tmp_path, "baseline.json", results,
        )
        current = _make_run_file(
            tmp_path, "current.json", results,
        )

        result = runner.invoke(
            cli, ["diff", str(baseline), str(current)],
        )
        assert result.exit_code == 0
        assert "No changes detected" in result.output

    def test_diff_regression(self, runner, tmp_path):
        baseline_results = [
            {"case_id": "q1", "case_name": "Math",
             "passed": True, "score": 1.0,
             "latency_ms": 100, "cost": 0.001,
             "input_tokens": 10, "output_tokens": 5},
        ]
        current_results = [
            {"case_id": "q1", "case_name": "Math",
             "passed": False, "score": 0.3,
             "latency_ms": 200, "cost": 0.002,
             "input_tokens": 10, "output_tokens": 5},
        ]
        baseline = _make_run_file(
            tmp_path, "baseline.json", baseline_results,
        )
        current = _make_run_file(
            tmp_path, "current.json", current_results,
        )

        result = runner.invoke(
            cli, ["diff", str(baseline), str(current)],
        )
        assert result.exit_code == 0
        assert "regression" in result.output.lower()

    def test_diff_fail_on_regression(self, runner, tmp_path):
        baseline_results = [
            {"case_id": "q1", "case_name": "Math",
             "passed": True, "score": 1.0,
             "latency_ms": 100, "cost": 0.001,
             "input_tokens": 10, "output_tokens": 5},
        ]
        current_results = [
            {"case_id": "q1", "case_name": "Math",
             "passed": False, "score": 0.0,
             "latency_ms": 100, "cost": 0.001,
             "input_tokens": 10, "output_tokens": 5},
        ]
        baseline = _make_run_file(
            tmp_path, "baseline.json", baseline_results,
        )
        current = _make_run_file(
            tmp_path, "current.json", current_results,
        )

        result = runner.invoke(
            cli,
            ["diff", str(baseline), str(current),
             "--fail-on-regression"],
        )
        assert result.exit_code == 1

    def test_diff_markdown_format(self, runner, tmp_path):
        results = [
            {"case_id": "q1", "case_name": "Math",
             "passed": True, "score": 1.0,
             "latency_ms": 100, "cost": 0.001,
             "input_tokens": 10, "output_tokens": 5},
        ]
        baseline = _make_run_file(
            tmp_path, "baseline.json", results,
        )
        current = _make_run_file(
            tmp_path, "current.json", results,
        )

        result = runner.invoke(
            cli,
            ["diff", str(baseline), str(current),
             "--format", "markdown"],
        )
        assert result.exit_code == 0
        assert "Evaluation Diff" in result.output

    def test_diff_file_not_found(self, runner):
        result = runner.invoke(
            cli, ["diff", "nonexistent.json", "also_missing.json"],
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_diff_improvements(self, runner, tmp_path):
        baseline_results = [
            {"case_id": "q1", "case_name": "Math",
             "passed": False, "score": 0.2,
             "latency_ms": 100, "cost": 0.001,
             "input_tokens": 10, "output_tokens": 5},
        ]
        current_results = [
            {"case_id": "q1", "case_name": "Math",
             "passed": True, "score": 1.0,
             "latency_ms": 100, "cost": 0.001,
             "input_tokens": 10, "output_tokens": 5},
        ]
        baseline = _make_run_file(
            tmp_path, "baseline.json", baseline_results,
        )
        current = _make_run_file(
            tmp_path, "current.json", current_results,
        )

        result = runner.invoke(
            cli, ["diff", str(baseline), str(current)],
        )
        assert result.exit_code == 0
        assert "improvement" in result.output.lower()


# ─── litmus scan ─────────────────────────────────────────────────


class TestScan:
    def test_scan_invalid_agent_path(self, runner):
        result = runner.invoke(
            cli, ["scan", "--agent", "no_colon"],
        )
        assert result.exit_code == 1

    def test_scan_missing_module(self, runner):
        result = runner.invoke(
            cli, ["scan", "--agent", "nonexistent_module:agent"],
        )
        assert result.exit_code == 1
        assert "Cannot import" in result.output

    def test_scan_invalid_category(self, runner, tmp_path):
        # Create a dummy agent module
        agent_file = tmp_path / "dummy_agent.py"
        agent_file.write_text(
            "from litmusai import Agent\n"
            "agent = Agent.from_function("
            "lambda t: 'hi', name='dummy')\n"
        )
        result = runner.invoke(
            cli,
            ["scan", "--agent", f"{agent_file}:agent",
             "--categories", "FAKE_CATEGORY"],
        )
        assert result.exit_code == 1
        assert "Unknown category" in result.output


# ─── litmus run --runs ───────────────────────────────────────────


class TestRunFlags:
    def test_run_help_shows_runs(self, runner):
        result = runner.invoke(cli, ["run", "--help"])
        assert "--runs" in result.output
        assert "--log-dir" in result.output

    def test_run_help_shows_all_options(self, runner):
        result = runner.invoke(cli, ["run", "--help"])
        assert "--suite" in result.output
        assert "--agent" in result.output
        assert "--concurrency" in result.output
        assert "--threshold" in result.output
        assert "--budget" in result.output


# ─── litmus badges ───────────────────────────────────────────────


class TestBadges:
    def test_badges_no_baseline(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["badges"])
            assert result.exit_code == 1
            assert "No baseline found" in result.output

    def test_badges_with_baseline(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            litmus_dir = Path(".litmus")
            litmus_dir.mkdir()
            baseline = {
                "summary": {"pass_rate": 0.95},
            }
            (litmus_dir / "baseline.json").write_text(
                json.dumps(baseline),
            )
            result = runner.invoke(cli, ["badges"])
            assert result.exit_code == 0
            assert "shields.io" in result.output


# ─── Version ─────────────────────────────────────────────────────


class TestVersion:
    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert "0.1.0" in result.output
