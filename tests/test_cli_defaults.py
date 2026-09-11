"""Regression tests for generated suites and CLI configuration precedence."""

from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from litmusai.cli.main import cli
from litmusai.core.agent import AgentResponse
from litmusai.core.scorer import Scorer
from litmusai.core.suite import TestSuite


def test_init_assertions_reject_wrong_answers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["init"])
    assert result.exit_code == 0
    suite = TestSuite.from_yaml(tmp_path / "suites/example.yaml")
    scorer = Scorer()
    for case, answer in zip(suite.cases, ["hello", "4"], strict=True):
        assert scorer.score(case, AgentResponse(output=answer)).passed
        assert not scorer.score(case, AgentResponse(output="wrong answer")).passed
        assert not scorer.score(case, AgentResponse(output="")).passed


@pytest.mark.parametrize("extra, expected", [
    ([], (2, 2)),
    (["--profile", "quick"], (10, 1)),
    (["--profile", "thorough"], (3, 3)),
    (["--profile", "quick", "--concurrency", "4", "--runs", "6"], (4, 6)),
])
def test_cli_overrides_profile_overrides_config(tmp_path, monkeypatch, extra, expected):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".litmus").mkdir()
    (tmp_path / ".litmus/config.yaml").write_text(
        "defaults:\n  concurrency: 2\n  runs: 2\n", encoding="utf-8",
    )
    evaluate = AsyncMock(return_value={"success": True})
    monkeypatch.setattr("litmusai.ci.run_evaluation", evaluate)
    result = CliRunner().invoke(cli, ["run", "-s", "coding", "-a", "agent:fn", *extra])
    assert result.exit_code == 0, result.output
    assert evaluate.call_args.kwargs["concurrency"] == expected[0]
    assert evaluate.call_args.kwargs["runs"] == expected[1]


@pytest.mark.parametrize("option", ["--concurrency", "--runs"])
def test_zero_counts_are_rejected(option):
    result = CliRunner().invoke(cli, ["run", "-s", "coding", "-a", "agent:fn", option, "0"])
    assert result.exit_code == 2
    assert "Invalid value" in result.output
