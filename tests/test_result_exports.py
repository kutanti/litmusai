"""Keep evaluation identities and every repetition across CLI exports."""

import json
import re
import xml.etree.ElementTree as ET

import pytest
from click.testing import CliRunner

from litmusai import Agent
from litmusai.cli.main import cli
from litmusai.core.agent import AgentResponse


@pytest.mark.parametrize("runs", [1, 2])
def test_cli_json_and_logs_preserve_payload(tmp_path, monkeypatch, runs):
    response = "[bold]literal[/bold] " + "café " * 160
    agent = Agent.from_function(lambda _: response, name="export-agent")
    monkeypatch.setattr("litmusai.ci.load_agent", lambda _: agent)
    suite = tmp_path / "suite.yaml"
    suite.write_text(json.dumps({"name": "exports", "cases": [
        {"id": "invoice", "name": "Invoice", "task": "hello"},
    ]}), encoding="utf-8")
    output = tmp_path / "result.json"
    logs = tmp_path / "logs"

    result = CliRunner().invoke(cli, [
        "run", "--suite", str(suite), "--agent", "test:agent",
        "--runs", str(runs), "--format", "json", "--output", str(output),
        "--log-dir", str(logs),
    ])

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    stdout_payload, _ = json.JSONDecoder().raw_decode(result.output[result.output.index("{"):])
    assert stdout_payload == payload
    data = payload["results"]
    assert data["schema_version"] == "1.0"
    assert data["evaluation_id"]
    assert data["summary"]["total"] == runs
    assert data["agent_name"] == "export-agent"
    assert [row["repetition"] for row in data["results"]] == list(range(1, runs + 1))
    for row in data["results"]:
        assert row["evaluation_id"] == data["evaluation_id"]
        assert row["case_id"] == "invoice"
        assert row["response"] == response
    log_files = list(logs.glob("*.json"))
    assert len(log_files) == 1
    assert json.loads(log_files[0].read_text(encoding="utf-8")) == data
    if runs > 1:
        assert data["repetition"] is None
        assert len(data["run_results"]) == runs
        for repetition, run in enumerate(data["run_results"], 1):
            assert run["schema_version"] == "1.0"
            assert run["evaluation_id"] == data["evaluation_id"]
            assert run["repetition"] == repetition
            assert run["results"] == [data["results"][repetition - 1]]

    report = CliRunner().invoke(cli, [
        "report", "-r", str(output), "--html", str(tmp_path / "report.html"),
        "--junit", str(tmp_path / "report.xml"), "--csv", str(tmp_path / "report.csv"),
    ])
    assert report.exit_code == 0, report.output
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    detail_ids = re.findall(r'id="(detail-[^"]+)"', html)
    targets = re.findall(r"toggleDetail\('([^']+)'\)", html)
    assert len(set(detail_ids)) == runs
    assert [f"detail-{target}" for target in targets] == detail_ids
    assert len(ET.parse(tmp_path / "report.xml").findall(".//testcase")) == runs
    assert (tmp_path / "report.csv").read_text(encoding="utf-8").count("invoice") == runs


@pytest.mark.parametrize("gate,value", [("--threshold", "0.75"), ("--budget", "0.025")])
def test_cli_checks_all_repetitions(tmp_path, monkeypatch, gate, value):
    responses = iter(["wrong", "right"])
    agent = Agent.from_function(
        lambda _: AgentResponse(output=next(responses), cost=0.02), name="flaky",
    )
    monkeypatch.setattr("litmusai.ci.load_agent", lambda _: agent)
    suite = tmp_path / "suite.yaml"
    suite.write_text(json.dumps({"name": "exports", "cases": [
        {"id": "case", "task": "answer", "ground_truth": {"answer": "right"}},
    ]}), encoding="utf-8")
    output = tmp_path / "result.json"

    result = CliRunner().invoke(cli, [
        "run", "--suite", str(suite), "--agent", "test:agent", "--runs", "2",
        "--format", "json", "--output", str(output), gate, value,
    ])

    assert result.exit_code == 1, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["success"] is False
    data = payload["results"]
    assert data["summary"]["pass_rate"] == 0.5
    assert data["summary"]["total_cost"] == 0.04
    assert [row["passed"] for row in data["results"]] == [False, True]
