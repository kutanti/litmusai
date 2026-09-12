"""Persistence and validation failures found while testing the complete PR stack."""

import asyncio
import json
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from litmusai import Agent, GroundTruth, MetricConfig, evaluate
from litmusai import TestCase as Case
from litmusai import TestSuite as Suite
from litmusai.cli.main import cli
from litmusai.core.agent import AgentResponse
from litmusai.metrics import extraction_observation
from litmusai.results import load_results


@pytest.mark.parametrize("concurrent", [False, True])
async def test_automatic_logs_preserve_runs_started_in_the_same_second(
    tmp_path, monkeypatch, concurrent,
):
    monkeypatch.setattr("litmusai.core.runner.time.strftime", lambda _: "2026-09-12T06:00:00")
    suite = Suite("same", [Case(id="greeting", task="hello")])
    agent = Agent.from_function(lambda _: "hello", name="same")

    async def run(repetition):
        # Explicit shared IDs must not make different repetitions overwrite either.
        return await evaluate(agent, suite, evaluation_id="shared", repetition=repetition,
                              log_dir=tmp_path, verbose=False)

    runs = await asyncio.gather(run(1), run(2)) if concurrent else [await run(1), await run(2)]
    saved = [load_results(path) for path in tmp_path.glob("*.json")]
    assert len(saved) == 2
    assert {row["repetition"] for row in saved} == {1, 2}
    assert sorted(saved, key=lambda row: row["repetition"]) == [run.to_dict() for run in runs]


@pytest.mark.parametrize("mapping", [False, True])
async def test_cyclic_extraction_truth_is_rejected_before_agent_calls(mapping):
    if mapping:
        truth = {}
        truth["self"] = truth
    else:
        truth = []
        truth.append(truth)
    config = MetricConfig(task_type="extraction")
    with pytest.raises(ValueError, match="acyclic JSON data"):
        extraction_observation(truth, "[]", config=config, case_id="cycle", evaluation_id="e")
    fn = Mock(return_value="[]")
    suite = Suite("cyclic", [Case(id="cycle", ground_truth=GroundTruth(
        answer=truth, answer_type="json"))], metrics=config)
    with pytest.raises(ValueError, match="case 'cycle'.*acyclic JSON data"):
        await evaluate(Agent.from_function(fn), suite, verbose=False)
    fn.assert_not_called()


async def test_completed_result_keeps_a_snapshot_of_nested_ground_truth():
    expected = [{"name": "Alice", "tags": ["a"]}]
    output = json.dumps(expected)
    suite = Suite("snapshot", [Case(id="entity", ground_truth=GroundTruth(
        answer=expected, answer_type="json"))], metrics=MetricConfig(task_type="extraction"))
    result = await evaluate(Agent.from_function(lambda _: output), suite, verbose=False)
    saved = json.loads(json.dumps(result.to_dict()))
    expected[0]["tags"].append("changed later")
    expected.append({"name": "Bob"})
    assert result.to_dict() == saved
    assert result.metrics["exact_match_accuracy"]["value"] == 1


def test_offline_observation_keeps_its_expected_values_and_evidence():
    expected = {"names": ["Alice"]}
    observation = extraction_observation(
        expected, json.dumps(expected), config=MetricConfig(task_type="extraction"),
        case_id="field", evaluation_id="e",
    )
    saved = observation.model_dump_json()
    expected["names"].append("Bob")
    assert observation.model_dump_json() == saved


async def test_completed_result_keeps_a_snapshot_of_metric_labels():
    config = MetricConfig(task_type="classification", labels=["a", "b"])
    suite = Suite("snapshot", [Case(id="label", ground_truth=GroundTruth(answer="a"))],
                  metrics=config)
    result = await evaluate(Agent.from_function(lambda _: "a"), suite, verbose=False)
    saved = json.loads(json.dumps(result.to_dict()))
    config.labels.append("later class")
    assert result.to_dict() == saved
    assert result.metric_config.labels == ["a", "b"]


@pytest.mark.parametrize("scenario", [
    "threshold", "budget", "regression", "missing-baseline", "save-baseline",
])
def test_json_stdout_stays_parseable_with_diagnostics(tmp_path, monkeypatch, scenario):
    monkeypatch.chdir(tmp_path)
    output = "wrong" if scenario in ("threshold", "regression") else "expected"
    agent = Agent.from_function(lambda _: AgentResponse(output=output, cost=1.0))
    monkeypatch.setattr("litmusai.ci.load_agent", lambda _: agent)
    suite = tmp_path / "suite.yaml"
    suite.write_text(json.dumps({"name": "json", "cases": [
        {"id": "one", "task": "hello", "expected_contains": ["expected"]},
    ]}), encoding="utf-8")
    args = ["run", "-s", str(suite), "-a", "local:agent", "--format", "json"]
    if scenario == "threshold":
        args += ["--threshold", ".9"]
    elif scenario == "budget":
        args += ["--budget", ".1"]
    elif scenario == "regression":
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({"summary": {"pass_rate": 1}}), encoding="utf-8")
        args += ["--baseline", str(baseline)]
    elif scenario == "missing-baseline":
        args += ["--baseline", str(tmp_path / "absent.json")]
    else:
        args += ["--save-baseline"]
    result = CliRunner().invoke(cli, args)
    payload = json.loads(result.stdout)
    failed = scenario in ("threshold", "budget", "regression")
    assert result.exit_code == int(failed), result.output
    assert payload["success"] is not failed
    assert payload["results"]["summary"]["total"] == 1
    assert "Running" in result.stderr
    if scenario == "save-baseline":
        assert "Baseline saved" in result.stderr


@pytest.mark.parametrize("invalid", ["agent", "suite"])
def test_json_loading_failures_return_json_and_stderr_diagnostics(tmp_path, monkeypatch, invalid):
    def load_agent(_):
        if invalid == "agent":
            raise ValueError("invalid agent configuration")
        return Agent.from_function(lambda _: "hello")

    monkeypatch.setattr("litmusai.ci.load_agent", load_agent)
    result = CliRunner().invoke(cli, [
        "run", "-s", str(tmp_path / "missing.yaml"), "-a", "local:agent", "--format", "json",
    ])
    payload = json.loads(result.stdout)
    assert result.exit_code == 1
    assert payload["success"] is False and payload["error"]
    assert f"Error loading {invalid}" in result.stderr
