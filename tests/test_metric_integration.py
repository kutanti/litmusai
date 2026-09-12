"""Labeled metrics through the runner, CLI, pipeline, saved JSON and HTML reports."""

import io
import json
import re
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from rich.console import Console

from litmusai import (
    Agent,
    Contains,
    GroundTruth,
    MetricConfig,
    Observation,
    Pipeline,
    Reporter,
    aggregate_metrics,
    apply_ground_truth,
    evaluate,
    multi_evaluate,
)
from litmusai import TestCase as Case
from litmusai import TestSuite as Suite
from litmusai.ci import format_report, format_table, results_to_dict, run_evaluation
from litmusai.cli.main import cli
from litmusai.reports import render_html
from litmusai.results import diff_results, load_results


def routing_suite(*expected):
    return Suite("routes", [Case(id=str(i), task=str(i), ground_truth=GroundTruth(answer=label))
                            for i, label in enumerate(expected)], metrics=MetricConfig(
                                task_type="classification", labels=["a", "b"]))


@pytest.mark.parametrize("bad", [None, "unknown", ["a"]])
async def test_all_ground_truth_is_validated_before_any_agent_call(bad):
    suite = routing_suite("a", bad)
    fn = Mock(return_value="a")
    with pytest.raises(ValueError, match="case '1'.*expected label"):
        await evaluate(Agent.from_function(fn), suite, verbose=False)
    fn.assert_not_called()


@pytest.mark.parametrize("case_id", ["0", "", "  "])
async def test_invalid_case_ids_fail_before_agent_call(case_id):
    suite = routing_suite("a", "a")
    suite.cases[1].id = case_id
    fn = Mock(return_value="a")
    with pytest.raises(ValueError, match="nonempty and unique"):
        await evaluate(Agent.from_function(fn), suite, verbose=False)
    fn.assert_not_called()


async def test_malformed_extraction_truth_is_rejected_before_calls():
    suite = Suite("fields", [Case(id="c", task="q")], metrics=MetricConfig(task_type="extraction"))
    fn = Mock(return_value="{}")
    with pytest.raises(ValueError, match="ground truth must be"):
        await evaluate(Agent.from_function(fn), suite, verbose=False)
    fn.assert_not_called()


async def test_full_prediction_metrics_are_independent_of_assertions(tmp_path):
    suite = routing_suite("a")
    suite.metrics = MetricConfig(task_type="classification", labels=["a", "b"],
                                 prediction_field="/label")
    suite.cases[0].assertions = [Contains(["missing text"])]
    output = json.dumps({"padding": "x" * 3000, "label": "a"})
    results = await evaluate(Agent.from_function(lambda _: output), suite, verbose=False)
    assert results.pass_rate == 0
    assert results.metrics["accuracy"]["value"] == 1
    data = load_results(results.save(tmp_path / "results.json"))
    assert data["results"][0]["response"] == output
    record = Observation.model_validate(data["results"][0]["observation"])
    assert record.predicted == "a"
    assert data["metrics"] == aggregate_metrics([record], MetricConfig.model_validate(
        data["metric_config"]))
    assert json.loads(Reporter.to_json(results)) == data
    assert "Classification metrics" in Reporter.to_markdown(results)


@pytest.mark.parametrize("verbose", [False, True])
async def test_runner_keeps_errors_and_invalid_predictions(verbose):
    def agent(task):
        if task == "1":
            raise RuntimeError("provider failed")
        return "unknown"

    results = await evaluate(Agent.from_function(agent), routing_suite("a", "b"), verbose=verbose)
    assert len(results.results) == 2
    assert results.metrics["execution_errors"] == 1
    assert results.metrics["invalid_predictions"] == 1
    assert results.metrics["micro"]["fn"] == 2
    assert results.metrics["prediction_coverage"]["value"] == 0


async def test_multi_run_persistence_retains_identities_and_recomputable_counts(tmp_path):
    outputs = iter(["a", "b", "b", ""])
    multi = await multi_evaluate(
        Agent.from_function(lambda _: next(outputs)), routing_suite("a", "b"),
        runs=2, concurrency=1, verbose=False,
    )
    data = load_results(multi.save(tmp_path / "multi.json"))
    assert data["summary"]["total"] == 4
    assert len(data["results"]) == 4
    assert len(data["run_results"]) == 2
    records = [Observation.model_validate(r["observation"]) for r in data["results"]]
    assert len({(r.evaluation_id, r.repetition, r.case_id) for r in records}) == 4
    assert data["metrics"] == aggregate_metrics(records[::-1], multi.combined.metric_config)
    assert multi.metrics["micro"]["f1"]["value"] == pytest.approx(4 / 7)
    assert results_to_dict(multi) == data
    with pytest.raises(ValueError, match="one repetition"):
        diff_results(data, data)
    assert not diff_results(data["run_results"][0], data["run_results"][0]).regressions


async def test_pipeline_pools_runs_and_saves_metrics(tmp_path):
    outputs = iter(["a", ""])
    result = await Pipeline(
        Agent.from_function(lambda _: next(outputs)), routing_suite("a"), runs=2,
        concurrency=1, verbose=False, report="html", report_path=str(tmp_path / "report.html"),
        log_dir=tmp_path / "logs",
    ).run()
    assert result.eval.metrics["accuracy"]["value"] == .5
    assert result.eval.pass_rate == .5
    assert len(result.eval.results) == 2
    data = load_results(next((tmp_path / "logs").glob("*.json")))
    assert data["metrics"] == result.eval.metrics
    assert "Prediction coverage: 1/2" in (tmp_path / "report.html").read_text(encoding="utf-8")


async def test_cli_examples_export_pooled_metrics_and_render_reports(tmp_path, monkeypatch):
    import litmusai.ci

    output = io.StringIO()
    monkeypatch.setattr(litmusai.ci, "console", Console(file=output, width=180))
    for name, fn in [("routing", "route"), ("extraction", "extract")]:
        path = tmp_path / f"{name}.json"
        result = await run_evaluation(
            f"examples/{name}.yaml", f"examples/labeled_agents.py:{fn}", runs=2,
            output_path=str(path), fmt="table", log_dir=str(tmp_path / "logs"),
        )
        assert result["success"] is True
        data = load_results(path)
        assert data == result["data"]
        assert len(data["run_results"]) == 2
        assert data["metrics"]["total"] == (8 if name == "routing" else 6)
        assert "Precision" in output.getvalue()
        assert "Prediction coverage" in output.getvalue()
        rendered = CliRunner().invoke(cli, ["report", "-r", str(path),
                                            "--html", str(tmp_path / f"{name}.html"),
                                            "--junit", str(tmp_path / f"{name}.xml"),
                                            "--csv", str(tmp_path / f"{name}.csv")])
        assert rendered.exit_code == 0, rendered.output
        report = (tmp_path / f"{name}.html").read_text(encoding="utf-8")
        assert "Metric evidence" in report
        assert "Precision" in report
        assert "Check Pass Rate" in report
        assert "nonempty output" in report
        ids = re.findall(r'id="(detail-[^"]+)"', report)
        assert len(ids) == len(set(ids)) == data["summary"]["total"]


def test_cli_run_command_uses_labeled_metrics(tmp_path):
    path = tmp_path / "routing.json"
    result = CliRunner().invoke(cli, [
        "run", "-s", "examples/routing.yaml", "-a", "examples/labeled_agents.py:route",
        "--runs", "2", "--output", str(path),
    ])
    assert result.exit_code == 0, result.output
    data = load_results(path)
    assert data["metrics"]["accuracy"]["value"] == .75
    assert data["metrics"]["total"] == 8


async def test_html_and_markdown_escape_labels_and_metric_evidence(tmp_path):
    label = '<script>alert("x")</script>'
    suite = Suite("escaping", [Case(id="c", task="t", ground_truth=GroundTruth(answer=label))],
                  metrics=MetricConfig(task_type="classification", labels=[label]))
    data = (await evaluate(Agent.from_function(lambda _: label), suite, verbose=False)).to_dict()
    report = render_html(data, tmp_path / "report.html").read_text(encoding="utf-8")
    assert label not in report
    assert "&lt;script&gt;" in report
    markdown = format_report(data)
    assert label not in markdown
    assert "Confusion matrix" in markdown


async def test_empty_and_legacy_suites_have_no_task_aggregate(tmp_path):
    agent = Agent.from_function(lambda _: "a")
    empty = await evaluate(agent, routing_suite(), verbose=False)
    legacy = await evaluate(agent, Suite("legacy", [Case(id="c", task="t")]), verbose=False)
    assert empty.metrics is None
    assert legacy.metrics is None
    format_table(empty.to_dict())
    assert "Classification metrics" not in render_html(
        empty.to_dict(), tmp_path / "empty.html").read_text(encoding="utf-8")


def test_future_result_version_is_rejected(tmp_path):
    path = tmp_path / "future.json"
    path.write_text('{"schema_version":"9.0"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported result schema_version"):
        load_results(path)


async def test_apply_ground_truth_updates_loaded_truth_and_preserves_assertions(tmp_path):
    suite = routing_suite("a")
    path = tmp_path / "suite.yaml"
    suite.to_yaml(path)
    suite = Suite.from_yaml(path)
    suite.cases[0].assertions = [Contains(["b"])]
    assert apply_ground_truth(suite, {"0": GroundTruth(answer="b")}) == 1
    results = await evaluate(Agent.from_function(lambda _: "b"), suite, verbose=False)
    assert results.metrics["accuracy"]["value"] == 1
    assert results.results[0].observation.expected == "b"
    assert results.pass_rate == 1
