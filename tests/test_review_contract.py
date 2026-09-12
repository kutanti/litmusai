"""Regressions for result readers, identities, and ground-truth entry points."""

import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import replace
from unittest.mock import Mock

import pytest
import yaml
from click.testing import CliRunner

from litmusai import Agent, GroundTruth, Reporter, apply_ground_truth, evaluate, load_ground_truth
from litmusai import TestCase as Case
from litmusai import TestSuite as Suite
from litmusai.assertions import Contains
from litmusai.ci import compare_with_baseline, format_report, load_baseline
from litmusai.cli.main import cli
from litmusai.core.agent import AgentResponse
from litmusai.core.runner import EvalResults, MultiRunResults
from litmusai.core.runner import TestResult as Result
from litmusai.core.scorer import ScoreResult
from litmusai.exports import to_csv, to_junit_xml
from litmusai.metrics import Observation
from litmusai.reports import render_html
from litmusai.results import diff_results, load_results, normalize_results


def make_row(**kwargs):
    return Result(
        case=Case(id="café", task="hello"), response=AgentResponse(output="東京"),
        score=ScoreResult(passed=True, score=1), passed=True, latency_ms=1, **kwargs,
    )


def make_run(**kwargs):
    return EvalResults(agent_name="agent", suite_name="suite", **kwargs)


def test_manual_result_inherits_parent_identity_without_mutation(tmp_path):
    row = make_row()
    result = make_run(results=[row], repetition=2)
    data = load_results(result.save(tmp_path / "run.json"))
    assert data["evaluation_id"]
    assert data["results"][0]["evaluation_id"] == data["evaluation_id"]
    assert data["results"][0]["repetition"] == 2
    assert row.evaluation_id == ""
    assert row.repetition is None
    assert result.to_dict() == data
    assert json.loads(Reporter.to_json(result, tmp_path / "reporter.json")) == data
    assert load_results(tmp_path / "reporter.json") == data


@pytest.mark.parametrize("field,value", [
    ("evaluation_id", "other"), ("repetition", 2), ("repetition", 0),
    ("repetition", True),
])
def test_conflicting_row_identity_is_rejected(field, value):
    with pytest.raises(ValueError, match="evaluation_id|repetition"):
        make_run(evaluation_id="eval", results=[make_row(**{field: value})]).to_dict()


@pytest.mark.parametrize("field,value", [
    ("evaluation_id", ""), ("evaluation_id", "  "), ("repetition", 0),
    ("repetition", True),
])
def test_invalid_parent_identity_is_rejected(field, value):
    with pytest.raises(ValueError, match="evaluation_id|repetition"):
        make_run(results=[make_row()], **{field: value}).to_dict()


@pytest.mark.parametrize("changes", [
    {"evaluation_id": "other"}, {"case_id": "other"}, {"repetition": 2},
])
def test_conflicting_observation_identity_is_rejected(changes):
    values = {"evaluation_id": "eval", "case_id": "café", "repetition": 1,
              "task_type": "classification", "expected": "a", **changes}
    observation = Observation.model_validate(values)
    with pytest.raises(ValueError, match="observation identity"):
        make_run(evaluation_id="eval", results=[make_row(observation=observation)]).to_dict()


def test_duplicate_result_identity_is_rejected_without_overwriting_saved_file(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate result identity"):
        make_run(results=[make_row(), make_row()]).save(path)
    assert path.read_text(encoding="utf-8") == "original"


def test_pooled_manual_results_retain_each_parent_repetition():
    runs = [make_run(evaluation_id="eval", repetition=i, results=[make_row()]) for i in (1, 2)]
    multi = MultiRunResults("agent", "suite", n_runs=2, evaluation_id="eval", run_results=runs)
    data = multi.to_dict()
    assert json.loads(Reporter.to_json(multi)) == data
    assert [row["repetition"] for row in data["results"]] == [1, 2]
    assert all(row["evaluation_id"] == "eval" for row in data["results"])
    assert data["results"] == [run["results"][0] for run in data["run_results"]]
    assert runs[0].results[0].repetition is None
    with pytest.raises(ValueError, match="parent evaluation_id"):
        replace(multi, evaluation_id="other").to_dict()
    with pytest.raises(ValueError, match="unique.*repetition"):
        replace(multi, run_results=[runs[0], runs[0]]).to_dict()


@pytest.mark.parametrize("case_id", ["duplicate", "", "  ", 1, None])
def test_yaml_rejects_invalid_or_duplicate_case_ids(tmp_path, case_id):
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({"cases": [{"id": "duplicate"}, {"id": case_id}]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="case IDs must be nonempty and unique"):
        Suite.from_yaml(path)


@pytest.mark.parametrize("as_list", [False, True])
async def test_duplicate_python_cases_fail_before_agent_calls(as_list):
    cases = [Case(id="same"), Case(id="same")]
    fn = Mock(return_value="answer")
    with pytest.raises(ValueError, match="nonempty and unique"):
        await evaluate(Agent.from_function(fn), cases if as_list else Suite("suite", cases),
                       verbose=False)
    fn.assert_not_called()


@pytest.mark.parametrize("truth", [
    {}, None, [], False, {"answer_type": "numeric"}, {"answer_type": "numeric", "answer": None},
])
def test_standalone_loader_rejects_malformed_truth(tmp_path, truth):
    path = tmp_path / "truth.yaml"
    path.write_text(yaml.safe_dump({"cases": [{"id": "missing", "ground_truth": truth}]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="Case 'missing'.*(requires an answer|must be a mapping)"):
        load_ground_truth(path)


@pytest.mark.parametrize("answer_type", ["text", "numeric", "json", "boolean", "list"])
def test_all_ground_truth_entry_points_reject_missing_answers(answer_type):
    truth = GroundTruth(answer_type=answer_type)
    with pytest.raises(ValueError, match="requires an answer"):
        GroundTruth.from_dict({"answer_type": answer_type})
    with pytest.raises(ValueError, match="requires an answer"):
        truth.to_assertions()
    suite = Suite("suite", [Case(id="first"), Case(id="bad", assertions=[Contains(["x"])])])
    with pytest.raises(ValueError, match="Case 'bad'.*requires an answer"):
        apply_ground_truth(suite, {"first": GroundTruth(answer="x"), "bad": truth})
    assert all(case.ground_truth is None and not case.metadata for case in suite.cases)
    assert not suite.cases[0].assertions
    assert len(suite.cases[1].assertions) == 1


@pytest.mark.parametrize("answer_type,answer", [
    ("text", ""), ("numeric", 0), ("boolean", False), ("json", {}), ("list", []),
    ("subjective", None),
])
def test_standalone_loader_retains_valid_empty_truth(tmp_path, answer_type, answer):
    path = tmp_path / "truth.yaml"
    path.write_text(yaml.safe_dump({"cases": [
        {"id": "café", "ground_truth": {"answer_type": answer_type, "answer": answer}},
        {"id": "unlabeled"},
    ]}, allow_unicode=True), encoding="utf-8")
    truth = load_ground_truth(path)
    assert list(truth) == ["café"]
    assert truth["café"].answer == answer


def test_standalone_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "truth.yaml"
    path.write_text(yaml.safe_dump({"cases": [
        {"id": "same", "ground_truth": {"answer": "a"}},
        {"id": "same", "ground_truth": {"answer": "b"}},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Case 'same': duplicate ID"):
        load_ground_truth(path)


@pytest.fixture
def pooled_payload():
    runs = [make_run(evaluation_id="eval", repetition=i, results=[make_row()]) for i in (1, 2)]
    runs[0].results[0].passed = False
    return MultiRunResults("agent", "suite", 2, evaluation_id="eval", run_results=runs).to_dict()


@pytest.mark.parametrize("wrapped", [False, True])
def test_python_readers_accept_pooled_cli_payloads(tmp_path, pooled_payload, wrapped):
    data = {"results": pooled_payload, "success": False} if wrapped else pooled_payload
    original = json.dumps(data)
    html = render_html(data, tmp_path / "report.html").read_text(encoding="utf-8")
    ids = re.findall(r'id="(detail-[^"]+)"', html)
    assert len(ids) == len(set(ids)) == 2
    assert "(run 1)" in html and "(run 2)" in html
    assert "café" in html and "東京" in html
    xml = ET.parse(to_junit_xml(data, tmp_path / "report.xml"))
    cases = xml.findall(".//testcase")
    assert len({(case.get("name"), case.get("classname")) for case in cases}) == 2
    for repetition, case in enumerate(cases, 1):
        assert f"run {repetition}" in case.get("name")
        properties = {p.get("name"): p.get("value") for p in case.findall("properties/property")}
        assert properties == {
            "evaluation_id": "eval", "case_id": "café", "repetition": str(repetition),
            "metadata": "{}", "response_metadata": "{}",
        }
    assert len(xml.findall(".//failure")) == 1
    with to_csv(data, tmp_path / "report.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["passed"] for row in rows] == ["False", "True"]
    assert all(row["response"] == "東京" for row in rows)
    assert [(r["evaluation_id"], r["case_id"], r["repetition"]) for r in rows] == [
        ("eval", "café", "1"), ("eval", "café", "2"),
    ]
    report = format_report(data)
    assert "50% (1/2)" in report
    assert "| Case ID | Run |" in report
    assert "| café | 1 | FAIL |" in report
    assert "| café | 2 | PASS |" in report
    assert json.loads(format_report(data, fmt="json"))["results"] == pooled_payload
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_results(path) == load_baseline(path) == pooled_payload
    assert not compare_with_baseline(data, pooled_payload)["has_regression"]
    assert json.dumps(data) == original


@pytest.fixture
def legacy_payload():
    return {
        "agent": "Earlier Agent", "suite": "Earlier Suite",
        "summary": {"total": 1, "passed": 0, "failed": 1, "pass_rate": 0},
        "results": [{"test": "Earlier café", "reason": "Expected a greeting",
                     "output": "東京", "passed": False, "score": 0}],
    }


@pytest.mark.parametrize("wrapped", [False, True])
def test_cli_reports_preserve_legacy_names_and_content(tmp_path, legacy_payload, wrapped):
    data = {"success": False, "results": legacy_payload} if wrapped else legacy_payload
    original = json.dumps(data)
    path = tmp_path / "legacy.json"
    path.write_text(original, encoding="utf-8")
    result = CliRunner().invoke(cli, ["report", "--results", str(path),
        "--html", str(tmp_path / "report.html"), "--junit", str(tmp_path / "report.xml"),
        "--csv", str(tmp_path / "report.csv"),
    ])
    assert result.exit_code == 0, result.output
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    for value in ("Earlier Agent", "Earlier Suite", "Earlier café", "Expected a greeting", "東京"):
        assert value in html
    xml = ET.parse(tmp_path / "report.xml")
    assert xml.find("testsuite").get("name") == "Earlier Agent/Earlier Suite"
    case = xml.find(".//testcase")
    assert case.get("name") == "Earlier café"
    assert case.get("classname") == "Earlier Agent.Earlier Suite"
    assert case.find("failure").get("message") == "Expected a greeting"
    assert "東京" in case.find("failure").text
    assert case.find("system-out").text == "東京"
    assert case.find("properties") is None
    with (tmp_path / "report.csv").open(encoding="utf-8", newline="") as f:
        row, = csv.DictReader(f)
    assert (row["case_name"], row["score_reason"], row["response"]) == (
        "Earlier café", "Expected a greeting", "東京",
    )
    assert row["case_id"] == row["evaluation_id"] == row["repetition"] == ""
    assert load_results(path) == normalize_results(data)
    to_csv(data, tmp_path / "direct.csv")
    to_junit_xml(data, tmp_path / "direct.xml")
    render_html(data, tmp_path / "direct.html")
    assert (tmp_path / "direct.csv").read_bytes() == (tmp_path / "report.csv").read_bytes()
    assert (tmp_path / "direct.xml").read_bytes() == (tmp_path / "report.xml").read_bytes()
    assert json.dumps(data) == original


def test_alias_normalization_preserves_canonical_values_and_child_runs(legacy_payload):
    original = json.dumps(legacy_payload)
    canonical = {**legacy_payload, "agent_name": "", "suite_name": "Current Suite",
                 "results": [{**legacy_payload["results"][0], "case_name": "Current case",
                              "response": "", "score_reason": "Current reason"}]}
    data = {**canonical, "run_results": [legacy_payload, canonical]}
    normalized = normalize_results(data)
    assert normalized["agent_name"] == ""
    assert normalized["suite_name"] == "Current Suite"
    assert normalized["results"] == canonical["results"]
    assert normalized["run_results"][0]["results"][0]["response"] == "東京"
    assert normalized["run_results"][1] == canonical
    assert "case_id" not in normalized["run_results"][0]["results"][0]
    assert normalize_results(normalized) == normalized
    assert json.dumps(legacy_payload) == original


def test_exports_preserve_full_ids_and_distinguish_same_named_cases(tmp_path):
    evaluation_id = "eval-" + "e" * 600
    case_ids = ["café|<id>\n" + "c" * 600 + str(i) for i in (1, 2)]
    data = {"results": [{"evaluation_id": evaluation_id, "case_id": case_id,
                         "repetition": 1, "case_name": "Same name", "passed": True}
                        for case_id in case_ids]}
    with to_csv(data, tmp_path / "report.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["case_id"] for r in rows] == case_ids
    assert all(r["evaluation_id"] == evaluation_id and r["repetition"] == "1" for r in rows)
    cases = ET.parse(to_junit_xml(data, tmp_path / "report.xml")).findall(".//testcase")
    assert len({case.get("name") for case in cases}) == 2
    for case, case_id in zip(cases, case_ids, strict=True):
        properties = {p.get("name"): p.get("value") for p in case.findall("properties/property")}
        assert properties == {"evaluation_id": evaluation_id, "case_id": case_id, "repetition": "1"}
    markdown = format_report(data)
    assert "café&#124;&lt;id&gt;<br>" in markdown
    assert all(line.count("|") == 8 for line in markdown.splitlines() if "Same name" in line)


@pytest.mark.parametrize("side", ["baseline", "current"])
def test_diff_explains_missing_legacy_case_ids(legacy_payload, side):
    with pytest.raises(ValueError, match="requires a nonempty case_id"):
        diff_results(**{
            "baseline": {"results": []}, "current": {"results": []}, side: legacy_payload,
        })


@pytest.mark.parametrize("side", ["baseline", "current"])
def test_diff_rejects_pooled_results_in_either_input(pooled_payload, side):
    single = pooled_payload["run_results"][0]
    args = {"baseline": single, "current": single, side: {"results": pooled_payload}}
    with pytest.raises(ValueError, match="select a run from run_results"):
        diff_results(**args)
    diff = diff_results(single, pooled_payload["run_results"][1])
    assert len(diff.improvements) == 1


def test_cli_diff_explains_how_to_select_a_repetition(tmp_path, pooled_payload):
    path = tmp_path / "pooled.json"
    path.write_text(json.dumps({"results": pooled_payload}), encoding="utf-8")
    result = CliRunner().invoke(cli, ["diff", str(path), str(path)])
    assert result.exit_code == 1
    assert "select a run from run_results" in " ".join(result.output.split())
    assert not isinstance(result.exception, (AttributeError, KeyError))


@pytest.mark.parametrize("reader", [load_results, load_baseline])
def test_future_result_version_is_not_silently_read(tmp_path, reader):
    path = tmp_path / "future.json"
    path.write_text('{"results":{"schema_version":"9","results":[]}}', encoding="utf-8")
    if reader is load_baseline:
        assert reader(path) is None
    else:
        with pytest.raises(ValueError, match="Unsupported result schema_version"):
            reader(path)
