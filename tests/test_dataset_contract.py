"""Dataset identity, portable inputs, and saved evaluation provenance."""

import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner

from litmusai import (
    Agent,
    AgentResponse,
    All,
    AnyOf,
    AtLeast,
    Contains,
    Custom,
    DatasetInfo,
    Exact,
    GroundTruth,
    JsonPath,
    JsonSchema,
    JsonValid,
    MetricConfig,
    NotContains,
    Numeric,
    RegexMatch,
    SourceReference,
    TestCase,
    TestSuite,
    Weighted,
    evaluate,
    multi_evaluate,
)
from litmusai.cli.main import cli
from litmusai.exports import to_csv, to_junit_xml
from litmusai.reports import render_html
from litmusai.results import diff_results, load_results, normalize_results


def dataset():
    return TestSuite("support", [TestCase(
        id="例-1", task="classify", inputs={"text": "café", "count": 0, "active": False},
        ground_truth=GroundTruth(answer="billing"),
        metadata={"split": "test", "nested": {"empty": [], "null": None}},
        source=SourceReference(provider="langsmith", dataset_id="dataset-1",
                               example_id="example-1", trace_id="source-trace-1"),
    )], metrics=MetricConfig(task_type="classification", labels=["billing", "technical"]),
        dataset=DatasetInfo(id="support-1", revision="revision-2", metadata={"owner": "Zoë"},
                            source=SourceReference(provider="langsmith", dataset_id="dataset-1")))


@pytest.mark.parametrize("fmt", ["dict", "json", "yaml"])
def test_dataset_round_trip_preserves_inputs_labels_and_provenance(tmp_path, fmt):
    original = dataset()
    data = original.to_dict()
    if fmt == "dict":
        loaded = TestSuite.from_dict(data)
    else:
        path = tmp_path / f"dataset.{fmt}"
        getattr(original, f"to_{fmt}")(path)
        loaded = getattr(TestSuite, f"from_{fmt}")(path)
    assert loaded.to_dict() == data
    assert data["task_type"] == "classification"
    assert original.to_dict() == data
    assert loaded.dataset.id == "support-1"
    assert loaded.dataset.revision == "revision-2"
    assert loaded.cases[0].source.trace_id == "source-trace-1"
    assert loaded.cases[0].inputs["count"] == 0
    assert loaded.cases[0].inputs["active"] is False


def test_fingerprint_ignores_case_key_order_and_external_revision():
    original = dataset()
    original.add_case(TestCase(id="other", inputs={"b": 2, "a": 1},
                               ground_truth=GroundTruth(answer="technical")))
    fingerprint = original.dataset_info.fingerprint
    original.cases.reverse()
    original.cases[0].inputs = {"a": 1, "b": 2}
    original.dataset = DatasetInfo(id="different-provider-id", revision="other-revision")
    assert original.dataset_info.fingerprint == fingerprint


@pytest.mark.parametrize("change", ["input", "truth", "metadata", "case-id", "metrics"])
def test_fingerprint_changes_with_content(change):
    original = dataset()
    fingerprint = original.dataset_info.fingerprint
    if change == "input":
        original.cases[0].inputs["text"] = "different"
    elif change == "truth":
        original.cases[0].ground_truth.answer = "technical"
    elif change == "metadata":
        original.cases[0].metadata["split"] = "train"
    elif change == "case-id":
        original.cases[0].id = "another-case"
    else:
        original.metrics = MetricConfig(task_type="classification", labels=["billing", "other"])
    assert original.dataset_info.fingerprint != fingerprint


def test_fingerprint_is_data_identity_not_callable_implementation_identity():
    original = dataset()
    fingerprint = original.dataset_info.fingerprint
    original.cases[0].assertions = [Custom(lambda _: False)]
    original.cases[0].timeout_seconds = 10
    assert original.dataset_info.fingerprint == fingerprint
    with pytest.raises(ValueError, match="case '例-1'.*Custom cannot be exported"):
        original.to_dict()


@pytest.mark.parametrize("fmt", ["json", "yaml"])
def test_rejected_dataset_export_preserves_existing_file(tmp_path, fmt):
    suite = dataset()
    suite.cases[0].assertions = [Custom(lambda _: True)]
    path = tmp_path / f"dataset.{fmt}"
    path.write_text("existing dataset", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be exported"):
        getattr(suite, f"to_{fmt}")(path)
    assert path.read_text(encoding="utf-8") == "existing dataset"


def test_local_dataset_has_fingerprint_without_inventing_external_identity():
    original = TestSuite("local", [TestCase(id="one", task="hello")])
    info = original.to_dict()["dataset"]
    assert info["id"] is None and info["revision"] is None
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", info["fingerprint"])


def test_reader_rejects_stale_fingerprint_and_unsupported_version():
    data = dataset().to_dict()
    data["cases"][0]["inputs"]["text"] = "modified"
    with pytest.raises(ValueError, match="fingerprint does not match"):
        TestSuite.from_dict(data)
    data["schema_version"] = "9.0"
    with pytest.raises(ValueError, match="Unsupported suite schema_version"):
        TestSuite.from_dict(data)


@pytest.mark.parametrize("malformed", [
    {"cases": "not a list"}, {"cases": [False]}, {"cases": [{"task": "missing ID"}]},
    {"cases": [{"id": "same"}, {"id": "same"}]},
    {"cases": [{"id": "bad", "inputs": []}]},
    {"cases": [{"id": "bad", "metadata": []}]},
    {"cases": [{"id": "bad", "source": {"provider": " "}}]},
    {"dataset": {"revision": " "}}, {"dataset_revision": "misspelled-location"},
    {"task_type": "retrieval"},
])
def test_dataset_reader_reports_malformed_records(malformed):
    with pytest.raises(ValueError):
        TestSuite.from_dict(malformed)


def test_declared_task_type_must_match_metric_configuration():
    data = dataset().to_dict()
    data["task_type"] = "extraction"
    with pytest.raises(ValueError, match="task_type must be 'classification'"):
        TestSuite.from_dict(data)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), {1: "ambiguous key"}, (1, 2)])
async def test_non_json_inputs_fail_before_agent_execution(invalid):
    suite = TestSuite("invalid", [TestCase(id="bad-input", inputs={"value": invalid})])
    fn = Mock(return_value="hello")
    with pytest.raises(ValueError, match="case 'bad-input' inputs"):
        await evaluate(Agent.from_function(fn), suite, verbose=False)
    fn.assert_not_called()


@pytest.mark.parametrize("fmt", ["json", "yaml"])
def test_declarative_assertions_round_trip_all_options(tmp_path, fmt):
    assertions = [
        Exact("Hello", case_sensitive=True, strip=False),
        Contains(["Hello"], mode="any", case_sensitive=True),
        NotContains(["BAD"], case_sensitive=True),
        Numeric(42, tolerance=2, relative_tolerance=.1),
        RegexMatch("Hello", flags=0, full_match=True), JsonValid(),
        JsonSchema({"type": "object", "required": ["count"]}),
        JsonPath("$.count", expected=0, operator="eq"),
        All(Contains(["hello"]), NotContains(["bad"])), AnyOf(Exact("a"), Exact("b")),
        AtLeast(1, [Exact("a"), Exact("b")]),
        Weighted([(Exact("a"), .8), (Exact("b"), .2)], threshold=.75),
    ]
    suite = TestSuite("checks", [TestCase(id="one", assertions=assertions)])
    path = tmp_path / f"checks.{fmt}"
    getattr(suite, f"to_{fmt}")(path)
    loaded = getattr(TestSuite, f"from_{fmt}")(path)
    assert loaded.to_dict() == suite.to_dict()
    for original, restored in zip(assertions, loaded.cases[0].assertions):
        for response in ["a", "b", "Hello", " hello ", "42", '{"count":0}', "BAD"]:
            assert restored.check(response) == original.check(response)


async def test_structured_inputs_and_provenance_are_snapshotted_across_repetitions(tmp_path):
    suite = dataset()
    initial_fingerprint = suite.dataset_info.fingerprint
    received = []
    response_metadata = {"new_evaluation_trace_id": "evaluation-trace", "nested": [1]}

    async def agent(task, *, inputs):
        received.append(deepcopy(inputs))
        inputs["count"] = 99
        suite.cases[0].inputs["count"] = 100
        suite.cases[0].metadata["nested"]["empty"].append("changed")
        suite.dataset.metadata["owner"] = "changed"
        return AgentResponse(output="billing", metadata=response_metadata)

    multi = await multi_evaluate(Agent.from_function(agent), suite, runs=2, verbose=False)
    response_metadata["nested"].append(2)
    data = load_results(multi.save(tmp_path / "results.json"))
    assert [item["count"] for item in received] == [0, 0]
    assert data["dataset"]["fingerprint"] == initial_fingerprint
    assert data["dataset"]["metadata"]["owner"] == "Zoë"
    for run in data["run_results"]:
        assert run["dataset"] == data["dataset"]
    for row in data["results"]:
        assert row["inputs"]["count"] == 0
        assert row["metadata"]["nested"]["empty"] == []
        assert row["source"]["example_id"] == "example-1"
        assert row["source"]["trace_id"] == "source-trace-1"
        assert row["response_metadata"]["nested"] == [1]
        assert row["evaluation_id"] != row["source"]["trace_id"]
    assert data["metrics"]["accuracy"]["value"] == 1


@pytest.mark.parametrize("inputs", [None, {}])
async def test_only_explicit_structured_inputs_add_a_keyword_argument(inputs):
    calls = []

    def agent(task, **kwargs):
        calls.append((task, kwargs))
        return "ok"

    suite = TestSuite("inputs", [TestCase(id="one", task="hello", inputs=inputs)])
    await evaluate(Agent.from_function(agent), suite, verbose=False)
    assert calls == [("hello", {} if inputs is None else {"inputs": {}})]


@pytest.mark.parametrize("adapter", [
    "from_openai_chat", "from_azure", "from_openai_agent", "from_cli",
])
@pytest.mark.parametrize("inputs", [{}, {"text": "café", "count": 0, "active": False}])
async def test_text_adapters_reject_structured_cases_before_external_calls(
    monkeypatch, adapter, inputs,
):
    unexpected_call = AsyncMock(side_effect=AssertionError("Unexpected external call"))
    monkeypatch.setattr("httpx.AsyncClient.post", unexpected_call)
    monkeypatch.setattr("asyncio.create_subprocess_shell", unexpected_call)
    monkeypatch.setattr("asyncio.create_subprocess_exec", unexpected_call)
    monkeypatch.setitem(sys.modules, "openai.agents",
                        SimpleNamespace(Runner=SimpleNamespace(run=unexpected_call)))
    if adapter == "from_openai_chat":
        agent = Agent.from_openai_chat()
    elif adapter == "from_azure":
        agent = Agent.from_azure(resource="test", deployment="test", api_key="test-key")
    elif adapter == "from_openai_agent":
        agent = Agent.from_openai_agent(object())
    else:
        agent = Agent.from_cli("unused-command")
    suite = dataset()
    suite.cases[0].inputs = inputs
    data = (await evaluate(agent, suite, verbose=False)).to_dict()
    row, = data["results"]
    assert row["passed"] is False
    assert f"{adapter} does not support structured inputs" in row["error"]
    assert "Agent.from_function" in row["error"]
    assert row["inputs"] == inputs
    assert data["metrics"]["accuracy"]["value"] == 0
    unexpected_call.assert_not_called()


async def test_saved_result_dict_does_not_alias_ground_truth_or_model_parameters():
    suite = TestSuite("snapshot", [TestCase(id="one", ground_truth=GroundTruth(
        answer={"items": []}, answer_type="json"))], metrics=MetricConfig(task_type="extraction"))
    parameters = {"nested": [0]}
    agent = Agent(lambda _: '{"items":[]}', model_params=parameters)
    result = await evaluate(agent, suite, verbose=False)
    before = deepcopy(result.to_dict())
    parameters["nested"].append(1)
    changed = result.to_dict()
    changed["results"][0]["ground_truth"]["answer"]["items"].append("changed")
    changed["config"]["model_params"]["nested"].append(2)
    assert result.to_dict() == before


async def test_mixing_dataset_revisions_in_multi_results_is_rejected():
    suite = dataset()
    result = await multi_evaluate(Agent.from_function(lambda _, inputs: "billing"), suite,
                                  runs=2, verbose=False)
    changed = replace(result.run_results[1], dataset=DatasetInfo(id="other", revision="v9"))
    with pytest.raises(ValueError, match="different dataset revisions"):
        replace(result, run_results=[result.run_results[0], changed]).to_dict()


@pytest.mark.parametrize("fmt", ["yaml", "json"])
def test_cli_preserves_provenance_through_logs_readers_reports_and_diff(tmp_path, fmt):
    suite = dataset()
    suite.cases[0].inputs["text"] = '<script>alert("unsafe")</script>'
    path = tmp_path / f"dataset.{fmt}"
    getattr(suite, f"to_{fmt}")(path)
    agent_path = tmp_path / "agent.py"
    full_response = "billing" + "x" * 3000
    agent_path.write_text(
        'from litmusai import AgentResponse\n'
        'def agent(task, *, inputs):\n'
        '    assert inputs["count"] == 0 and inputs["active"] is False\n'
        f'    return AgentResponse(output={full_response!r}, metadata={{"trace": "new-trace"}})\n',
        encoding="utf-8")
    output = tmp_path / "results.json"
    result = CliRunner().invoke(cli, [
        "run", "-s", str(path), "-a", f"{agent_path}:agent", "--format", "json",
        "--runs", "2", "--output", str(output), "--log-dir", str(tmp_path / "logs"),
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload == json.loads(output.read_text(encoding="utf-8"))
    data = load_results(output)
    assert data == load_results(next((tmp_path / "logs").glob("*.json")))
    assert data["dataset"] == suite.dataset_info.model_dump(mode="json")
    assert data["results"][0]["response"] == full_response
    assert len(diff_results(data["run_results"][0], data["run_results"][1]).cases) == 1
    report = render_html(data, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Dataset provenance" in report and "source-trace-1" in report
    assert "&lt;script&gt;" in report and '<script>alert("unsafe")</script>' not in report
    csv_path = to_csv(data, tmp_path / "results.csv")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["dataset_id"] == "support-1"
    assert rows[0]["dataset_revision"] == "revision-2"
    assert rows[0]["dataset_fingerprint"] == data["dataset"]["fingerprint"]
    assert json.loads(rows[0]["dataset_metadata"]) == {"owner": "Zoë"}
    assert json.loads(rows[0]["dataset_source"]) == suite.dataset.source.model_dump(mode="json")
    assert json.loads(rows[0]["inputs"]) == suite.cases[0].inputs
    assert json.loads(rows[0]["metadata"]) == data["results"][0]["metadata"]
    assert json.loads(rows[0]["source"])["example_id"] == "example-1"
    assert rows[0]["response"] == full_response
    xml = ET.parse(to_junit_xml(data, tmp_path / "results.xml"))
    props = {p.get("name"): p.get("value") for p in xml.findall(".//testsuite/properties/property")}
    assert json.loads(props["dataset"])["fingerprint"] == data["dataset"]["fingerprint"]
    case_props = {p.get("name"): p.get("value")
                  for p in xml.findall(".//testcase/properties/property")}
    assert json.loads(case_props["source"])["trace_id"] == "source-trace-1"
    assert xml.find(".//testcase/system-out").text == full_response


@pytest.mark.parametrize("provenance,expected_id,expected_metadata", [
    pytest.param({}, "", "", id="legacy-missing-dataset"),
    pytest.param({"dataset": None}, "", "", id="null-dataset"),
    pytest.param({"dataset": {}}, "", "{}", id="empty-dataset"),
    pytest.param({"dataset": {"id": "support-1"}}, "support-1", "{}", id="identity-only"),
    pytest.param({"dataset": DatasetInfo().model_dump(mode="json")}, "", "{}",
                 id="empty-metadata-null-source"),
])
def test_csv_dataset_fields_handle_missing_provenance(
    tmp_path, provenance, expected_id, expected_metadata,
):
    data = {**provenance, "results": [{"case_id": "one", "response": "billing"}]}
    path = to_csv(data, tmp_path / "results.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        row, = csv.DictReader(handle)
    assert row["dataset_id"] == expected_id
    assert row["dataset_revision"] == row["dataset_fingerprint"] == ""
    assert row["dataset_metadata"] == expected_metadata
    assert row["dataset_source"] == ""


@pytest.mark.parametrize("inputs", [None, {}])
async def test_csv_case_fields_leave_missing_values_blank_and_keep_empty_mappings(tmp_path, inputs):
    suite = TestSuite("empty", [TestCase(id="one", task="hello", inputs=inputs)])
    result = await evaluate(Agent.from_function(lambda _, **kwargs: "ok"), suite, verbose=False)
    path = to_csv(result.to_dict(), tmp_path / "results.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        row, = csv.DictReader(handle)
    assert row["source"] == row["ground_truth"] == ""
    assert row["inputs"] == ("" if inputs is None else "{}")
    assert row["metadata"] == row["response_metadata"] == "{}"


def test_legacy_result_identity_stays_missing_and_unicode_errors_survive(tmp_path):
    data = {"success": False, "results": {"agent": "legacy", "suite": "support", "results": [
        {"test": "café", "output": "échec", "reason": "erreur", "success": False,
         "error": "délai dépassé", "passed": False, "score": 0},
    ]}}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    normalized = load_results(path)
    assert "dataset" not in normalized
    assert "case_id" not in normalized["results"][0]
    assert normalized["results"][0]["response"] == "échec"
    assert normalized["results"][0]["error"] == "délai dépassé"
    with pytest.raises(ValueError, match="nonempty case_id"):
        diff_results(normalized, normalized)


def test_result_reader_copies_new_metadata_without_mutating_input():
    data = {"dataset": dataset().dataset_info.model_dump(mode="json"), "results": [
        {"case_id": "one", "inputs": {"values": [0, False, None]},
         "metadata": {"nested": []}, "source": {"provider": "local", "example_id": "one"}},
    ]}
    before = deepcopy(data)
    normalized = normalize_results(data)
    normalized["results"][0]["metadata"]["nested"].append("changed")
    assert data == before
